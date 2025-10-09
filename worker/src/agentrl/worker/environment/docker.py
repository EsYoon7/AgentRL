from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from random import shuffle
from typing import Union, List, Tuple, Dict, TypedDict, Optional, TYPE_CHECKING

from ._base import EnvironmentController
from .state import create_state_provider

try:
    import aiodocker
    from aiodocker.containers import DockerContainer
except Exception:
    aiodocker = None

if TYPE_CHECKING:
    from aiodocker.stream import Stream
    from ._base import EnvironmentDelegation

LABEL_MANAGED_BY = 'agentrl.managed-by'
LABEL_MANAGED_BY_VALUE = 'agentrl'
LABEL_TASK_NAME = 'agentrl.task-name'
LABEL_SUBTYPE_NAME = 'agentrl.subtype-name'
LABEL_DEPENDS_ON = 'agentrl.depends-on'
LABEL_EXCLUSIVE = 'agentrl.exclusive'

logger = logging.getLogger(__name__)


class SessionData(TypedDict):
    containers: Dict[str, str]
    exclusive_containers: List[str]


class DockerEnvironmentController(EnvironmentController):
    """
    This driver manages Docker containers for tasks through the Docker API.

    Important Notice:
      To enable communication between the worker and each environment,
      worker containers and environment containers must be in the same Docker network.
      This network must be a custom bridge network, not the default network.
      The name of this network must be set in the `network_name` parameter.
    """

    def __init__(self,
                 delegation: EnvironmentDelegation,
                 connection: dict,
                 network_name: str,
                 state_driver: str,
                 state_options: dict = None):
        super().__init__(delegation)
        self.task_name = delegation.get_name()
        self.valid_subtypes = delegation.get_subtypes()

        if not aiodocker:
            raise ImportError('aiodocker client library is not installed.')

        self._client = None
        self._client_connection_params = connection
        self.network_name = network_name

        self.client_id = str(uuid.uuid4())
        if state_options is None:
            state_options = {}
        self.state = create_state_provider(
            driver=state_driver,
            prefix=f'agentrl:{self.task_name}:{self.network_name}',
            **state_options
        )

        self.shells: Dict[str, Stream] = {}

    async def _get_client(self) -> aiodocker.Docker:
        if self._client is None:
            self._client = aiodocker.Docker(**self._client_connection_params)
        return self._client

    async def start_session(self, subtypes: Union[List[str], str], immutable: bool = True, **kwargs) -> Tuple[str, Dict[str, str], Dict[str, str]]:
        subtypes = subtypes if isinstance(subtypes, list) else [subtypes]
        for subtype in subtypes:
            assert subtype in self.valid_subtypes

        session_id = str(uuid.uuid4())
        containers_allocated: Dict[str, DockerContainer] = {}
        containers_allocated_exclusive: Dict[str, DockerContainer] = {}
        non_exclusive_subtypes = []

        for subtype in subtypes:
            usage_limit = self.delegation.get_reuse_limit(subtype)
            if usage_limit == 1 or (usage_limit != 0 and not immutable):
                # exclusive allocation, no need to lock
                container = await self.create_container(subtype, exclusive=True, **kwargs)
                await self.state.allocate_container(container.id, session_id)
                containers_allocated[subtype] = container
                containers_allocated_exclusive[subtype] = container
            else:
                non_exclusive_subtypes.append(subtype)

        if len(non_exclusive_subtypes) > 0:
            async with self.state.with_lock('docker', f'{self.client_id}.{session_id}'):
                containers = await self._identify_containers(non_exclusive_subtypes)
                for subtype in non_exclusive_subtypes:
                    existing_containers = containers.get(subtype, [])

                    # shuffle candidates to balance load
                    shuffle(existing_containers)

                    # determine concurrency limit for each container
                    concurrency_limit = self.delegation.get_concurrency_limit(subtype)
                    usage_limit = self.delegation.get_reuse_limit(subtype)

                    for container in existing_containers:
                        if usage_limit > 0 and await self.state.container_total_uses(container.id) >= usage_limit:
                            # container has reached its reuse limit, skip it
                            continue

                        if concurrency_limit > 0 and await self.state.container_current_uses(container.id) >= concurrency_limit:
                            # container concurrency is full, skip it
                            continue

                        containers_allocated[subtype] = container
                        await self.state.allocate_container(container.id, session_id)
                        break
                    else:
                        # no suitable container found, create a new one
                        container = await self.create_container(subtype, exclusive=False, **kwargs)
                        await self.state.allocate_container(container.id, session_id)
                        containers_allocated[subtype] = container

                        if usage_limit > 1:
                            # create one more container for future use
                            await self.create_container(subtype, exclusive=False, **kwargs)

        if self.delegation.has_homepage():
            # create dedicated homepage container for the session
            homepage_subtype = self.delegation.get_homepage_subtype()
            homepage_envs = self.delegation.get_homepage_envs({
                subtype: self.get_container_url(containers_allocated, subtype)
                for subtype in containers_allocated.keys()
            })
            homepage_container = await self.create_container(homepage_subtype, homepage_envs, exclusive=True, **kwargs)
            await self.state.allocate_container(homepage_container.id, session_id)
            containers_allocated[homepage_subtype] = homepage_container
            containers_allocated_exclusive[homepage_subtype] = homepage_container

        # save allocated container ids to the session
        await self.state.store_session(session_id, {
            'containers': {
                subtype: container.id
                for subtype, container in containers_allocated.items()
            },
            'exclusive_containers': [
                container.id
                for subtype, container in containers_allocated_exclusive.items()
            ]
        })

        # log allocations
        for subtype, container in containers_allocated.items():
            logger.info(f'Allocated {subtype} container {container.id} to session {session_id}')

        # release the lock, while wait for containers to be healthy
        logger.info('Waiting for containers to be healthy')
        await self._wait_for_health(*containers_allocated.values())

        # return session_id, container ids and environment urls
        return session_id, {
            subtype: container.id
            for subtype, container in containers_allocated.items()
        }, {
            subtype: self.get_container_url(containers_allocated, subtype)
            for subtype in containers_allocated.keys()
        }

    async def renew_session(self, session_id: str):
        await self.state.renew_session(session_id)
        session: Optional[SessionData] = await self.state.get_session(session_id)
        if session:
            for container in session.get('containers', {}).values():
                await self.state.renew_container(container, session_id)

    async def end_session(self, session_id: str):
        client = await self._get_client()

        session: Optional[SessionData] = await self.state.get_session(session_id)
        if session:
            async with self.state.with_lock('docker', f'{self.client_id}.{session_id}'):
                exclusive_containers = session.get('exclusive_containers', [])
                for container_id in exclusive_containers:
                    container = client.containers.container(container_id)
                    await self.delete_container(container)

                for subtype, container_id in session.get('containers', {}).items():
                    if container_id in exclusive_containers:
                        continue  # already deleted
                    await self.state.release_container(container_id, session_id)
                    logger.info(f'Released container {container_id}')

        await self.state.delete_session(session_id)

    async def execute_command(self, environment_id: str, command: Union[str, List[str]], timeout: int = 30) -> Tuple[int, bytes, bytes]:
        client = await self._get_client()
        container = client.containers.container(environment_id)
        exec_ = await container.exec(command)

        stdout_data = bytearray()
        stderr_data = bytearray()
        async with exec_.start(timeout=timeout, detach=False) as stream:
            while True:
                message = await stream.read_out()
                if message is None:
                    break
                if message.stream == 1:
                    stdout_data.extend(message.data)
                elif message.stream == 2:
                    stderr_data.extend(message.data)

        exit_code = (await exec_.inspect()).get('ExitCode', 0)
        return exit_code, bytes(stdout_data), bytes(stderr_data)

    async def create_shell(self, environment_id: str, shell: str = '/bin/bash --login'):
        client = await self._get_client()
        container = client.containers.container(environment_id)

        exec_ = await container.exec(shell, stdin=True, tty=True)
        stream = exec_.start(detach=False)
        self.shells[container.id] = stream
        await stream._init()

        # consume first prompt
        async def read_until_prompt():
            while True:
                message = await stream.read_out()
                if message is None:
                    break
                if re.search(b'\x1b.+@.+[#|$] ', message.data):
                    break
        await asyncio.wait_for(read_until_prompt(), 5)

    async def execute_shell(self, environment_id: str, command: str, timeout: int = 30) -> bytes:
        client = await self._get_client()
        container = client.containers.container(environment_id)

        if container.id not in self.shells:
            await self.create_shell(environment_id)
        stream = self.shells[container.id]

        await stream.write_in(command.encode('utf-8') + b'\n')

        async def read_until_prompt():
            data = bytearray()
            ignored_first_line = False
            while True:
                message = await stream.read_out()
                if message is None:
                    break
                if ignored_first_line:
                    data.extend(message.data)
                else:
                    ignored_first_line = True
                if re.search(b'\x1b.+@.+[#|$] ', message.data):
                    break
            return bytes(data)

        return await asyncio.wait_for(read_until_prompt(), timeout)

    async def get_env_variables(self, environment_id: str) -> Dict[str, str]:
        client = await self._get_client()
        container = client.containers.container(environment_id)
        await container.show()

        if not container._container:
            return {}
        if not container._container.get('Config', {}).get('Env'):
            return {}

        env_vars = {}
        for env in container._container['Config']['Env']:
            key, value = env.split('=', 1)
            env_vars[key] = value

        return env_vars

    async def background_task(self):
        while True:
            try:
                if await self.state.acquire_lock('background', self.client_id, 120):
                    try:
                        await self._clean_containers()
                    except:
                        logger.warning('Error while cleaning containers', exc_info=True)
                    await self.state.release_lock('background', self.client_id)
            except:
                logger.warning('Error in background task', exc_info=True)
            await asyncio.sleep(10)

    async def _identify_containers(self, subtypes: Optional[List[str]] = None) -> Dict[str, List[DockerContainer]]:
        client = await self._get_client()
        return {
            subtype: await client.containers.list(filters=json.dumps({
                'label': [
                    f'{LABEL_MANAGED_BY}={LABEL_MANAGED_BY_VALUE}',
                    f'{LABEL_TASK_NAME}={self.task_name}',
                    f'{LABEL_SUBTYPE_NAME}={subtype}',
                    f'{LABEL_EXCLUSIVE}=false'
                ],
                'status': ['running'],
                'health': ['starting', 'healthy', 'none'],
                'network': [self.network_name]  # only manage containers in the same network
            }))
            for subtype in subtypes or self.valid_subtypes
        }

    async def _wait_for_health(self, *containers: Union[DockerContainer, str]):
        client = await self._get_client()
        container_ids = [c.id if isinstance(c, DockerContainer) else c for c in containers]
        while True:
            not_started_containers = await client.containers.list(filters=json.dumps({
                'id': container_ids,
                'status': ['created'],
                'network': [self.network_name]  # only manage containers in the same network
            }))
            if len(not_started_containers) > 0:
                await asyncio.sleep(1)
                continue

            unhealthy_containers = await client.containers.list(filters=json.dumps({
                'id': container_ids,
                'health': ['starting', 'unhealthy'],
                'network': [self.network_name]  # only manage containers in the same network
            }))
            if len(unhealthy_containers) == 0:
                break
            await asyncio.sleep(1)

    async def create_container(self, subtype: str, extra_envs: Dict[str, str] = None, exclusive: Optional[bool] = None, **kwargs) -> DockerContainer:
        client = await self._get_client()

        if not extra_envs:
            extra_envs = {}

        if exclusive is None:
            exclusive = self.delegation.get_reuse_limit(subtype) == 1

        # generate container name
        container_name = f'{subtype.replace("_", "-").lower()}-{uuid.uuid4().hex[:8]}'

        attrs = {
            'Name': container_name,
            'Env': {},
            'Labels': {
                LABEL_MANAGED_BY: LABEL_MANAGED_BY_VALUE,
                LABEL_TASK_NAME: self.task_name,
                LABEL_SUBTYPE_NAME: subtype,
                LABEL_EXCLUSIVE: str(exclusive).lower(),
            },
            'HostConfig': {
                'AutoRemove': True,
                'Init': True,
                'NetworkMode': self.network_name
            }
        }

        # delegate container configuration
        attrs = await self.delegation.create_docker_container(attrs, subtype, **kwargs)
        if 'Image' not in attrs:
            attrs['Image'] = self.delegation.get_container_images()[subtype]

        # override extra envs
        for k, v in extra_envs.items():
            attrs['Env'][k] = v

        # transform attrs to required format
        attrs['Env'] = [
            f'{k}={v}'
            for k, v in attrs['Env'].items()
        ]
        if 'Name' in attrs:
            container_name = attrs['Name']
            del attrs['Name']

        # create container
        try:
            container = await client.containers.create(attrs, name=container_name)
        except aiodocker.DockerError as e:
            if e.status == 404:
                logger.warning(f'Image {attrs["Image"]} is not found, pulling it to try again...')
                try:
                    await asyncio.wait_for(client.images.pull(attrs['Image']), 120)
                except asyncio.TimeoutError:
                    logger.error(f'Timeout while pulling image {attrs["Image"]}')
                    raise e
                container = await client.containers.create(attrs, name=container_name)
            else:
                raise

        logger.debug(f'Created container {container_name} with {attrs=}')

        # start container
        await container.start()

        # retrieve container info
        await container.show()

        # call post-create hook in new coroutine to prevent blocking lock
        asyncio.create_task(self.post_create_container(subtype, container))

        return container

    async def post_create_container(self, subtype: str, container: DockerContainer):
        if 'post_create_docker_container' not in self.delegation.__class__.__dict__:
            return  # not implemented by the delegation

        await self._wait_for_health(container)
        await self.delegation.post_create_docker_container(subtype, container.id, self.get_container_url(container, subtype))

    async def delete_container(self, container: DockerContainer):
        client = await self._get_client()

        if container.id in self.shells and self.shells[container.id]:
            await self.shells[container.id].close()
            del self.shells[container.id]

        try:
            # try to get container labels
            await container.show()
        except:
            pass
        if 'Labels' in container._container and LABEL_DEPENDS_ON in container['Labels']:
            for dep in container['Labels'][LABEL_DEPENDS_ON].split(','):
                dep_container = client.containers.container(dep)
                await self.delete_container(dep_container)

        try:
            await container.delete(v=True, force=True)
        except:
            pass

        await self.state.remove_container(container.id)
        logger.info(f'Deleted container {container.id}')

    async def _clean_containers(self):
        client = await self._get_client()

        # remove unused exclusive containers
        # no need to lock
        containers = await client.containers.list(filters=json.dumps({
            'label': [
                f'{LABEL_MANAGED_BY}={LABEL_MANAGED_BY_VALUE}',
                f'{LABEL_TASK_NAME}={self.task_name}',
                f'{LABEL_EXCLUSIVE}=true'
            ],
            'network': [self.network_name]  # only manage containers in the same network
        }))
        for container in containers:
            if not await self.state.container_is_allocated(container.id):
                await self.delete_container(container)

        # remove not used unhealthy non-exclusive containers
        # no need to lock
        containers = await client.containers.list(filters=json.dumps({
            'label': [
                f'{LABEL_MANAGED_BY}={LABEL_MANAGED_BY_VALUE}',
                f'{LABEL_TASK_NAME}={self.task_name}',
                f'{LABEL_EXCLUSIVE}=false'
            ],
            'health': ['unhealthy'],
            'network': [self.network_name]  # only manage containers in the same network
        }))
        for container in containers:
            if await self.state.container_is_allocated(container.id):
                continue
            await self.delete_container(container)

        # remove containers that have reached their reuse limit and is not currently allocated
        for subtype, containers in (await self._identify_containers()).items():
            usage_limit = self.delegation.get_reuse_limit(subtype)
            if usage_limit == 0:
                continue
            for container in containers:
                if await self.state.container_is_allocated(container.id):
                    continue
                if await self.state.container_total_uses(container.id) < usage_limit:
                    continue
                await self.delete_container(container)

    def get_container_url(self, containers: Union[Dict[str, Union[DockerContainer, str]], DockerContainer, str], subtype: str) -> str:
        if isinstance(containers, DockerContainer):
            # if input is a single container instance, get name / ip from it;
            container = containers
        elif isinstance(containers, dict):
            # if input is a dict, get the container of the given subtype;
            container = containers[subtype]
        else:
            container = containers

        ip = self.get_container_ip(container)
        port = self.delegation.get_service_port(subtype)
        if not port:
            return ip
        if port == 80:
            return f'http://{ip}'
        return f'http://{ip}:{port}'

    def get_container_ip(self, container: DockerContainer) -> str:
        return container['NetworkSettings']['Networks'][self.network_name]['IPAddress']

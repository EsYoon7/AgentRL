from __future__ import annotations

import asyncio
import logging
import shlex
import uuid
from random import shuffle
from typing import TYPE_CHECKING, TypedDict, Dict, List, Optional, Any, Union, Tuple, Callable, Awaitable, TypeVar

import aiohttp
import kubernetes_asyncio as k8s

from ._base import EnvironmentController
from ._const import *
from ._typings import StateDriver
from .state import create_state_provider

if TYPE_CHECKING:
    from ._delegation import EnvironmentDelegation

logger = logging.getLogger(__name__)

T = TypeVar('T')


class SessionData(TypedDict):
    pods: Dict[str, str]
    exclusive_pods: List[str]


class K8sEnvironmentController(EnvironmentController):
    """
    This driver manages Kubernetes pods for tasks through the API.

    A dedicated namespace should be specified for each task to ensure isolation and performance.
    """

    def __init__(self,
                 delegation: EnvironmentDelegation,
                 connection: Optional[dict],
                 namespace: str,
                 state_driver: StateDriver,
                 state_options: Optional[Dict[str, Any]] = None):
        super().__init__(delegation)
        self.task_name = delegation.get_name()
        self.valid_subtypes = delegation.get_subtypes()

        self._client: Optional[k8s.client.ApiClient] = None
        self._ws_client: Optional[k8s.stream.WsApiClient] = None
        self._kube_config = connection
        self.namespace = namespace
        self.state_driver = state_driver
        self.state_options = state_options or {}

        if state_options is None:
            state_options = {}
        self.state = create_state_provider(
            driver=state_driver,
            prefix=f'{self.namespace}:{self.task_name}',
            **state_options
        )

        self._shells: Dict[str, Dict[str, Any]] = {}
        self._pod_cache: Dict[str, Dict[str, k8s.client.V1Pod]] = {}
        self._pod_cache_initialized = False
        self._pod_cache_resource_version: Optional[str] = None
        self._pod_watch_task: Optional[asyncio.Task] = None
        self._pod_cache_lock: Optional[asyncio.Lock] = None
        self._retryable_statuses = {0, 500, 502, 503, 504}
        self._shell_keepalive_interval = 25
        self._shell_receive_timeout = 10

    async def _get_client(self) -> k8s.client.ApiClient:
        if self._client is not None:
            pool = getattr(self._client.rest_client, 'pool_manager', None)
            if pool is not None and getattr(pool, 'closed', False):
                await self._client.close()
                self._client = None

        if self._client is None:
            if self._kube_config is None:
                k8s.config.load_incluster_config()
            else:
                await k8s.config.load_config(**self._kube_config)

            self._client = k8s.client.ApiClient()

        return self._client

    async def _get_ws_client(self) -> k8s.stream.WsApiClient:
        if self._ws_client is not None:
            pool = getattr(self._ws_client.rest_client, 'pool_manager', None)
            if pool is not None and getattr(pool, 'closed', False):
                await self._ws_client.close()
                self._ws_client = None

        if self._ws_client is None:
            if self._kube_config is None:
                k8s.config.load_incluster_config()
            else:
                await k8s.config.load_config(**self._kube_config)

            self._ws_client = k8s.stream.WsApiClient()

        return self._ws_client

    async def _reset_client(self):
        if self._client is not None:
            try:
                await self._client.close()
            except Exception:
                logger.warning('Error closing k8s ApiClient during reset', exc_info=True)
            self._client = None
        await self._reset_ws_client()

    async def _reset_ws_client(self):
        if self._ws_client is not None:
            try:
                await self._ws_client.close()
            except Exception:
                logger.warning('Error closing k8s WsApiClient during reset', exc_info=True)
            self._ws_client = None

    async def _api_call(self,
                        operation: Callable[[k8s.client.CoreV1Api], Awaitable[T]],
                        description: str,
                        retry: bool = True) -> T:
        attempt = 0
        max_attempts = 2 if retry else 1
        while True:
            client = await self._get_client()
            api = k8s.client.CoreV1Api(client)
            try:
                return await operation(api)
            except k8s.client.exceptions.ApiException as exc:
                should_retry = exc.status in self._retryable_statuses and attempt < max_attempts - 1
                if not should_retry:
                    raise
                attempt += 1
                logger.warning('K8s %s failed with status %s, reconnecting (attempt %s/%s)',
                               description, exc.status, attempt + 1, max_attempts)
                await self._reset_client()
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if attempt >= max_attempts - 1:
                    raise
                attempt += 1
                logger.warning('K8s %s transport error %s, reconnecting (attempt %s/%s)',
                               description, exc, attempt + 1, max_attempts)
                await self._reset_client()

    async def _api(self,
                   method: str,
                   *args,
                   retry: bool = True,
                   description: Optional[str] = None,
                   use_ws: bool = False,
                   **kwargs):
        caller = self._ws_call if use_ws else self._api_call

        async def operation(api: k8s.client.CoreV1Api):
            fn = getattr(api, method)
            return await fn(*args, **kwargs)

        return await caller(operation, description or method, retry=retry)

    async def _ws_call(self,
                       operation: Callable[[k8s.client.CoreV1Api], Awaitable[T]],
                       description: str,
                       retry: bool = True) -> T:
        attempt = 0
        max_attempts = 2 if retry else 1
        while True:
            client = await self._get_ws_client()
            api = k8s.client.CoreV1Api(client)
            try:
                return await operation(api)
            except k8s.client.exceptions.ApiException as exc:
                should_retry = exc.status in self._retryable_statuses and attempt < max_attempts - 1
                if not should_retry:
                    raise
                attempt += 1
                logger.warning('K8s websocket %s failed with status %s, reconnecting (attempt %s/%s)',
                               description, exc.status, attempt + 1, max_attempts)
                await self._reset_ws_client()
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if attempt >= max_attempts - 1:
                    raise
                attempt += 1
                logger.warning('K8s websocket %s transport error %s, reconnecting (attempt %s/%s)',
                               description, exc, attempt + 1, max_attempts)
                await self._reset_ws_client()

    async def start_session(self,
                            subtypes: Union[List[str], str],
                            immutable: bool = True,
                            **kwargs) -> Tuple[str, Dict[str, str], Dict[str, str]]:
        subtypes = subtypes if isinstance(subtypes, list) else [subtypes]
        for subtype in subtypes:
            assert subtype in self.valid_subtypes, f'invalid subtype {subtype} for task {self.task_name}'

        session_id = await self.state.generate_session_id()
        pods_allocated: Dict[str, k8s.client.V1Pod] = {}
        pods_allocated_exclusive: Dict[str, k8s.client.V1Pod] = {}
        non_exclusive_subtypes = []

        for subtype in subtypes:
            usage_limit = self.delegation.get_reuse_limit(subtype)
            if usage_limit == 1 or (usage_limit != 0 and not immutable):
                # exclusive allocation, no need to lock
                pod = await self.create_pod(subtype, exclusive=True, **kwargs)
                await self.state.allocate_container(pod.metadata.name, session_id)
                pods_allocated[subtype] = pod
                pods_allocated_exclusive[subtype] = pod
            else:
                non_exclusive_subtypes.append(subtype)

        shared_candidates: Dict[str, List[k8s.client.V1Pod]] = {}
        if len(non_exclusive_subtypes) > 0:
            shared_candidates = await self._identify_pods(non_exclusive_subtypes)

        for subtype in non_exclusive_subtypes:
            existing_pods = shared_candidates.get(subtype, [])
            shuffle(existing_pods)

            concurrency_limit = self.delegation.get_concurrency_limit(subtype)
            usage_limit = self.delegation.get_reuse_limit(subtype)

            while True:
                selected_pod: Optional[k8s.client.V1Pod] = None

                async with self.state.with_lock(f'allocation:{subtype}', session_id):
                    for pod in existing_pods:
                        pod_name = pod.metadata.name
                        if 0 < usage_limit <= await self.state.container_total_uses(pod_name):
                            continue
                        if 0 < concurrency_limit <= await self.state.container_current_uses(pod_name):
                            continue
                        selected_pod = pod
                        break

                    if selected_pod is not None:
                        pods_allocated[subtype] = selected_pod
                        await self.state.allocate_container(selected_pod.metadata.name, session_id)

                if selected_pod is not None:
                    break

                new_pod = await self.create_pod(subtype, exclusive=False, **kwargs)
                existing_pods.append(new_pod)

        if self.delegation.has_homepage():
            # create dedicated homepage pod for the session
            homepage_subtype = self.delegation.get_homepage_subtype()
            homepage_envs = self.delegation.get_homepage_envs({
                subtype: await self.get_pod_url(pods_allocated, subtype)
                for subtype in pods_allocated.keys()
            })
            homepage_pod = await self.create_pod(homepage_subtype, homepage_envs, exclusive=True, **kwargs)
            await self.state.allocate_container(homepage_pod.metadata.name, session_id)
            pods_allocated[homepage_subtype] = homepage_pod
            pods_allocated_exclusive[homepage_subtype] = homepage_pod

        # save allocated pod ids to the session
        await self.state.store_session(session_id, SessionData(
            pods={
                subtype: pod.metadata.name
                for subtype, pod in pods_allocated.items()
            },
            exclusive_pods=[
                pod.metadata.name
                for subtype, pod in pods_allocated_exclusive.items()
            ]
        ))

        # log allocations
        for subtype, pod in pods_allocated.items():
            logger.info(f'Allocated pod {pod.metadata.name} to session {session_id}')

        # release the lock, while wait for pods to be healthy
        logger.info('Waiting for pods to be healthy')
        await self._wait_for_health(*pods_allocated.values())

        # return session_id, pod ids and environment urls
        return session_id, {
            subtype: pod.metadata.name
            for subtype, pod in pods_allocated.items()
        }, {
            subtype: await self.get_pod_url(pods_allocated, subtype)
            for subtype in pods_allocated.keys()
        }

    async def renew_session(self, session_id: str):
        await self.state.renew_session(session_id)
        session: Optional[SessionData] = await self.state.get_session(session_id)
        if session:
            for pod in session.get('pods', {}).values():
                await self.state.renew_container(pod, session_id)

    async def end_session(self, session_id: str):
        session: Optional[SessionData] = await self.state.get_session(session_id)
        if session:
            exclusive_pods = list(session.get('exclusive_pods', []) or [])
            exclusive_pod_ids = set(exclusive_pods)

            shared_pods = [
                (subtype, pod_id)
                for subtype, pod_id in session.get('pods', {}).items()
                if pod_id not in exclusive_pod_ids
            ]

            for subtype, pod_id in shared_pods:
                async with self.state.with_lock(f'allocation:{subtype}', session_id):
                    await self.state.release_container(pod_id, session_id)
                    logger.info(f'Released pod {pod_id}')

            for pod_id in exclusive_pods:
                try:
                    pod = await self._api(
                        'read_namespaced_pod',
                        name=pod_id,
                        namespace=self.namespace,
                        description=f'read pod {pod_id}'
                    )
                except k8s.client.exceptions.ApiException as exc:
                    if exc.status != 404:
                        logger.warning('Failed to read pod %s for deletion', pod_id, exc_info=True)
                    continue
                await self.delete_pod(pod)

        await self.state.delete_session(session_id)

    async def execute_command(self,
                              environment_id: str,
                              command: Union[str, List[str]],
                              timeout: int = 30) -> Tuple[int, bytes, bytes]:
        exec_command = command if isinstance(command, list) else ['/bin/sh', '-c', command]

        stdout_data = bytearray()
        stderr_data = bytearray()
        exit_code = 0

        resp = await self._api(
            'connect_get_namespaced_pod_exec',
            name=environment_id,
            namespace=self.namespace,
            container='main',
            command=exec_command,
            stderr=True,
            stdin=False,
            stdout=True,
            tty=False,
            _preload_content=False,
            description=f'exec pod {environment_id}',
            retry=False,
            use_ws=True
        )

        async def consume():
            nonlocal exit_code
            ws = await resp
            try:
                async for message in ws:
                    if message.type == aiohttp.WSMsgType.CLOSE:
                        break
                    if message.type == aiohttp.WSMsgType.ERROR:
                        raise RuntimeError('exec websocket error')
                    if message.type not in (aiohttp.WSMsgType.TEXT, aiohttp.WSMsgType.BINARY):
                        continue
                    raw = message.data.encode('utf-8') if isinstance(message.data, str) else message.data
                    if not raw:
                        continue
                    channel = raw[0]
                    payload = raw[1:]
                    if channel == k8s.stream.ws_client.STDOUT_CHANNEL:
                        stdout_data.extend(payload)
                    elif channel == k8s.stream.ws_client.STDERR_CHANNEL:
                        stderr_data.extend(payload)
                    elif channel == k8s.stream.ws_client.ERROR_CHANNEL:
                        try:
                            exit_code = k8s.stream.WsApiClient.parse_error_data(payload.decode('utf-8'))
                        except Exception:
                            exit_code = 1
                        break
            finally:
                await ws.close()

        await asyncio.wait_for(consume(), timeout)
        return exit_code, bytes(stdout_data), bytes(stderr_data)

    async def create_shell(self, environment_id: str, shell: str = '/bin/bash --login'):
        existing = self._shells.get(environment_id)
        if existing and not existing['ws'].closed:
            return
        if existing:
            self._shells.pop(environment_id, None)
            keepalive = existing.get('keepalive_task') if isinstance(existing, dict) else None
            if keepalive and not keepalive.done():
                keepalive.cancel()
                try:
                    await keepalive
                except Exception:
                    pass
            try:
                await existing['ws'].close()
            except Exception:
                pass

        shell_command = shlex.split(shell)

        resp = await self._api(
            'connect_get_namespaced_pod_exec',
            name=environment_id,
            namespace=self.namespace,
            container='main',
            command=shell_command,
            stderr=True,
            stdin=True,
            stdout=True,
            tty=True,
            _preload_content=False,
            description=f'shell pod {environment_id}',
            retry=False,
            use_ws=True
        )
        ws = await resp

        async def wait_for_prompt():
            while True:
                try:
                    message = await asyncio.wait_for(ws.receive(), self._shell_receive_timeout)
                except asyncio.TimeoutError:
                    raise RuntimeError('shell prompt wait timed out')
                if message.type == aiohttp.WSMsgType.CLOSE:
                    raise RuntimeError('shell closed before prompt was ready')
                if message.type == aiohttp.WSMsgType.ERROR:
                    raise RuntimeError('shell websocket error')
                if message.type not in (aiohttp.WSMsgType.TEXT, aiohttp.WSMsgType.BINARY):
                    continue
                raw = message.data.encode('utf-8') if isinstance(message.data, str) else message.data
                if not raw:
                    continue
                channel = raw[0]
                payload = raw[1:]
                if channel == k8s.stream.ws_client.ERROR_CHANNEL:
                    code = k8s.stream.WsApiClient.parse_error_data(payload.decode('utf-8'))
                    raise RuntimeError(f'shell exited with code {code}')
                if channel in (k8s.stream.ws_client.STDOUT_CHANNEL, k8s.stream.ws_client.STDERR_CHANNEL):
                    if SHELL_PROMPT_RE.search(payload):
                        break

        try:
            await asyncio.wait_for(wait_for_prompt(), 5)
        except asyncio.TimeoutError:
            logger.warning('Timed out waiting for shell prompt for %s', environment_id)
            await ws.close()
            raise
        except Exception:
            logger.warning('Shell initialization failed for %s (code=%s, reason=%s)',
                           environment_id, getattr(ws, 'close_code', None), getattr(ws, 'close_reason', None),
                           exc_info=True)
            await ws.close()
            raise

        session_data = {
            'ws': ws,
            'lock': asyncio.Lock()
        }

        async def keepalive():
            try:
                while not ws.closed:
                    await asyncio.sleep(self._shell_keepalive_interval)
                    if ws.closed:
                        break
                    try:
                        await ws.ping()
                    except Exception:
                        logger.debug('shell keepalive ping failed for %s', environment_id, exc_info=True)
                        break
            except asyncio.CancelledError:
                pass

        session_data['keepalive_task'] = asyncio.create_task(keepalive())
        self._shells[environment_id] = session_data

    async def execute_shell(self, environment_id: str, command: str, timeout: int = 30) -> bytes:
        session = self._shells.get(environment_id)
        if not session or session['ws'].closed:
            await self.create_shell(environment_id)
            session = self._shells.get(environment_id)
            if not session:
                raise RuntimeError('failed to initialize shell session')

        ws = session['ws']
        lock: asyncio.Lock = session['lock']

        async with lock:
            data = bytearray()
            ignored_first_chunk = False

            payload = command.encode('utf-8') + b'\n'
            try:
                await ws.send_bytes(bytes([k8s.stream.ws_client.STDIN_CHANNEL]) + payload)
            except Exception:
                logger.warning('Shell websocket send failed for %s (code=%s, reason=%s)',
                               environment_id, getattr(ws, 'close_code', None), getattr(ws, 'close_reason', None),
                               exc_info=True)
                keepalive = session.get('keepalive_task') if session else None
                if keepalive and not keepalive.done():
                    keepalive.cancel()
                    try:
                        await keepalive
                    except Exception:
                        pass
                self._shells.pop(environment_id, None)
                try:
                    await ws.close()
                except Exception:
                    pass
                raise

            async def read_until_prompt():
                nonlocal ignored_first_chunk
                while True:
                    try:
                        message = await asyncio.wait_for(ws.receive(), self._shell_receive_timeout)
                    except asyncio.TimeoutError:
                        raise RuntimeError('shell read timed out')
                    if message.type == aiohttp.WSMsgType.CLOSE:
                        logger.warning('Shell websocket closed for %s (code=%s, reason=%s)',
                                       environment_id, getattr(ws, 'close_code', None), getattr(ws, 'close_reason', None))
                        raise RuntimeError('shell connection closed')
                    if message.type == aiohttp.WSMsgType.ERROR:
                        logger.warning('Shell websocket errored for %s (code=%s, reason=%s)',
                                       environment_id, getattr(ws, 'close_code', None), getattr(ws, 'close_reason', None))
                        raise RuntimeError('shell websocket error')
                    if message.type not in (aiohttp.WSMsgType.TEXT, aiohttp.WSMsgType.BINARY):
                        continue
                    raw = message.data.encode('utf-8') if isinstance(message.data, str) else message.data
                    if not raw:
                        continue
                    channel = raw[0]
                    chunk = raw[1:]
                    if channel == k8s.stream.ws_client.ERROR_CHANNEL:
                        code = k8s.stream.WsApiClient.parse_error_data(chunk.decode('utf-8'))
                        raise RuntimeError(f'shell exited with code {code}')
                    if channel not in (k8s.stream.ws_client.STDOUT_CHANNEL, k8s.stream.ws_client.STDERR_CHANNEL):
                        continue
                    if ignored_first_chunk:
                        data.extend(chunk)
                        if SHELL_PROMPT_RE.search(chunk):
                            break
                    else:
                        ignored_first_chunk = True
                return bytes(data)

            try:
                return await asyncio.wait_for(read_until_prompt(), timeout)
            except Exception:
                logger.warning('Shell websocket read failed for %s (code=%s, reason=%s)',
                               environment_id, getattr(ws, 'close_code', None), getattr(ws, 'close_reason', None),
                               exc_info=True)
                keepalive = session.get('keepalive_task') if session else None
                if keepalive and not keepalive.done():
                    keepalive.cancel()
                    try:
                        await keepalive
                    except Exception:
                        pass
                self._shells.pop(environment_id, None)
                try:
                    await ws.close()
                except Exception:
                    pass
                raise

    async def get_env_variables(self, environment_id: str) -> Dict[str, str]:
        """
        Get environment variables for the specified pod.

        Notes for the k8s implementation:
        - If multiple containers are in the pod, we return them combined,
          with the main container's variables taking precedence.
        - Only variables with direct values are returned;
          those sourced from secrets or config maps are ignored.
        """
        pod = await self._api(
            'read_namespaced_pod',
            name=environment_id,
            namespace=self.namespace,
            description=f'read pod {environment_id}'
        )

        envs: Dict[str, str] = {}
        for container in pod.spec.containers:
            if not container.env:
                continue
            for env_var in container.env:
                if env_var.value is not None:
                    envs[env_var.name] = env_var.value
        for container in pod.spec.containers:
            if container.name == 'main':
                if not container.env:
                    continue
                for env_var in container.env:
                    if env_var.value is not None:
                        envs[env_var.name] = env_var.value
                break

        return envs

    async def background_task(self):
        while True:
            try:
                if await self.state.acquire_lock('background', timeout=120):
                    try:
                        await self._clean_pods()
                    except Exception:
                        logger.warning('Error while cleaning pods', exc_info=True)
                    await self.state.release_lock('background')
            except Exception:
                logger.warning('Error in background task', exc_info=True)
            await asyncio.sleep(10)

    async def create_pod(self,
                         subtype: str,
                         extra_envs: Dict[str, str] = None,
                         exclusive: Optional[bool] = None,
                         **kwargs) -> k8s.client.V1Pod:
        if not extra_envs:
            extra_envs = {}
        if exclusive is None:
            exclusive = self.delegation.get_reuse_limit(subtype) == 1

        # generate pod name
        pod_name = f'{subtype.replace("_", "-").lower()}-{uuid.uuid4().hex[:8]}'

        try:
            container_ports = self.delegation.get_service_port(subtype) or []
        except NotImplementedError:
            container_ports = []
        if not isinstance(container_ports, list):
            container_ports = [container_ports]
        container_ports = [
            k8s.client.V1ContainerPort(container_port=port, name=f'port-{port}')
            for port in container_ports
        ]

        container = k8s.client.V1Container(
            name='main',
            image_pull_policy='IfNotPresent',
            env=[],
            ports=container_ports,
            resources=k8s.client.V1ResourceRequirements(
                requests={
                    'cpu': '500m',
                    'memory': '500Mi'
                },
                limits={
                    'cpu': '1000m',
                    'memory': '1Gi',
                    'ephemeral-storage': '10Gi'
                }
            )
        )

        spec = k8s.client.V1PodSpec(
            active_deadline_seconds=self.delegation.get_max_execution_time(subtype) if exclusive else None,
            node_selector={},
            containers=[container],
            restart_policy='Never',
            termination_grace_period_seconds=0
        )

        # allow task-specific customization similar to Docker delegation
        metadata_overrides: Dict[str, Any] = {}
        spec_result = await self.delegation.create_k8s_pod(spec, subtype, **kwargs)
        if isinstance(spec_result, tuple):
            spec, metadata_overrides = spec_result
        else:
            spec = spec_result
        if metadata_overrides and not isinstance(metadata_overrides, dict):
            raise TypeError('metadata overrides from create_k8s_pod must be a dict')
        metadata_overrides = metadata_overrides or {}

        # overrides to ensure proper behavior
        spec.automount_service_account_token = False
        spec.restart_policy = 'Never'
        spec.priority_class_name = 'agentrl-environment'
        for container in spec.containers:
            if not getattr(container, 'readiness_probe', None) and getattr(container, 'liveness_probe', None):
                container.readiness_probe = container.liveness_probe
                container.liveness_probe = None

        # re-fetch the main container after delegation customization
        primary_container: Optional[k8s.client.V1Container] = None
        if spec.containers:
            primary_container = next((c for c in spec.containers if getattr(c, 'name', None) == 'main'), spec.containers[0])

        if not isinstance(primary_container, dict):
            if not getattr(primary_container, 'image', None):
                primary_container.image = self.delegation.get_container_images()[subtype]

            if extra_envs:
                existing_env = {
                    env.name: env
                    for env in (primary_container.env or [])
                    if env and env.name
                }
                for key, value in extra_envs.items():
                    existing_env[key] = k8s.client.V1EnvVar(name=key, value=value)
                primary_container.env = list(existing_env.values())

        metadata = k8s.client.V1ObjectMeta(
            name=pod_name,
            labels={
                K8S_LABEL_ROLE: k8S_LABEL_ROLE_ENVIRONMENT,
                K8S_LABEL_TASK_TYPE: self.task_name,
                K8S_LABEL_SUBTYPE_NAME: subtype,
                K8S_LABEL_EXCLUSIVE: str(exclusive).lower(),
            }
        )
        if metadata_overrides:
            if 'labels' in metadata_overrides:
                metadata.labels = {
                    **(metadata.labels or {}),
                    **metadata_overrides['labels']
                }
            if 'annotations' in metadata_overrides:
                metadata.annotations = {
                    **(metadata.annotations or {}),
                    **metadata_overrides['annotations']
                }

        pod = await self._api(
            'create_namespaced_pod',
            namespace=self.namespace,
            body=k8s.client.V1Pod(metadata=metadata, spec=spec),
            description=f'create pod {pod_name}',
            retry=False
        )

        # check for dependencies, inject ownerReference for them to ensure cleanup
        if getattr(pod.metadata, 'annotations', None):
            for dep_name in (pod.metadata.annotations.get(K8S_ANNOTATION_DEPENDS_ON) or '').split(','):
                try:
                    await self._api(
                        'patch_namespaced_pod',
                        name=dep_name,
                        namespace=self.namespace,
                        body=[{
                            'op': 'add',
                            'path': '/metadata/ownerReferences',
                            'value': [k8s.client.V1OwnerReference(
                                api_version='v1',
                                kind='Pod',
                                name=pod.metadata.name,
                                uid=pod.metadata.uid
                            )]
                        }]
                    )
                except k8s.client.exceptions.ApiException:
                    logger.warning('failed to patch dependency pod %s for pod %s', dep_name, pod_name, exc_info=True)

        logger.debug('created pod %s with spec: %s', pod_name, spec.to_dict())

        asyncio.create_task(self.post_create_pod(subtype, pod))

        self._record_pod_cache(pod)

        return pod

    async def post_create_pod(self, subtype: str, pod: k8s.client.V1Pod):
        if 'post_create_k8s_pod' not in self.delegation.__class__.__dict__:
            return  # not implemented by the delegation

        await self._wait_for_health(pod)
        await self.delegation.post_create_k8s_pod(
            subtype,
            pod.metadata.name,
            await self.get_pod_url(pod, subtype)
        )

    async def delete_pod(self, pod: k8s.client.V1Pod):
        if pod.metadata and pod.metadata.name:
            shell_session = self._shells.pop(pod.metadata.name, None)
            if shell_session:
                keepalive = shell_session.get('keepalive_task')
                if keepalive and not keepalive.done():
                    keepalive.cancel()
                    try:
                        await keepalive
                    except Exception:
                        pass
                ws = shell_session.get('ws')
                if ws and not ws.closed:
                    try:
                        await ws.close()
                    except Exception:
                        pass

        try:
            await self._api(
                'delete_namespaced_pod',
                name=pod.metadata.name,
                namespace=self.namespace,
                grace_period_seconds=1,
                propagation_policy='Foreground',
                description=f'delete pod {pod.metadata.name}'
            )
        except Exception:
            logger.warning('failed to delete pod %s', pod.metadata.name, exc_info=True)
            return

        self._evict_pod_from_cache(pod)
        await self.state.remove_container(pod.metadata.name)
        logger.debug('deleted pod %s', pod.metadata.name)

    async def get_pod_url(self,
                          pods: Union[Dict[str, Union[k8s.client.V1Pod, str]], k8s.client.V1Pod, str],
                          subtype: str) -> str:
        if isinstance(pods, k8s.client.V1Pod):
            pod: Union[k8s.client.V1Pod, str] = pods
        elif isinstance(pods, dict):
            pod = pods[subtype]
        else:
            pod = pods

        if isinstance(pod, k8s.client.V1Pod):
            metadata = pod.metadata or None
            pod_name = metadata.name if metadata else None
            pod_status = pod.status
            if (not pod_status or not pod_status.pod_ip) and pod_name:
                pod = await self._api(
                    'read_namespaced_pod',
                    name=pod_name,
                    namespace=self.namespace,
                    description=f'read pod {pod_name} for url'
                )
        elif isinstance(pod, str):
            pod = await self._api(
                'read_namespaced_pod',
                name=pod,
                namespace=self.namespace,
                description=f'read pod {pod} for url'
            )

        ip = pod.status.pod_ip if pod.status else None
        port = self.delegation.get_service_port(subtype)
        if not port:
            return ip
        if port == 80:
            return f'http://{ip}'
        return f'http://{ip}:{port}'

    async def resolve_service_ip(self, service: str) -> str:
        svc = await self._api(
            'read_namespaced_service',
            name=service,
            namespace=self.namespace,
            description=f'read service {service}'
        )
        return svc.spec.cluster_ip

    async def _identify_pods(self, subtypes: Optional[List[str]] = None) -> Dict[str, List[k8s.client.V1Pod]]:
        if not self._pod_cache_initialized:
            if self._pod_cache_lock is None:
                self._pod_cache_lock = asyncio.Lock()
            async with self._pod_cache_lock:
                if not self._pod_cache_initialized:
                    await self._refresh_pod_cache()
            await self._ensure_pod_watch_started()

        result: Dict[str, List[k8s.client.V1Pod]] = {}
        requested_subtypes = subtypes or self.valid_subtypes
        for subtype in requested_subtypes:
            cached = self._pod_cache.get(subtype, {})
            if not cached:
                result[subtype] = []
                continue

            filtered: List[k8s.client.V1Pod] = []
            for pod in cached.values():
                metadata = pod.metadata
                if not metadata or metadata.deletion_timestamp:
                    continue
                labels = metadata.labels or {}
                if labels.get(K8S_LABEL_EXCLUSIVE, '').lower() != str(False).lower():
                    continue
                status = pod.status
                phase = (status.phase or '').lower() if status and status.phase else ''
                if phase in {'failed', 'unknown', 'succeeded'}:
                    continue
                filtered.append(pod)

            result[subtype] = filtered

        return result

    async def _clean_pods(self):
        # remove unused exclusive pods
        exclusive_pods = await self._api(
            'list_namespaced_pod',
            namespace=self.namespace,
            label_selector=self._label_selector(exclusive=True),
            description='list exclusive pods'
        )
        for pod in exclusive_pods.items or []:
            if not await self.state.container_is_allocated(pod.metadata.name):
                await self.delete_pod(pod)

        # remove unhealthy non-exclusive pods that are not allocated
        shared_pods = await self._api(
            'list_namespaced_pod',
            namespace=self.namespace,
            label_selector=self._label_selector(exclusive=False),
            description='list shared pods'
        )
        for pod in shared_pods.items or []:
            if await self.state.container_is_allocated(pod.metadata.name):
                continue
            delete_pod = False
            metadata = pod.metadata
            if metadata and metadata.deletion_timestamp:
                delete_pod = True
            status = pod.status
            if not delete_pod and status:
                phase = status.phase.lower() if status.phase else ''
                if phase in {'', 'failed', 'unknown', 'succeeded'}:
                    delete_pod = True
                elif status.container_statuses:
                    for container_status in status.container_statuses:
                        state = container_status.state
                        if not state:
                            continue
                        if getattr(state, 'waiting', None) or getattr(state, 'terminated', None):
                            delete_pod = True
                            break
            if delete_pod:
                await self.delete_pod(pod)

        # remove pods that hit the reuse limit and are idle
        pods_by_subtype = await self._identify_pods()
        for subtype, pods in pods_by_subtype.items():
            usage_limit = self.delegation.get_reuse_limit(subtype)
            if usage_limit == 0:
                continue
            for pod in pods:
                if await self.state.container_is_allocated(pod.metadata.name):
                    continue
                if await self.state.container_total_uses(pod.metadata.name) < usage_limit:
                    continue
                await self.delete_pod(pod)

    async def _wait_for_health(self, *pods: Union[k8s.client.V1Pod, str]):
        if not pods:
            return

        pod_names: List[str] = []
        for pod in pods:
            if isinstance(pod, k8s.client.V1Pod):
                if pod.metadata and pod.metadata.name:
                    pod_names.append(pod.metadata.name)
            elif isinstance(pod, str):
                pod_names.append(pod)

        if not pod_names:
            return

        failure_reasons = {'CrashLoopBackOff', 'ErrImagePull', 'ImagePullBackOff', 'RunContainerError'}

        while True:
            wait_needed = False
            for name in pod_names:
                try:
                    pod_obj = await self._api(
                        'read_namespaced_pod',
                        name=name,
                        namespace=self.namespace,
                        description=f'read pod {name}'
                    )
                except k8s.client.exceptions.ApiException as e:
                    if e.status == 404:
                        wait_needed = True
                        break
                    raise

                metadata = pod_obj.metadata
                if metadata and metadata.deletion_timestamp:
                    continue

                # 1. check for pod phase
                status = pod_obj.status
                phase = (status.phase or '').lower() if status and status.phase else ''
                if phase in {'failed', 'succeeded'}:
                    continue
                if phase in {'pending', 'unknown', ''}:
                    wait_needed = True
                    break

                # 2. check for readiness condition
                for condition in getattr(status, 'conditions', None) or []:
                    if condition.type == 'Ready' and condition.status == 'False':
                        wait_needed = True
                        break

                # 3. check container statuses
                container_statuses = status.container_statuses or []
                if not container_statuses:
                    wait_needed = True
                    break
                unhealthy_reason: Optional[str] = None
                still_starting = False
                for container_status in container_statuses:
                    state = container_status.state
                    if not state:
                        continue
                    waiting = getattr(state, 'waiting', None)
                    if waiting:
                        reason = (waiting.reason or '').strip()
                        if reason in failure_reasons:
                            unhealthy_reason = waiting.message or reason or 'unknown'
                            break
                        still_starting = True
                        break
                    terminated = getattr(state, 'terminated', None)
                    if terminated and terminated.reason and terminated.reason not in {'Completed'}:
                        unhealthy_reason = terminated.reason
                        break
                if unhealthy_reason:
                    raise RuntimeError(f'Pod {name} failed health check: {unhealthy_reason}')
                if still_starting:
                    wait_needed = True
                    break

            if not wait_needed:
                return

            await asyncio.sleep(1)

    async def _ensure_pod_watch_started(self):
        if self._pod_watch_task and not self._pod_watch_task.done():
            return

        if self._pod_watch_task and self._pod_watch_task.done():
            try:
                self._pod_watch_task.result()
            except Exception:
                logger.warning('Pod watch task terminated unexpectedly; restarting', exc_info=True)

        self._pod_watch_task = asyncio.create_task(self._watch_pods())

    async def _watch_pods(self):
        label_selector = self._label_selector(exclusive=False)
        backoff = 1

        while True:
            try:
                client = await self._get_client()
                api = k8s.client.CoreV1Api(client)

                if not self._pod_cache_initialized:
                    await self._refresh_pod_cache()

                watcher = k8s.watch.Watch()
                stream_kwargs = {
                    'namespace': self.namespace,
                    'label_selector': label_selector,
                    'timeout_seconds': 60
                }
                if self._pod_cache_resource_version:
                    stream_kwargs['resource_version'] = self._pod_cache_resource_version

                async for event in watcher.stream(api.list_namespaced_pod, **stream_kwargs):
                    event_type = event.get('type')
                    obj = event.get('object')

                    if event_type == 'ERROR':
                        self._pod_cache_initialized = False
                        self._pod_cache_resource_version = None
                        break

                    if not isinstance(obj, k8s.client.V1Pod):
                        continue

                    if obj.metadata and obj.metadata.resource_version:
                        self._pod_cache_resource_version = obj.metadata.resource_version

                    if event_type == 'BOOKMARK':
                        continue

                    if event_type == 'DELETED':
                        self._evict_pod_from_cache(obj)
                    else:
                        self._record_pod_cache(obj)

                watcher.stop()
                backoff = 1
            except asyncio.CancelledError:
                break
            except k8s.client.exceptions.ApiException as e:
                if e.status == 410:
                    # resource version is too old, force a relist
                    self._pod_cache_initialized = False
                    self._pod_cache_resource_version = None
                logger.warning('Pod watch API error, restarting')
                await self._reset_client()
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)
            except Exception:
                logger.warning('Pod watch error, restarting', exc_info=True)
                await self._reset_client()
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)

    async def _refresh_pod_cache(self):
        response = await self._api(
            'list_namespaced_pod',
            namespace=self.namespace,
            label_selector=self._label_selector(exclusive=False),
            description='refresh pod cache'
        )

        self._pod_cache.clear()
        for pod in response.items or []:
            self._record_pod_cache(pod)

        self._pod_cache_initialized = True
        self._pod_cache_resource_version = response.metadata.resource_version if response.metadata else None

    def _record_pod_cache(self, pod: k8s.client.V1Pod):
        metadata = pod.metadata

        labels = metadata.labels if metadata and metadata.labels else None
        if not metadata or not labels:
            return
        if labels.get(K8S_LABEL_EXCLUSIVE, '').lower() != str(False).lower():
            return

        subtype = labels.get(K8S_LABEL_SUBTYPE_NAME)
        if not subtype:
            return

        self._pod_cache.setdefault(subtype, {})[pod.metadata.name] = pod

    def _evict_pod_from_cache(self, pod: k8s.client.V1Pod):
        if not pod.metadata:
            return

        subtype = None
        if pod.metadata.labels:
            subtype = pod.metadata.labels.get(K8S_LABEL_SUBTYPE_NAME)
        if not subtype:
            # fall back to metadata name lookup across all subtypes
            for cache in self._pod_cache.values():
                cache.pop(pod.metadata.name, None)
            return

        cache = self._pod_cache.get(subtype)
        if cache is not None:
            cache.pop(pod.metadata.name, None)

    def _label_selector(self, exclusive: Optional[bool] = None, subtype: Optional[str] = None) -> str:
        labels = {
            K8S_LABEL_ROLE: k8S_LABEL_ROLE_ENVIRONMENT,
            K8S_LABEL_TASK_TYPE: self.task_name,
        }

        if exclusive is not None:
            labels[K8S_LABEL_EXCLUSIVE] = str(exclusive).lower()

        if subtype:
            labels[K8S_LABEL_SUBTYPE_NAME] = subtype

        return ','.join(f'{k}={v}' for k, v in labels.items())

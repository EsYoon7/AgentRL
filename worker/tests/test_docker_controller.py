from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import suppress
from dataclasses import dataclass
from types import MethodType
from typing import Any, Dict, List, Optional

import pytest

from agentrl.worker.environment import EnvironmentDelegation
from agentrl.worker.environment import docker as docker_module
from agentrl.worker.environment._const import (
    LABEL_DEPENDS_ON,
    LABEL_SUBTYPE_NAME,
)
from agentrl.worker.environment.docker import DockerEnvironmentController
from .stubs import MemoryStateProvider, TracingStateProvider


@dataclass
class DelegationConfig:
    name: str
    subtypes: List[str]
    concurrency: Dict[str, int]
    reuse: Dict[str, int]
    service_ports: Dict[str, Optional[int]]
    images: Dict[str, str]
    homepage: bool = False
    homepage_subtype: Optional[str] = None


class StubDelegation(EnvironmentDelegation):
    def __init__(self, config: DelegationConfig):
        super().__init__(config.name)
        self.config = config
        self.created: List[tuple[str, Dict[str, Any]]] = []

    def get_subtypes(self) -> List[str]:
        return self.config.subtypes

    def get_service_port(self, subtype: str) -> Optional[int]:
        return self.config.service_ports.get(subtype)

    def get_concurrency_limit(self, subtype: str) -> int:
        return self.config.concurrency.get(subtype, 0)

    def get_reuse_limit(self, subtype: str) -> int:
        return self.config.reuse.get(subtype, 0)

    def get_container_images(self) -> Dict[str, str]:
        return self.config.images

    async def create_docker_container(self, attrs: dict, subtype: str, **kwargs) -> dict:
        self.created.append((subtype, dict(attrs)))
        attrs.setdefault('Env', {})
        return attrs

    def has_homepage(self) -> bool:
        return self.config.homepage

    def get_homepage_subtype(self) -> str:
        assert self.config.homepage_subtype
        return self.config.homepage_subtype

    def get_homepage_envs(self, site_urls: Dict[str, str]) -> dict:
        return {'PRIMARY_URL': next(iter(site_urls.values()))}


class StubDockerContainer:
    """Base class so isinstance checks in the controller succeed."""


class FakeContainer(StubDockerContainer):
    def __init__(self, manager: 'FakeContainerManager', attrs: dict, name: str, ip_suffix: int):
        self.manager = manager
        self.attrs = json.loads(json.dumps(attrs))  # deep copy
        self.id = name or f'container-{uuid.uuid4().hex[:8]}'
        self.name = self.id
        self.status = 'created'
        self.health = 'healthy'
        self._container: Optional[dict] = None
        self._ip = f'10.1.0.{ip_suffix}'

    async def start(self):
        self.status = 'running'

    async def show(self):
        env_entries = self.attrs.get('Env', [])
        network = self.attrs.get('HostConfig', {}).get('NetworkMode')
        self._container = {
            'Config': {'Env': env_entries},
            'Labels': self.attrs.get('Labels', {}),
            'NetworkSettings': {
                'Networks': {
                    network: {
                        'IPAddress': self._ip
                    }
                }
            }
        }
        return self._container

    async def delete(self, v: bool = True, force: bool = True):
        self.manager.remove(self.id)

    def set_health(self, state: str):
        self.health = state

    def matches(self, filters: Dict[str, Any]) -> bool:
        if not filters:
            return True
        labels = self.attrs.get('Labels', {})
        for key, value in filters.items():
            if key == 'label':
                for condition in value:
                    if '=' in condition:
                        label_key, expected = condition.split('=', 1)
                        if labels.get(label_key) != expected:
                            return False
                    else:
                        if condition not in labels:
                            return False
            elif key == 'network':
                network_mode = self.attrs.get('HostConfig', {}).get('NetworkMode')
                if network_mode not in value:
                    return False
            elif key == 'status':
                if self.status not in value:
                    return False
            elif key == 'health':
                if self.health not in value:
                    return False
            elif key == 'id':
                if self.id not in value:
                    return False
        return True

    def __getitem__(self, item: str) -> Any:
        if self._container is None:
            raise KeyError('container not inspected yet')
        return self._container[item]


class FakeContainerManager:
    def __init__(self):
        self._containers: Dict[str, FakeContainer] = {}
        self._ip_seq = 2

    async def create(self, attrs: dict, name: Optional[str] = None) -> FakeContainer:
        container = FakeContainer(self, attrs, name or f'container-{uuid.uuid4().hex[:8]}', self._ip_seq)
        self._containers[container.id] = container
        self._ip_seq += 1
        return container

    async def list(self, filters: Optional[str] = None) -> List[FakeContainer]:
        parsed = json.loads(filters) if filters else {}
        return [c for c in list(self._containers.values()) if c.matches(parsed)]

    def container(self, container_id: str) -> FakeContainer:
        return self._containers[container_id]

    def remove(self, container_id: str):
        self._containers.pop(container_id, None)


class FakeImages:
    def __init__(self):
        self.pulled: List[str] = []

    async def pull(self, image: str):
        self.pulled.append(image)


class FakeDockerClient:
    def __init__(self):
        self.containers = FakeContainerManager()
        self.images = FakeImages()
        self.session = type('Session', (), {'closed': False})()

    async def close(self):
        self.session.closed = True


def get_containers_by_subtype(client: FakeDockerClient, subtype: str) -> List[FakeContainer]:
    return [c for c in client.containers._containers.values()
            if c.attrs.get('Labels', {}).get(LABEL_SUBTYPE_NAME) == subtype]


def build_controller(monkeypatch: pytest.MonkeyPatch,
                     delegation: StubDelegation,
                     state: MemoryStateProvider,
                     client: FakeDockerClient) -> DockerEnvironmentController:
    monkeypatch.setattr(docker_module, 'create_state_provider', lambda *args, **kwargs: state)
    monkeypatch.setattr(docker_module.aiodocker, 'Docker', lambda **kwargs: client)
    monkeypatch.setattr(docker_module, 'DockerContainer', StubDockerContainer)
    controller = DockerEnvironmentController(
        delegation=delegation,
        connection={},
        network_name='agent-net',
        state_driver='dummy'
    )
    return controller


def default_config() -> DelegationConfig:
    return DelegationConfig(
        name='test-task',
        subtypes=['exclusive', 'shared', 'homepage'],
        concurrency={'exclusive': 0, 'shared': 2, 'homepage': 0},
        reuse={'exclusive': 1, 'shared': 0, 'homepage': 1},
        service_ports={'exclusive': 80, 'shared': 8080, 'homepage': 8081},
        images={'exclusive': 'exclusive:latest', 'shared': 'shared:latest', 'homepage': 'home:latest'},
        homepage=True,
        homepage_subtype='homepage'
    )


@pytest.mark.asyncio
async def test_start_session_allocates_exclusive_and_shared(monkeypatch):
    state = MemoryStateProvider()
    client = FakeDockerClient()
    config = default_config()
    delegation = StubDelegation(config)
    controller = build_controller(monkeypatch, delegation, state, client)

    session_id, containers, urls = await controller.start_session(['exclusive', 'shared'])

    assert set(containers.keys()) == {'exclusive', 'shared', 'homepage'}
    assert urls['exclusive'].startswith('http://')
    assert urls['shared'].startswith('http://') and urls['shared'].endswith(':8080')
    assert urls['homepage'].startswith('http://')

    for subtype, container_id in containers.items():
        assert session_id in state.allocations[container_id]

    session_record = await state.get_session(session_id)
    assert set(session_record['exclusive_containers']) == {
        containers['exclusive'],
        containers['homepage']
    }
    assert containers['shared'] not in session_record['exclusive_containers']


@pytest.mark.asyncio
async def test_start_session_reuses_existing_nonexclusive_container(monkeypatch):
    state = MemoryStateProvider()
    client = FakeDockerClient()
    config = default_config()
    delegation = StubDelegation(config)
    controller = build_controller(monkeypatch, delegation, state, client)

    # Create initial container and leave it idle
    existing = await controller.create_container('shared', exclusive=False)
    await existing.start()
    await existing.show()

    session_id, containers, _ = await controller.start_session(['shared'])

    assert containers['shared'] == existing.id
    assert session_id in state.allocations[existing.id]
    assert client.images.pulled == []  # image was available and not re-pulled


@pytest.mark.asyncio
async def test_start_session_skips_container_past_reuse_limit(monkeypatch):
    state = MemoryStateProvider()
    client = FakeDockerClient()
    config = default_config()
    config.reuse['shared'] = 2
    delegation = StubDelegation(config)
    controller = build_controller(monkeypatch, delegation, state, client)

    exhausted = await controller.create_container('shared', exclusive=False)
    await exhausted.start()
    await exhausted.show()
    state.total_uses[exhausted.id] = 2  # at limit

    _, containers, _ = await controller.start_session(['shared'])

    assert containers['shared'] != exhausted.id
    assert exhausted.id not in state.allocations or not state.allocations[exhausted.id]


@pytest.mark.asyncio
async def test_concurrency_limit_triggers_new_container(monkeypatch):
    state = MemoryStateProvider()
    client = FakeDockerClient()
    config = default_config()
    config.concurrency['shared'] = 1
    delegation = StubDelegation(config)
    controller = build_controller(monkeypatch, delegation, state, client)

    busy = await controller.create_container('shared', exclusive=False)
    await busy.start()
    await busy.show()
    state.allocations[busy.id].add('other-session')

    _, containers, _ = await controller.start_session(['shared'])

    assert containers['shared'] != busy.id
    shared_ids = [c.id for c in get_containers_by_subtype(client, 'shared')]
    assert busy.id in shared_ids
    assert len(shared_ids) == 2


@pytest.mark.asyncio
async def test_usage_limit_does_not_spawn_spare_container(monkeypatch):
    state = MemoryStateProvider()
    client = FakeDockerClient()
    config = default_config()
    config.reuse['shared'] = 3
    delegation = StubDelegation(config)
    controller = build_controller(monkeypatch, delegation, state, client)

    await controller.start_session(['shared'])

    shared_containers = get_containers_by_subtype(client, 'shared')
    assert len(shared_containers) == 1


@pytest.mark.asyncio
async def test_homepage_container_provision(monkeypatch):
    state = MemoryStateProvider()
    client = FakeDockerClient()
    config = default_config()
    delegation = StubDelegation(config)
    controller = build_controller(monkeypatch, delegation, state, client)

    session_id, containers, urls = await controller.start_session(['shared'])

    session_meta = await state.get_session(session_id)
    assert len(session_meta['exclusive_containers']) == 1
    homepage_id = session_meta['exclusive_containers'][0]
    homepage_container = client.containers.container(homepage_id)
    await homepage_container.show()

    env_entries = homepage_container['Config']['Env']
    assert any(entry.startswith('PRIMARY_URL=') for entry in env_entries)
    assert containers['homepage'] == homepage_id
    assert urls['shared'].startswith('http://')


@pytest.mark.asyncio
async def test_get_container_url_accepts_container_id(monkeypatch):
    state = MemoryStateProvider()
    client = FakeDockerClient()
    config = default_config()
    delegation = StubDelegation(config)
    controller = build_controller(monkeypatch, delegation, state, client)

    container = await controller.create_container('shared', exclusive=False)
    await container.start()
    await container.show()

    network_ip = container['NetworkSettings']['Networks']['agent-net']['IPAddress']
    url = await controller.get_container_url(container.id, 'shared')

    assert url == f'http://{network_ip}:8080'


@pytest.mark.asyncio
async def test_clean_containers_removes_unused_resources(monkeypatch):
    state = MemoryStateProvider()
    client = FakeDockerClient()
    config = default_config()
    config.reuse['shared'] = 2
    delegation = StubDelegation(config)
    controller = build_controller(monkeypatch, delegation, state, client)

    exclusive = await controller.create_container('exclusive', exclusive=True)
    await exclusive.start()
    await exclusive.show()

    unhealthy = await controller.create_container('shared', exclusive=False)
    unhealthy.set_health('unhealthy')
    await unhealthy.start()
    await unhealthy.show()

    overused = await controller.create_container('shared', exclusive=False)
    await overused.start()
    await overused.show()
    state.total_uses[overused.id] = 3

    dependent = await controller.create_container('shared', exclusive=False)
    child = await controller.create_container('shared', exclusive=False)
    dependent.attrs['Labels'][LABEL_DEPENDS_ON] = child.id
    dependent.set_health('unhealthy')
    await dependent.start()
    await dependent.show()
    await child.start()
    await child.show()

    await controller._clean_containers()

    removed_ids = set(state.removed_containers)
    assert exclusive.id in removed_ids
    assert unhealthy.id in removed_ids
    assert overused.id in removed_ids
    assert child.id in removed_ids and dependent.id in removed_ids


@pytest.mark.asyncio
async def test_background_task_uses_global_lock(monkeypatch):
    state = TracingStateProvider()
    client = FakeDockerClient()
    config = default_config()
    delegation = StubDelegation(config)
    controller_a = build_controller(monkeypatch, delegation, state, client)
    controller_b = build_controller(monkeypatch, delegation, state, client)

    clean_calls = 0
    clean_concurrency = 0
    max_clean_concurrency = 0
    cleaned_controllers: set[int] = set()
    stop_event = asyncio.Event()

    async def slow_clean(self):
        nonlocal clean_calls, clean_concurrency, max_clean_concurrency
        clean_concurrency += 1
        max_clean_concurrency = max(max_clean_concurrency, clean_concurrency)
        await asyncio.sleep(0.01)
        clean_concurrency -= 1
        clean_calls += 1
        cleaned_controllers.add(id(self))
        if clean_calls >= 2 and len(cleaned_controllers) >= 2:
            stop_event.set()

    controller_a._clean_containers = MethodType(slow_clean, controller_a)
    controller_b._clean_containers = MethodType(slow_clean, controller_b)

    task_a = asyncio.create_task(controller_a.background_task())
    task_b = asyncio.create_task(controller_b.background_task())

    await asyncio.wait_for(stop_event.wait(), timeout=2)

    for task in (task_a, task_b):
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    assert clean_calls >= 2
    assert len(cleaned_controllers) == 2
    assert state.max_lock_holders['background'] == 1
    assert max_clean_concurrency == 1


@pytest.mark.asyncio
async def test_end_session_releases_shared_and_destroys_exclusive(monkeypatch):
    state = MemoryStateProvider()
    client = FakeDockerClient()
    config = default_config()
    delegation = StubDelegation(config)
    controller = build_controller(monkeypatch, delegation, state, client)

    session_id, containers, _ = await controller.start_session(['exclusive', 'shared'])
    await controller.end_session(session_id)

    assert containers['exclusive'] in state.removed_containers
    assert not state.allocations.get(containers['shared'])
    assert await state.get_session(session_id) is None

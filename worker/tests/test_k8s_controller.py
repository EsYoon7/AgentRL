from __future__ import annotations

import copy
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock
import uuid

import aiohttp
import pytest
from kubernetes_asyncio.client.exceptions import ApiException

from agentrl.worker.environment import EnvironmentDelegation
from agentrl.worker.environment import k8s as k8s_module
from agentrl.worker.environment._const import (
    K8S_ANNOTATION_DEPENDS_ON,
    K8S_LABEL_SUBTYPE_NAME,
)
from agentrl.worker.environment.k8s import K8sEnvironmentController
from .stubs import MemoryStateProvider


class FakeKubernetesCluster:
    def __init__(self):
        self._namespaces: Dict[str, Dict[str, Any]] = {}
        self._ip_counter = 10

    def _next_ip(self) -> str:
        ip = f'10.0.0.{self._ip_counter}'
        self._ip_counter += 1
        return ip

    def _pod_store(self, namespace: str) -> Dict[str, Any]:
        return self._namespaces.setdefault(namespace, {})

    def create_pod(self, namespace: str, body):
        pod = copy.deepcopy(body)
        pod.metadata.namespace = namespace
        pod.metadata.annotations = dict(pod.metadata.annotations or {})
        pod.metadata.labels = dict(pod.metadata.labels or {})
        pod.metadata.resource_version = str(self._ip_counter)
        pod.metadata.uid = str(uuid.uuid4())
        status = SimpleNamespace(
            phase='Running',
            pod_ip=self._next_ip(),
            container_statuses=[SimpleNamespace(state=SimpleNamespace(waiting=None, terminated=None))]
        )
        pod.status = status
        self._pod_store(namespace)[pod.metadata.name] = pod
        return pod

    def list_pods(self, namespace: str, label_selector: Optional[str]):
        pods = list(self._pod_store(namespace).values())
        if label_selector:
            required: Dict[str, str] = {}
            for part in label_selector.split(','):
                if not part or '=' not in part:
                    continue
                key, value = part.split('=', 1)
                required[key] = value

            def matches(pod) -> bool:
                labels = pod.metadata.labels or {}
                return all(labels.get(key) == expected for key, expected in required.items())

            pods = [p for p in pods if matches(p)]
        metadata = SimpleNamespace(resource_version=str(self._ip_counter))
        return SimpleNamespace(items=pods, metadata=metadata)

    def read_pod(self, namespace: str, name: str):
        pod = self._pod_store(namespace).get(name)
        if not pod:
            raise ApiException(status=404, reason='Not Found')
        return pod

    def delete_pod(self, namespace: str, name: str):
        pod = self._pod_store(namespace).pop(name, None)
        if pod:
            pod.metadata.deletion_timestamp = True


class FakeCoreV1Api:
    def __init__(self, cluster: FakeKubernetesCluster, namespace: str):
        self.cluster = cluster
        self.namespace = namespace

    async def create_namespaced_pod(self, namespace: str, body):
        return self.cluster.create_pod(namespace, body)

    async def list_namespaced_pod(self, namespace: str, label_selector: Optional[str] = None, **_):
        return self.cluster.list_pods(namespace, label_selector)

    async def read_namespaced_pod(self, name: str, namespace: str):
        return self.cluster.read_pod(namespace, name)

    async def delete_namespaced_pod(self, name: str, namespace: str, **_):
        self.cluster.delete_pod(namespace, name)

    async def patch_namespaced_pod(self, name: str, namespace: str, body):
        pod = self.cluster.read_pod(namespace, name)
        if isinstance(body, list):
            for patch in body:
                if patch.get('path') == '/metadata/ownerReferences':
                    pod.metadata.owner_references = patch.get('value')
        return pod


def attach_fake_cluster(controller: K8sEnvironmentController) -> FakeKubernetesCluster:
    cluster = FakeKubernetesCluster()
    api = FakeCoreV1Api(cluster, controller.namespace)

    async def fake_call(operation, description: str, retry: bool = True):
        return await operation(api)

    controller._api_call = fake_call  # type: ignore[attr-defined]
    controller._ws_call = fake_call  # type: ignore[attr-defined]
    return cluster


def disable_pod_cache(controller: K8sEnvironmentController):
    controller._ensure_pod_cache_ready = AsyncMock(return_value=None)


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
        self.created: List[str] = []

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

    async def create_k8s_pod(self, spec, subtype: str, **kwargs):
        self.created.append(subtype)
        return spec

    def has_homepage(self) -> bool:
        return self.config.homepage

    def get_homepage_subtype(self) -> str:
        assert self.config.homepage_subtype
        return self.config.homepage_subtype

    def get_homepage_envs(self, site_urls: Dict[str, str]) -> dict:
        return {'PRIMARY_URL': next(iter(site_urls.values()), '')}


class MetadataDelegation(StubDelegation):
    async def create_k8s_pod(self, spec, subtype: str, **kwargs):
        spec = await super().create_k8s_pod(spec, subtype, **kwargs)
        depends_on = kwargs.get('depends_on')
        if depends_on:
            return spec, {'annotations': {K8S_ANNOTATION_DEPENDS_ON: depends_on}}
        return spec


def default_config() -> DelegationConfig:
    return DelegationConfig(
        name='task',
        subtypes=['exclusive', 'shared', 'homepage'],
        concurrency={'exclusive': 0, 'shared': 2, 'homepage': 0},
        reuse={'exclusive': 1, 'shared': 0, 'homepage': 1},
        service_ports={'exclusive': 80, 'shared': 8080, 'homepage': 8081},
        images={'exclusive': 'exclusive:latest', 'shared': 'shared:latest', 'homepage': 'home:latest'},
        homepage=True,
        homepage_subtype='homepage'
    )


def build_controller(monkeypatch: pytest.MonkeyPatch,
                     delegation: StubDelegation,
                     state: MemoryStateProvider) -> K8sEnvironmentController:
    monkeypatch.setattr(k8s_module, 'create_state_provider', lambda *args, **kwargs: state)
    controller = K8sEnvironmentController(
        delegation=delegation,
        connection={},
        namespace='agent-ns',
        state_driver='dummy'
    )
    return controller


@pytest.mark.asyncio
async def test_start_session_allocates_pods(monkeypatch):
    state = MemoryStateProvider()
    config = default_config()
    delegation = StubDelegation(config)
    controller = build_controller(monkeypatch, delegation, state)
    cluster = attach_fake_cluster(controller)
    disable_pod_cache(controller)

    session_id, pods, urls = await controller.start_session(['exclusive', 'shared'])

    assert set(pods.keys()) == {'exclusive', 'shared', 'homepage'}
    assert urls['exclusive'].startswith('http://')
    assert urls['shared'].endswith(':8080')
    assert urls['homepage'].endswith(':8081')

    session = await state.get_session(session_id)
    assert set(session['exclusive_pods']) == {pods['exclusive'], pods['homepage']}

    shared_pods = [p for p in cluster._pod_store('agent-ns').values()
                   if p.metadata.labels[K8S_LABEL_SUBTYPE_NAME] == 'shared']
    assert len(shared_pods) == 1


@pytest.mark.asyncio
async def test_start_session_reuses_existing_shared_pod(monkeypatch):
    state = MemoryStateProvider()
    controller = build_controller(monkeypatch, StubDelegation(default_config()), state)
    attach_fake_cluster(controller)
    disable_pod_cache(controller)

    existing = await controller.create_pod('shared', exclusive=False)

    session_id, pods, _ = await controller.start_session(['shared'])

    assert pods['shared'] == existing.metadata.name
    assert session_id in state.allocations[existing.metadata.name]


@pytest.mark.asyncio
async def test_concurrency_limit_creates_new_pod(monkeypatch):
    state = MemoryStateProvider()
    config = default_config()
    config.concurrency['shared'] = 1
    controller = build_controller(monkeypatch, StubDelegation(config), state)
    attach_fake_cluster(controller)
    disable_pod_cache(controller)

    busy = await controller.create_pod('shared', exclusive=False)
    state.allocations[busy.metadata.name].add('existing-session')

    _, pods, _ = await controller.start_session(['shared'])

    assert pods['shared'] != busy.metadata.name


@pytest.mark.asyncio
async def test_usage_limit_does_not_spawn_spare_pod(monkeypatch):
    state = MemoryStateProvider()
    config = default_config()
    config.reuse['shared'] = 3
    controller = build_controller(monkeypatch, StubDelegation(config), state)
    cluster = attach_fake_cluster(controller)
    disable_pod_cache(controller)

    await controller.start_session(['shared'])

    shared_pods = [p for p in cluster._pod_store('agent-ns').values()
                   if p.metadata.labels[K8S_LABEL_SUBTYPE_NAME] == 'shared']
    assert len(shared_pods) == 1


@pytest.mark.asyncio
async def test_homepage_pod_receives_primary_url(monkeypatch):
    state = MemoryStateProvider()
    controller = build_controller(monkeypatch, StubDelegation(default_config()), state)
    cluster = attach_fake_cluster(controller)
    disable_pod_cache(controller)

    session_id, _, urls = await controller.start_session(['shared'])
    session = await state.get_session(session_id)
    homepage_id = session['exclusive_pods'][0]
    homepage_pod = cluster.read_pod('agent-ns', homepage_id)
    main_container = homepage_pod.spec.containers[0]
    env_pairs = {env.name: env.value for env in (main_container.env or [])}

    assert env_pairs['PRIMARY_URL'] == urls['shared']


@pytest.mark.asyncio
async def test_get_pod_url_accepts_pod_name(monkeypatch):
    state = MemoryStateProvider()
    controller = build_controller(monkeypatch, StubDelegation(default_config()), state)
    attach_fake_cluster(controller)
    disable_pod_cache(controller)

    pod = await controller.create_pod('shared', exclusive=False)
    url = await controller.get_pod_url(pod.metadata.name, 'shared')

    assert url.endswith(':8080')


@pytest.mark.asyncio
async def test_create_pod_allows_metadata_overrides(monkeypatch):
    state = MemoryStateProvider()
    controller = build_controller(monkeypatch, MetadataDelegation(default_config()), state)
    cluster = attach_fake_cluster(controller)
    disable_pod_cache(controller)

    child = await controller.create_pod('shared', exclusive=False)
    parent = await controller.create_pod('shared', exclusive=False, depends_on=child.metadata.name)

    assert parent.metadata.annotations[K8S_ANNOTATION_DEPENDS_ON] == child.metadata.name
    updated_child = cluster.read_pod('agent-ns', child.metadata.name)
    refs = getattr(updated_child.metadata, 'owner_references', None)
    assert refs and refs[0].name == parent.metadata.name


@pytest.mark.asyncio
async def test_clean_pods_relies_on_owner_gc(monkeypatch):
    state = MemoryStateProvider()
    controller = build_controller(monkeypatch, StubDelegation(default_config()), state)
    cluster = attach_fake_cluster(controller)

    exclusive = await controller.create_pod('exclusive', exclusive=True)

    unhealthy = await controller.create_pod('shared', exclusive=False)
    unhealthy.status.container_statuses[0].state.waiting = SimpleNamespace(
        reason='CrashLoopBackOff', message='boom'
    )

    overused = await controller.create_pod('shared', exclusive=False)
    state.total_uses[overused.metadata.name] = 2
    controller.delegation.config.reuse['shared'] = 2

    parent = await controller.create_pod('shared', exclusive=False)
    child = await controller.create_pod('shared', exclusive=False)
    parent.metadata.annotations[K8S_ANNOTATION_DEPENDS_ON] = child.metadata.name
    parent.status.container_statuses[0].state.waiting = SimpleNamespace(
        reason='CrashLoopBackOff', message='parent unhealthy'
    )

    await controller._clean_pods()

    removed = set(state.removed_containers)
    assert exclusive.metadata.name in removed
    assert unhealthy.metadata.name in removed
    assert overused.metadata.name in removed
    assert parent.metadata.name in removed
    assert child.metadata.name not in removed


@pytest.fixture(autouse=True)
def avoid_real_aiohttp_sessions(monkeypatch):
    class DummySession:
        def __init__(self, *args, **kwargs):
            self.closed = False

        async def close(self):
            self.closed = True

    monkeypatch.setattr(aiohttp, 'ClientSession', DummySession)

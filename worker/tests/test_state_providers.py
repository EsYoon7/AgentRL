from __future__ import annotations

import asyncio
import time
from typing import Dict, Any, Generator, Union
from uuid import uuid4

import httpx
import pytest
import pytest_asyncio
from testcontainers.core.container import DockerContainer
from testcontainers.redis import RedisContainer

from agentrl.worker.environment.state.consul import ConsulStateProvider
from agentrl.worker.environment.state.etcd import EtcdStateProvider
from agentrl.worker.environment.state.local import LocalStateProvider
from agentrl.worker.environment.state.redis import RedisStateProvider


def _wait_http(url: str, timeout: float = 30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = httpx.get(url, timeout=2.0)
            if response.status_code < 500:
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"service at {url} did not become ready")


@pytest.fixture(scope="module")
def redis_service() -> Generator[Dict[str, Union[str, int]], Any, None]:
    container = RedisContainer(image="redis:7")
    container.start()
    host = container.get_container_host_ip()
    port = container.get_exposed_port(6379)
    try:
        yield {"host": host, "port": int(port)}
    finally:
        container.stop()


@pytest.fixture(scope="module")
def consul_service() -> Generator[Dict[str, str], Any, None]:
    container = DockerContainer("hashicorp/consul:1.22")
    container.with_exposed_ports(8500)
    container.with_command("agent -dev -client=0.0.0.0 -bind=0.0.0.0")
    container.with_env("CONSUL_BIND_INTERFACE", "eth0")
    container.start()
    host = container.get_container_host_ip()
    port = container.get_exposed_port(8500)
    url = f"http://{host}:{port}"
    _wait_http(f"{url}/v1/status/leader")
    try:
        yield {"address": url}
    finally:
        container.stop()


@pytest.fixture(scope="module")
def etcd_service() -> Generator[Dict[str, str | int], Any, None]:
    container = DockerContainer("quay.io/coreos/etcd:v3.6.6")
    container.with_env("ALLOW_NONE_AUTHENTICATION", "yes")
    container.with_env("ETCD_ENABLE_V2", "false")
    container.with_env("ETCD_LISTEN_CLIENT_URLS", "http://0.0.0.0:2379")
    container.with_env("ETCD_ADVERTISE_CLIENT_URLS", "http://0.0.0.0:2379")
    container.with_exposed_ports(2379)
    container.start()
    host = container.get_container_host_ip()
    port = container.get_exposed_port(2379)
    _wait_http(f"http://{host}:{port}/health")
    try:
        yield {"host": host, "port": int(port)}
    finally:
        container.stop()


@pytest_asyncio.fixture(params=["local", "redis", "consul", "etcd"], scope="function")
async def state_provider(request):
    name = request.param
    prefix = f"test:{name}:{uuid4()}"
    if name == "local":
        provider = LocalStateProvider()
    elif name == "redis":
        cfg = request.getfixturevalue("redis_service")
        provider = RedisStateProvider({"host": cfg["host"], "port": cfg["port"]}, prefix, None)
    elif name == "consul":
        cfg = request.getfixturevalue("consul_service")
        provider = ConsulStateProvider(cfg["address"], None, None, None, prefix)
    else:
        cfg = request.getfixturevalue("etcd_service")
        provider = EtcdStateProvider({"host": cfg["host"], "port": cfg["port"]}, prefix)
    try:
        yield name, provider
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_session_lifecycle(state_provider):
    _, provider = state_provider
    session_id = await provider.generate_session_id()
    payload = {"containers": {"alpha": "cid"}, "exclusive_containers": []}
    await provider.store_session(session_id, payload)
    fetched = await provider.get_session(session_id)
    assert fetched == payload
    await provider.renew_session(session_id)
    await provider.delete_session(session_id)
    assert await provider.get_session(session_id) is None


@pytest.mark.asyncio
async def test_container_allocation_flow(state_provider):
    _, provider = state_provider
    session_id = await provider.generate_session_id()
    container_id = f"c-{uuid4()}"
    await provider.allocate_container(container_id, session_id)
    assert await provider.container_is_allocated(container_id)
    assert await provider.container_current_uses(container_id) == 1
    assert await provider.container_total_uses(container_id) >= 1
    await provider.release_container(container_id, session_id)
    assert await provider.container_current_uses(container_id) == 0
    await provider.delete_session(session_id)


@pytest.mark.asyncio
async def test_concurrent_allocations(state_provider):
    _, provider = state_provider
    container_id = f"cc-{uuid4()}"
    sessions = [await provider.generate_session_id() for _ in range(4)]

    async def allocate(session_id: str):
        await provider.allocate_container(container_id, session_id)
        await asyncio.sleep(0.01)
        await provider.release_container(container_id, session_id)
        await provider.delete_session(session_id)

    await asyncio.gather(*(allocate(s) for s in sessions))
    assert await provider.container_total_uses(container_id) >= len(sessions)


@pytest.mark.asyncio
async def test_lock_contention(state_provider):
    name, provider = state_provider
    if name == "local":
        pytest.skip("local provider intentionally skips locking semantics")

    active = 0
    max_active = 0
    guard = asyncio.Lock()

    async def worker(idx: int):
        nonlocal active, max_active
        client_id = await provider.generate_session_id()
        async with provider.with_lock("shared", client_id):
            async with guard:
                active += 1
                max_active = max(max_active, active)
            await asyncio.sleep(0.02)
            async with guard:
                active -= 1
        await provider.delete_session(client_id)

    await asyncio.gather(*(worker(i) for i in range(5)))
    assert max_active == 1

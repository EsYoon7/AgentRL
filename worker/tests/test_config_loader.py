from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import pytest
import yaml
from testcontainers.core.container import DockerContainer

from agentrl.worker.config.consul import ConsulConfigLoader
from agentrl.worker.config.loader import ConfigLoader


def _wait_http(url: str, timeout: float = 30.0) -> None:
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


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload))


@pytest.fixture(scope="module")
def consul_service():
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


def test_load_from_handles_imports_across_file_types(tmp_path):
    base = tmp_path / "base.yaml"
    shared = tmp_path / "shared.yaml"
    extra = tmp_path / "extra.json"
    nested = tmp_path / "nested.yaml"

    _write_yaml(shared, {"shared": {"value": 1}})
    extra.write_text(json.dumps({"json": {"enabled": False}}))
    _write_yaml(nested, {"nested": {"nodes": [{"name": "alpha"}, {"name": "beta"}]}})

    _write_yaml(
        base,
        {
            "import": ["shared.yaml", "extra.json"],
            "service": {
                "import": "nested.yaml",
                "feature": True,
            },
        },
    )

    loader = ConfigLoader()
    config = loader.load_from(str(base))

    assert config["shared"]["value"] == 1
    assert config["json"]["enabled"] is False
    assert config["service"]["feature"] is True
    assert config["service"]["nested"]["nodes"][1]["name"] == "beta"


def test_parse_default_and_overwrite_rules(tmp_path):
    path = tmp_path / "defaults.yaml"
    _write_yaml(
        path,
        {
            "jobs": {
                "default": {"timeout": 5, "region": "us-east"},
                "overwrite": {"enabled": True},
                "alpha": {"timeout": 10},
                "beta": {"region": "eu-central"},
                "gamma": {},
            }
        },
    )

    loader = ConfigLoader()
    config = loader.load_from(str(path))

    # default/overwrite blocks are consumed during parsing
    assert "default" not in config["jobs"]
    assert "overwrite" not in config["jobs"]

    assert config["jobs"]["alpha"] == {
        "timeout": 10,
        "region": "us-east",
        "enabled": True,
    }
    assert config["jobs"]["beta"] == {
        "timeout": 5,
        "region": "eu-central",
        "enabled": True,
    }
    assert config["jobs"]["gamma"] == {
        "timeout": 5,
        "region": "us-east",
        "enabled": True,
    }


def test_file_override_via_environment(tmp_path, monkeypatch):
    base = tmp_path / "base.yaml"
    override = tmp_path / "override.yaml"

    _write_yaml(
        base,
        {
            "jobs": {
                "alpha": {"timeout": 10},
                "beta": {"timeout": 20},
            }
        },
    )
    _write_yaml(
        override,
        {
            "jobs": {
                "beta": {"timeout": 25},
                "gamma": {"enabled": True},
            }
        },
    )

    monkeypatch.setenv("AGENTRL_CONFIG_OVERRIDE", str(override))
    loader = ConfigLoader()
    config = loader.load_from(str(base))

    assert config["jobs"]["alpha"]["timeout"] == 10
    assert config["jobs"]["beta"]["timeout"] == 25
    assert config["jobs"]["gamma"]["enabled"] is True


def test_environment_variable_override_casts_values(tmp_path, monkeypatch):
    path = tmp_path / "env.yaml"
    _write_yaml(
        path,
        {
            "service": {
                "port": 8000,
                "debug": False,
                "enabled": True,
                "workers": 4,
                "nested": {"throttle": 0.1, "name": "alpha", "ratio": 2.5},
            }
        },
    )

    monkeypatch.setenv("SERVICE_PORT", "9000")
    monkeypatch.setenv("SERVICE_DEBUG", "true")
    monkeypatch.setenv("SERVICE_ENABLED", "false")
    monkeypatch.setenv("SERVICE_WORKERS", "8")
    monkeypatch.setenv("SERVICE_NESTED_THROTTLE", "0.5")
    monkeypatch.setenv("SERVICE_NESTED_NAME", "omega")
    monkeypatch.setenv("SERVICE_NESTED_RATIO", "3.75")

    loader = ConfigLoader()
    config = loader.load_from(str(path))

    assert config["service"]["port"] == 9000
    assert config["service"]["debug"] is True
    assert config["service"]["enabled"] is False
    assert config["service"]["workers"] == 8
    assert config["service"]["nested"]["throttle"] == 0.5
    assert config["service"]["nested"]["name"] == "omega"
    assert config["service"]["nested"]["ratio"] == 3.75


def test_list_values_are_replaced_not_extended(tmp_path, monkeypatch):
    base = tmp_path / "list-base.yaml"
    override = tmp_path / "list-override.yaml"

    _write_yaml(
        base,
        {
            "service": {
                "hosts": ["a", "b"],
                "stages": [
                    {"name": "alpha", "steps": [1, 2]},
                    {"name": "beta", "steps": [3]},
                ],
            }
        },
    )
    _write_yaml(
        override,
        {
            "service": {
                "hosts": ["override"],
                "stages": [{"name": "gamma", "steps": [9, 10]}],
            }
        },
    )

    monkeypatch.setenv("AGENTRL_CONFIG_OVERRIDE", str(override))
    loader = ConfigLoader()
    config = loader.load_from(str(base))

    assert config["service"]["hosts"] == ["override"]
    assert config["service"]["stages"] == [{"name": "gamma", "steps": [9, 10]}]


def test_consul_override_is_applied(tmp_path, consul_service):
    path = tmp_path / "consul.yaml"
    _write_yaml(path, {"worker": {"retries": 1, "region": "us"}})

    override_payload = {
        "region": "eu",
        "backoff": {"max": 30},
        "new_flag": True,
    }
    consul_url = consul_service["address"]
    response = httpx.put(
        f"{consul_url}/v1/kv/agentrl/config/worker",
        data=json.dumps(override_payload),
        timeout=5.0,
    )
    response.raise_for_status()

    loader = ConfigLoader()
    loader._consul_loader = ConsulConfigLoader(base_url=consul_url, token=None)
    config = loader.load_from(str(path), name="worker")

    assert config["worker"]["retries"] == 1
    assert config["worker"]["region"] == "eu"
    assert config["worker"]["backoff"]["max"] == 30
    assert config["worker"]["new_flag"] is True

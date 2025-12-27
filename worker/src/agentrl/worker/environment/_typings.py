"""
Typing annotations for the environment framework,
useful for type checking in task init parameters
"""

from __future__ import annotations

from typing import Literal, TYPE_CHECKING, TypeAlias, TypedDict, Union, Tuple, List

if TYPE_CHECKING:
    from typing import Optional, Dict

    from ._delegation import EnvironmentDelegation


class StateProviderOptions(TypedDict):
    prefix: Optional[str]


class LocalStateProviderOptions(StateProviderOptions):
    pass


class RedisStateProviderOptions(StateProviderOptions):
    connection: Optional[dict]
    sentinel: Optional[List[Tuple[str, int]]]


class ConsulStateProviderOptions(StateProviderOptions):
    connection: str
    token: Optional[str]
    datacenter: Optional[str]
    namespace: Optional[str]


class EtcdStateProviderOptions(StateProviderOptions):
    connection: Optional[Union[str, dict]]


StateDriver: TypeAlias = Literal['local', 'redis', 'consul', 'etcd']


class EnvironmentControllerOptions(TypedDict):
    delegation: EnvironmentDelegation
    state_driver: StateDriver
    state_options: Optional[StateProviderOptions]


class ManualEnvironmentControllerOptions(EnvironmentControllerOptions):
    urls: Dict[str, str]


class DockerEnvironmentControllerOptions(EnvironmentControllerOptions):
    connection: Optional[dict]
    network_name: str


class K8sEnvironmentControllerOptions(EnvironmentControllerOptions):
    connection: Optional[dict]
    namespace: str


EnvironmentDriver: TypeAlias = Literal['manual', 'docker', 'k8s']

EnvironmentOptions: TypeAlias = EnvironmentControllerOptions

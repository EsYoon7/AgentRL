"""
Typing annotations for the environment framework,
useful for type checking in task init parameters
"""

from __future__ import annotations

from typing import Literal, TYPE_CHECKING, TypeAlias, TypedDict

if TYPE_CHECKING:
    from typing import Optional, Dict

    from ._delegation import EnvironmentDelegation


class StateProviderOptions(TypedDict):
    prefix: Optional[str]


class LocalStateProviderOptions(StateProviderOptions):
    pass


class RedisStateProviderOptions(StateProviderOptions):
    connection: Optional[dict]


StateDriver: TypeAlias = Literal['local', 'redis']


class EnvironmentControllerOptions(TypedDict):
    delegation: EnvironmentDelegation
    state_driver: StateDriver
    state_options: Optional[StateProviderOptions]


class ManualEnvironmentControllerOptions(EnvironmentControllerOptions):
    urls: Dict[str, str]


class DockerEnvironmentControllerOptions(EnvironmentControllerOptions):
    connection: Optional[dict]
    network_name: str


EnvironmentDriver: TypeAlias = Literal['manual', 'docker']

EnvironmentOptions: TypeAlias = EnvironmentControllerOptions

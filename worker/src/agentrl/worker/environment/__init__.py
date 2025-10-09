from ._base import EnvironmentController
from ._delegation import EnvironmentDelegation


def create_controller(driver: str, delegation: EnvironmentDelegation, **config) -> EnvironmentController:
    if driver == 'manual':
        from .manual import ManualEnvironmentController
        return ManualEnvironmentController(
            delegation,
            config.get('urls', {})
        )

    if driver == 'docker':
        from .docker import DockerEnvironmentController
        return DockerEnvironmentController(
            delegation,
            config.get('connection', {}),
            config['network_name'],
            config.get('state_driver', 'redis'),
            config.get('state_options', {})
        )

    raise ValueError(f'Unknown environment controller driver: {driver}')

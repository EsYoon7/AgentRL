from ._base import EnvironmentController
from ._const import K8S_LABEL_KVM, K8S_LABEL_BTRFS
from ._delegation import EnvironmentDelegation
from ._typings import EnvironmentDriver, EnvironmentOptions


def create_controller(driver: EnvironmentDriver, delegation: EnvironmentDelegation, **config) -> EnvironmentController:
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

    if driver == 'k8s':
        from .k8s import K8sEnvironmentController
        return K8sEnvironmentController(
            delegation,
            config.get('connection'),
            config['namespace'],
            config.get('state_driver', 'redis'),
            config.get('state_options', {})
        )

    raise ValueError(f'Unknown environment controller driver: {driver}')

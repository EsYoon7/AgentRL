import os

from .._typings import StateDriver
from ._base import StateProvider


def create_state_provider(driver: StateDriver, prefix: str = '', **config) -> StateProvider:
    if driver == 'local':
        from .local import LocalStateProvider
        return LocalStateProvider()

    if driver == 'redis':
        from .redis import RedisStateProvider
        return RedisStateProvider(
            config.get('connection', {}),
            prefix,
            config.get('sentinel')
        )

    if driver == 'consul':
        from .consul import ConsulStateProvider
        return ConsulStateProvider(
            config.get('connection', os.getenv('CONSUL_HTTP_ADDR')),
            config.get('token', os.getenv('CONSUL_HTTP_TOKEN')),
            config.get('datacenter'),
            config.get('namespace'),
            prefix
        )

    if driver == 'etcd':
        from .etcd import EtcdStateProvider
        return EtcdStateProvider(
            config.get('connection', {}),
            prefix
        )

    raise ValueError(f'Unknown lock provider driver: {driver}')

from ._base import StateProvider


def create_state_provider(driver: str, prefix: str = '', **config) -> StateProvider:
    if driver == 'local':
        from .local import LocalStateProvider
        return LocalStateProvider()

    if driver == 'redis':
        from .redis import RedisStateProvider
        return RedisStateProvider(
            config.get('connection', {}),
            prefix
        )

    raise ValueError(f'Unknown lock provider driver: {driver}')

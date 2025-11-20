from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from typing import Any, Optional, Type, Union

from .types import Listener, T


class EventBus:

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.listeners: dict[str, set[Listener]] = {}

    def subscribe(self, event_type: Union[Type[T], str], listener: Listener[T]):
        if not isinstance(event_type, str):
            event_type = event_type.model_fields['type'].default
        if event_type not in self.listeners:
            self.listeners[event_type] = set()
        self.listeners[event_type].add(listener)

    def unsubscribe(self, event_type: Union[Type[T], str], listener: Listener[T]):
        if not isinstance(event_type, str):
            event_type = event_type.model_fields['type'].default
        if event_type not in self.listeners:
            return
        if listener in self.listeners[event_type]:
            self.listeners[event_type].remove(listener)

    async def publish(self, event: T):
        listeners = list(self.listeners.get(event.type, []))

        futures: list[Coroutine[Any, Any, None]] = []
        for listener in listeners:
            try:
                result = listener(event)
                if asyncio.iscoroutine(result):
                    futures.append(result)
            except Exception:
                self.logger.exception('error handling event "%s"', event.type)
        if futures:
            results = await asyncio.gather(*futures, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    self.logger.error('error handling event "%s"', event.type, exc_info=result)

    def publish_defer(self, event: T):
        asyncio.create_task(self.publish(event))

    def publish_sync(self, event: T):
        asyncio.run_coroutine_threadsafe(self.publish(event), asyncio.get_event_loop())

    async def wait_for(
        self,
        event_type: Union[Type[T], str],
        check: Optional[Callable[[T], bool]] = None,
        timeout: Optional[float] = None,
    ) -> Optional[T]:
        """
        Wait for the next event of the given type.
        Optionally filter with a check() and support timeout.
        """
        future: asyncio.Future[T] = asyncio.get_event_loop().create_future()

        def _listener(event: T):
            if check is None or check(event):
                if not future.done():
                    future.set_result(event)

        self.subscribe(event_type, _listener)

        try:
            return await asyncio.wait_for(future, timeout)
        finally:
            self.unsubscribe(event_type, _listener)

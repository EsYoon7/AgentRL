from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any, Dict, List, Optional

from agentrl.worker.environment.state import StateProvider


class MemoryStateProvider(StateProvider):
    """Deterministic in-memory state provider tailored for tests."""

    def __init__(self):
        self._session_seq = 1
        self.sessions: Dict[str, Any] = {}
        self.allocations: Dict[str, set[str]] = defaultdict(set)
        self.total_uses: Dict[str, int] = defaultdict(int)
        self.removed_containers: List[str] = []
        self._locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def acquire_lock(self, lock_name: str, client_id: Optional[str] = None, timeout: int = 10) -> bool:
        lock = self._locks[lock_name]
        try:
            return await asyncio.wait_for(lock.acquire(), timeout=timeout)
        except asyncio.TimeoutError:
            return False

    async def release_lock(self, lock_name: str, client_id: Optional[str] = None):
        lock = self._locks[lock_name]
        if lock.locked():
            lock.release()

    async def allocate_container(self, container_id: str, session_id: str):
        self.allocations[container_id].add(session_id)
        self.total_uses[container_id] += 1

    async def renew_container(self, container_id: str, session_id: str):
        return None

    async def container_is_allocated(self, container_id: str) -> bool:
        return bool(self.allocations.get(container_id))

    async def container_current_uses(self, container_id: str) -> int:
        return len(self.allocations.get(container_id, set()))

    async def container_total_uses(self, container_id: str) -> int:
        return self.total_uses.get(container_id, 0)

    async def containers_total_uses_gte(self, min_uses: int) -> List[str]:
        return [cid for cid, uses in self.total_uses.items() if uses >= min_uses]

    async def release_container(self, container_id: str, session_id: str):
        self.allocations.get(container_id, set()).discard(session_id)

    async def remove_container(self, container_id: str):
        self.allocations.pop(container_id, None)
        self.total_uses.pop(container_id, None)
        self.removed_containers.append(container_id)

    async def generate_session_id(self) -> str:
        sid = f'session-{self._session_seq}'
        self._session_seq += 1
        return sid

    async def store_session(self, session_id: str, data: Any):
        self.sessions[session_id] = data

    async def renew_session(self, session_id: str):
        return None

    async def get_session(self, session_id: str) -> Optional[Any]:
        return self.sessions.get(session_id)

    async def delete_session(self, session_id: str):
        self.sessions.pop(session_id, None)

    async def close(self):
        return None


class TracingStateProvider(MemoryStateProvider):
    def __init__(self):
        super().__init__()
        self.active_locks: Dict[str, int] = defaultdict(int)
        self.max_lock_holders: Dict[str, int] = defaultdict(int)

    async def acquire_lock(self, lock_name: str, client_id: Optional[str] = None, timeout: int = 10) -> bool:
        acquired = await super().acquire_lock(lock_name, client_id, timeout)
        if acquired:
            self.active_locks[lock_name] += 1
            self.max_lock_holders[lock_name] = max(
                self.max_lock_holders[lock_name],
                self.active_locks[lock_name]
            )
        return acquired

    async def release_lock(self, lock_name: str, client_id: Optional[str] = None):
        if self.active_locks[lock_name] > 0:
            self.active_locks[lock_name] -= 1
        await super().release_lock(lock_name, client_id)

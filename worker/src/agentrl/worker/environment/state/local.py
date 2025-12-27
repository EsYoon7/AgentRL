from typing import Optional, Any, Set, Dict, List

from ._base import StateProvider


class LocalStateProvider(StateProvider):
    """
    Basic state provider that stores state in memory.
    No persistence, expiry and locking is implemented.
    For testing purposes only.
    """

    def __init__(self):
        self.containers: Dict[str, Set[str]] = {}
        self.container_uses: Dict[str, int] = {}
        self.sessions: Dict[str, Any] = {}

    async def acquire_lock(self, lock_name: str, client_id: Optional[str] = None, timeout: int = 10) -> bool:
        return True

    async def release_lock(self, lock_name: str, client_id: Optional[str] = None):
        pass

    async def allocate_container(self, container_id: str, session_id: str):
        if container_id not in self.container_uses:
            self.containers[container_id] = set()
        self.containers[container_id].add(session_id)
        if container_id not in self.container_uses:
            self.container_uses[container_id] = 0
        self.container_uses[container_id] += 1

    async def renew_container(self, container_id: str, session_id: str):
        pass

    async def container_is_allocated(self, container_id: str) -> bool:
        return container_id in self.containers and len(self.containers[container_id]) > 0

    async def release_container(self, container_id: str, session_id: str):
        if container_id in self.containers:
            self.containers[container_id].discard(session_id)

    async def container_current_uses(self, container_id: str) -> int:
        return len(self.containers.get(container_id, set()))

    async def container_total_uses(self, container_id: str) -> int:
        return self.container_uses.get(container_id, 0)

    async def containers_total_uses_gte(self, min_uses: int) -> List[str]:
        return [
            container_id
            for container_id, uses in self.container_uses.items()
            if uses >= min_uses
        ]

    async def store_session(self, session_id: str, data: Any):
        self.sessions[session_id] = data

    async def renew_session(self, session_id: str):
        pass

    async def get_session(self, session_id: str) -> Optional[Any]:
        return self.sessions.get(session_id)

    async def delete_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]

    async def remove_container(self, container_id: str):
        if container_id in self.containers:
            del self.containers[container_id]
        if container_id in self.container_uses:
            del self.container_uses[container_id]

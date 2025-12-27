import asyncio
import random
from contextlib import asynccontextmanager
from typing import Optional, Any, List
from uuid import uuid4


class StateProvider:
    """
    Abstract exclusive lock and state storage provider.
    """

    async def acquire_lock(self, lock_name: str, client_id: Optional[str] = None, timeout: int = 10) -> bool:
        """Acquire a lock to ensure mutual exclusivity."""
        raise NotImplementedError

    async def release_lock(self, lock_name: str, client_id: Optional[str] = None):
        """Release the acquired lock."""
        raise NotImplementedError

    @asynccontextmanager
    async def with_lock(self, lock_name: str, client_id: Optional[str] = None, timeout: int = 10):
        """The two locking methods wrapped as an async context manager."""
        while True:
            if await self.acquire_lock(lock_name, client_id, timeout):
                break
            await asyncio.sleep(random.uniform(1, 2))
        try:
            yield
        finally:
            await self.release_lock(lock_name, client_id)

    async def allocate_container(self, container_id: str, session_id: str):
        """Record allocation of a container to a session."""
        raise NotImplementedError

    async def renew_container(self, container_id: str, session_id: str):
        """Renew the allocation of a container to a session."""
        raise NotImplementedError

    async def container_is_allocated(self, container_id: str) -> bool:
        """Check if a container is allocated to a session."""
        raise NotImplementedError

    async def container_current_uses(self, container_id: str) -> int:
        """Get the current sessions using a container."""
        raise NotImplementedError

    async def container_total_uses(self, container_id: str) -> int:
        """Get the total sessions that have used a container."""
        raise NotImplementedError

    async def containers_total_uses_gte(self, min_uses: int) -> List[str]:
        """Get a list of containers with total uses greater than or equal to a specified number."""
        raise NotImplementedError

    async def release_container(self, container_id: str, session_id: str):
        """Release the allocation of a container to a session."""
        raise NotImplementedError

    async def remove_container(self, container_id: str):
        """Remove all keys related to a container."""
        raise NotImplementedError

    async def generate_session_id(self) -> str:
        return str(uuid4())

    async def store_session(self, session_id: str, data: Any):
        """Store session data."""
        raise NotImplementedError

    async def renew_session(self, session_id: str):
        raise NotImplementedError

    async def get_session(self, session_id: str) -> Optional[Any]:
        """Retrieve session data."""
        raise NotImplementedError

    async def delete_session(self, session_id: str):
        """Delete session data."""
        raise NotImplementedError

    async def close(self):
        """Cleanup resources held by the state provider."""
        pass

from __future__ import annotations

from abc import ABC
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..convert import MessageRecord, FunctionDefinition


class BaseClient(ABC):

    async def get_model_name(self) -> str:
        raise NotImplementedError

    async def query(self,
                    messages: list[MessageRecord],
                    tools: Optional[list[FunctionDefinition]] = None,
                    cache_key: Optional[str] = None) -> list[MessageRecord]:
        raise NotImplementedError

    async def close(self):
        pass

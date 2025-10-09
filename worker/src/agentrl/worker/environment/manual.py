from __future__ import annotations

from typing import Union, List, Tuple, Dict, TYPE_CHECKING

from ._base import EnvironmentController

if TYPE_CHECKING:
    from ._delegation import EnvironmentDelegation


class ManualEnvironmentController(EnvironmentController):
    def __init__(self, delegation: EnvironmentDelegation, urls: Dict[str, str]):
        super().__init__(delegation)
        self.urls = urls

    async def start_session(self, subtypes: Union[List[str], str], immutable: bool = True, **kwargs) -> Tuple[str, Dict[str, str], Dict[str, str]]:
        return 'manual', {}, self.urls

    async def renew_session(self, session_id: str):
        pass

    async def end_session(self, session_id: str):
        pass

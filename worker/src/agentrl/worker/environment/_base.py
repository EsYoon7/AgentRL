from __future__ import annotations

import asyncio
from typing import List, Union, Tuple, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from asyncio import EventLoop
    from ._delegation import EnvironmentDelegation


class EnvironmentController:
    """
    Abstract controller for managing external environments.
    """

    loop: EventLoop  # for sync code that calls the controller

    def __init__(self, delegation: EnvironmentDelegation):
        self.delegation = delegation
        self.delegation.controller = self

    async def start_session(self, subtypes: Union[List[str], str], immutable: bool = True, **kwargs) -> Tuple[str, Dict[str, str], Dict[str, str]]:
        """
        Claim using a subtype or a list of subtype of environments, get a session id, the environment ids and the url map.
        Some subtypes (i.e., wikipedia and map in webarena) do not involve mutating actions, so they should always be available.
        Extra parameters may be passed to the delegation.
        """
        raise NotImplementedError

    def sync_start_session(self, *args, **kwargs) -> Tuple[str, Dict[str, str], Dict[str, str]]:
        return asyncio.run_coroutine_threadsafe(
            self.start_session(*args, **kwargs),
            self.loop
        ).result()

    async def renew_session(self, session_id: str):
        """
        Indicate that a session is still being used, and should not expire.
        """
        raise NotImplementedError

    def sync_renew_session(self, session_id: str):
        asyncio.run_coroutine_threadsafe(
            self.renew_session(session_id),
            self.loop
        ).result()

    async def end_session(self, session_id: str):
        """
        Release a session, indicating that these websites are done using and maybe released or reused by other tasks.
        If writes is set when creating the session, the websites used should be destroyed and recreated as they may be polluted.
        """
        raise NotImplementedError

    def sync_end_session(self, session_id: str):
        asyncio.run_coroutine_threadsafe(
            self.end_session(session_id),
            self.loop
        ).result()

    async def execute_command(self, environment_id: str, command: Union[str, List[str]], timeout: int = 30) -> Tuple[int, bytes, bytes]:
        """
        Execute a command in the environment. For example, exec in a Docker container.
        Returns the return code, stdout and stderr.
        """
        raise NotImplementedError

    def sync_execute_command(self, *args, **kwargs) -> Tuple[int, bytes, bytes]:
        return asyncio.run_coroutine_threadsafe(
            self.execute_command(*args, **kwargs),
            self.loop
        ).result()

    async def create_shell(self, environment_id: str, shell: str = '/bin/bash'):
        """
        Creates an interactive shell in the environment. For example, exec with stdin in a Docker container.
        Returns the shell id for further use.
        """
        raise NotImplementedError

    async def execute_shell(self, environment_id: str, command: str, timeout: int = 30) -> bytes:
        """
        Execute a command in the shell.
        Returns the stdout and stderr combined.
        """
        raise NotImplementedError

    def sync_execute_shell(self, *args, **kwargs) -> bytes:
        return asyncio.run_coroutine_threadsafe(
            self.execute_shell(*args, **kwargs),
            self.loop
        ).result()

    async def get_env_variables(self, environment_id: str) -> Dict[str, str]:
        """
        Get the environment variables of the environment.
        """
        raise NotImplementedError

    def sync_get_env_variables(self, environment_id: str) -> Dict[str, str]:
        return asyncio.run_coroutine_threadsafe(
            self.get_env_variables(environment_id),
            self.loop
        ).result()

    async def background_task(self):
        pass

from __future__ import annotations

from copy import copy
from typing import Optional

from httpx import AsyncClient, Response, Timeout
from openai.types.chat import ChatCompletionAssistantMessageParam

from .types import InteractResponse, TaskIndex


class ControllerClient:

    def __init__(self,
                 base_url: str,
                 *,
                 proxy_url: Optional[str],
                 insecure: bool = False):
        self.base_url = base_url
        self.proxy_url = proxy_url
        self.insecure = insecure

        self._client: Optional[AsyncClient] = None

    async def _get_client(self) -> AsyncClient:
        if self._client is None:
            self._client = AsyncClient(
                base_url=self.base_url,
                http2=True,
                proxy=self.proxy_url,
                timeout=Timeout(None, connect=5.0),
                verify=not self.insecure
            )

        return self._client

    async def get_indices(self, task: str) -> list[TaskIndex]:
        client = await self._get_client()
        response = await client.get('get_indices', params={
            'name': task
        })
        self._raise_for_status(response)
        return response.json()

    async def start_sample(self,
                           task: str,
                           index: TaskIndex,
                           custom_params: Optional[dict] = None) -> tuple[int, InteractResponse]:
        if custom_params:
            custom_task = copy(custom_params)
            if index != -1:
                custom_task['index'] = index
            index = -1
        else:
            custom_task = None
            assert index != -1, 'custom task is not supported'

        client = await self._get_client()
        response = await client.post(f'start_sample', json={
            'name': task,
            'index': index,
            'custom_task': custom_task
        })
        self._raise_for_status(response)
        session_id = int(response.headers['session_id'])
        return session_id, InteractResponse.model_validate(response.json())

    async def interact(self, session_id: int, messages: list[dict]) -> InteractResponse:
        client = await self._get_client()
        response = await client.post(f'interact', headers={
            'session_id': str(session_id)
        }, json={
            'messages': messages
        })
        self._raise_for_status(response)
        return InteractResponse.model_validate(response.json())

    async def renew(self, session_id: int) -> None:
        """
        Send a message with empty content to not perform any action while renewing the session.
        Warning: use on supported tasks only.
        """
        await self.interact(session_id, messages=[
            dict(ChatCompletionAssistantMessageParam(
                role='assistant',
                content=''
            ))
        ])

    async def cancel(self, session_id: int) -> None:
        client = await self._get_client()
        await client.post(f'cancel', headers={
            'session_id': str(session_id)
        })  # ignore result

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @staticmethod
    def _raise_for_status(response: Response):
        try:
            response.raise_for_status()
        except Exception as e:
            raise RuntimeError(f'controller request to {response.url} failed: {response.text}') from e

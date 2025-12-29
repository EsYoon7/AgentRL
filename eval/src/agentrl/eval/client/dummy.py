from __future__ import annotations

import asyncio
from typing import Optional, Literal

from openai.types.chat import ChatCompletionAssistantMessageParam
from pydantic import BaseModel, Field

from ._base import BaseClient
from ..convert import FunctionDefinition, MessageRecord, OpenAIChatCompletionInputMessageRecord


class DummyOptions(BaseModel):
    provider: Literal['dummy'] = 'dummy'
    model: Optional[str] = None
    interval: float = Field(default=1.0, ge=0.0)


class DummyClient(BaseClient):
    """
    Some environments already have a built-in agent that does the actual interaction,
    so we might just need a dummy model client that does nothing,
    but occasionally sends a response to keep the session alive.
    """

    def __init__(self, options: DummyOptions, *args, **kwargs):
        self.model = options.model
        self.interval = options.interval

        self._first_query = True

    async def get_model_name(self) -> str:
        if self.model is not None:
            return self.model
        return 'dummy'

    async def query(self,
                    messages: list[MessageRecord],
                    tools: Optional[list[FunctionDefinition]] = None,
                    cache_key: Optional[str] = None) -> list[MessageRecord]:
        if self.interval > 0 and not self._first_query:
            await asyncio.sleep(self.interval)
        self._first_query = False
        return [OpenAIChatCompletionInputMessageRecord([
            ChatCompletionAssistantMessageParam(
                role='assistant',
                content='Are you done?'
            )
        ])]

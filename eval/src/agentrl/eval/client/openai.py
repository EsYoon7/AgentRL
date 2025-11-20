from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Literal, Optional, TYPE_CHECKING

from httpx import AsyncClient, Timeout
from openai import AsyncOpenAI, BadRequestError, InternalServerError
from openai.types import Reasoning
from pydantic import BaseModel, Field, SecretStr
from pydantic_core import Url

from ._base import BaseClient
from ..convert import (FunctionDefinition,
                       MessageRecord,
                       OpenAIChatCompletionOutputMessageRecord,
                       OpenAIResponseOutputMessageRecord)
from ..utils import normalize_model_name

if TYPE_CHECKING:
    from ..session.tokens import TokenCounter


class OpenAIOptions(BaseModel):
    provider: Literal['openai'] = 'openai'
    model: Optional[str] = None
    api_key: Optional[SecretStr] = None
    base_url: Optional[Url] = None
    proxy_url: Optional[Url] = None
    temperature: Optional[float] = Field(default=0.8, ge=0.0, le=1.0)
    parallel_tool_calls: Optional[bool] = True
    max_output_tokens: Optional[int] = Field(default=16000, ge=1024)
    max_retries: int = Field(default=2, ge=0)
    chat_completions: bool = False
    thinking: bool = True
    insecure: bool = False


class OpenAIClient(BaseClient):

    def __init__(self,
                 options: OpenAIOptions,
                 token_counter: Optional[TokenCounter] = None):
        self.logger = logging.getLogger(__name__)

        self.model = options.model
        self.api_key = options.api_key
        self.base_url = str(options.base_url) if options.base_url is not None else None
        self.proxy_url = str(options.proxy_url) if options.proxy_url is not None else None
        self.temperature = options.temperature
        self.parallel_tool_calls = options.parallel_tool_calls
        self.max_output_tokens = options.max_output_tokens
        self.max_retries = options.max_retries
        self.use_thinking = options.thinking
        self.insecure = options.insecure
        self.token_counter = token_counter

        self._client: Optional[AsyncOpenAI] = None
        self._use_responses = not options.chat_completions

    async def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            timeout = Timeout(None, connect=5.0)
            self._client = AsyncOpenAI(
                api_key=self.api_key.get_secret_value() if self.api_key else '_not_needed_',  # no key required for self-hosted models
                base_url=self.base_url,
                timeout=timeout,
                max_retries=self.max_retries,
                http_client=AsyncClient(
                    http2=True,
                    proxy=self.proxy_url,
                    timeout=timeout,
                    verify=not self.insecure
                ),
            )

        return self._client

    async def get_model_name(self) -> str:
        if self.model is not None:
            return self.model

        if self.base_url is None:
            raise ValueError('Model name must be specified when using OpenAI API')

        client = await self._get_client()
        async for model in client.models.list():
            self.model = model.id
            self.logger.info('using model %s', self.model)
            return self.model

        raise ValueError('No models available')

    async def query(self,
                    messages: list[MessageRecord],
                    tools: Optional[list[FunctionDefinition]] = None,
                    cache_key: Optional[str] = None) -> list[MessageRecord]:
        if self._use_responses:
            return await self._query_responses(messages, tools, cache_key)
        return await self._query_chat_completions(messages, tools, cache_key)

    async def _reasoning_params(self) -> Optional[Reasoning]:
        model = await self.get_model_name()
        model = normalize_model_name(model)

        if 'thinking' in model or model.startswith('o') or model.startswith('gpt-5'):
            # gpt-5.1 supports toggling thinking mode, and is disabled by default
            if model.startswith('gpt-5.1') and not self.use_thinking:
                return None
            return Reasoning(
                effort='medium' if not model.startswith('gpt-5-pro') else 'high',
                summary='auto' if not model.startswith('computer-use') else 'concise'
            )

        return None

    async def _query_responses(self,
                               messages: list[MessageRecord],
                               tools: Optional[list[FunctionDefinition]] = None,
                               cache_key: Optional[str] = None) -> list[MessageRecord]:
        client = await self._get_client()
        model = await self.get_model_name()
        thinking = await self._reasoning_params()

        try:
            response = await client.responses.create(
                input=MessageRecord.convert_all(messages, 'openai_response_input'),
                max_output_tokens=self.max_output_tokens,
                model=model,
                parallel_tool_calls=self.parallel_tool_calls if tools else None,
                prompt_cache_key=cache_key,
                reasoning=thinking,
                temperature=self.temperature if not thinking else None,
                tools=FunctionDefinition.convert_all(tools, 'openai_response'),
                truncation='auto'
            )
        except (BadRequestError, InternalServerError) as e:
            self.logger.warning('OpenAI Responses API error: %s, falling back to Chat Completions API', e)
            self._use_responses = False
            return await self._query_chat_completions(messages, tools)

        if self.token_counter and hasattr(response, 'usage') and response.usage:
            self.token_counter.add_from_usage(response.usage)

        for error_key in {'error', 'message', 'error_message'}:
            if getattr(response, error_key, None):
                self.logger.debug('API response: %s', response)
                raise RuntimeError(f'API error: {getattr(response, error_key)}')

        for message in response.output:
            if hasattr(message, 'content') and isinstance(message.content, Iterable):
                for part in message.content:
                    if part.type == 'refusal':
                        raise RuntimeError(part.refusal)

        return [OpenAIResponseOutputMessageRecord(response.output)]

    async def _query_chat_completions(self,
                                      messages: list[MessageRecord],
                                      tools: Optional[list[FunctionDefinition]] = None,
                                      cache_key: Optional[str] = None) -> list[MessageRecord]:
        client = await self._get_client()
        model = await self.get_model_name()
        thinking = await self._reasoning_params()
        response = await client.chat.completions.create(
            messages=MessageRecord.convert_all(messages, 'openai_chat_completion_input'),
            model=model,
            max_completion_tokens=self.max_output_tokens,
            parallel_tool_calls=self.parallel_tool_calls if tools else None,
            prompt_cache_key=cache_key,
            reasoning_effort=thinking.effort if thinking else None,
            temperature=self.temperature if not thinking else None,
            tools=FunctionDefinition.convert_all(tools, 'openai_chat_completion')
        )

        if self.token_counter and hasattr(response, 'usage') and response.usage:
            self.token_counter.add_from_usage(response.usage)

        for error_key in {'error', 'message', 'error_message'}:
            if getattr(response, error_key, None):
                self.logger.debug('API response: %s', response)
                raise RuntimeError(f'API error: {getattr(response, error_key)}')

        if response.choices[0].message.refusal:
            raise RuntimeError(response.choices[0].message.refusal)
        if response.choices[0].finish_reason == 'content_filter':
            raise RuntimeError('Content filter triggered')

        return [OpenAIChatCompletionOutputMessageRecord([response.choices[0].message])]

    async def close(self):
        if self._client is not None:
            await self._client.close()
            self._client = None

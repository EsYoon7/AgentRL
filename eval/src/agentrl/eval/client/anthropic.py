from __future__ import annotations

import logging
from functools import cached_property
from typing import Literal, Optional, TYPE_CHECKING, Union, Any

from anthropic import AsyncAnthropic, AsyncAnthropicBedrock
from anthropic.types import (AnthropicBetaParam,
                             CacheControlEphemeralParam,
                             ThinkingConfigDisabledParam,
                             ThinkingConfigEnabledParam,
                             ToolChoiceAutoParam)
from httpx import AsyncClient, Request, Timeout, URL, Response
from pydantic import BaseModel, Field, SecretStr
from pydantic_core import Url

from ._base import BaseClient
from ..convert import AnthropicMessageOutputMessageRecord, FunctionDefinition, MessageRecord
from ..utils import resize_images, trim_images

if TYPE_CHECKING:
    from ..session.tokens import TokenCounter


class AnthropicOptions(BaseModel):
    provider: Literal['anthropic'] = 'anthropic'
    model: Optional[str] = None
    api_key: Optional[SecretStr] = None
    base_url: Optional[Url] = None
    proxy_url: Optional[Url] = None
    override_url: Optional[Url] = None
    thinking: bool = False
    computer_use: bool = False
    temperature: Optional[float] = Field(default=0.8, ge=0.0, le=1.0)
    parallel_tool_calls: Optional[bool] = True
    max_thinking_tokens: int = Field(default=10000, ge=1024)
    max_output_tokens: Optional[int] = Field(default=16000, ge=1024)
    max_retries: int = Field(default=2, ge=0)
    max_images: Optional[int] = Field(default=None, ge=0)
    image_size: Optional[str] = None
    extra_body: Optional[dict[str, Any]] = None
    extra_headers: Optional[dict[str, str]] = None
    insecure: bool = False


class BedrockOptions(AnthropicOptions):
    provider: Literal['bedrock'] = 'bedrock'
    aws_access_key: SecretStr
    aws_secret_key: SecretStr
    aws_region: str


class CustomAsyncClient(AsyncClient):

    def __init__(self,
                 model: Optional[str] = None,
                 api_key: Optional[str] = None,
                 override_url: Optional[str] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.api_key = api_key
        self.override_url = override_url

    def build_request(self, **kwargs) -> Request:
        # some internal providers require a pinned URL and possibly a different authentication method
        if self.override_url is not None:
            kwargs['url'] = URL(self.override_url)

        # the anthropic client might have removed our custom model name, so we need to restore it
        if self.model is not None:
            if isinstance(kwargs.get('json'), dict):
                kwargs['json']['model'] = self.model

        return super().build_request(**kwargs)

    async def send(self, request: Request, **kwargs) -> Response:
        # if override url and api key is both set, we override the authentication method here
        if self.override_url is not None and self.api_key is not None:
            request.headers['Authorization'] = f'Bearer {self.api_key}'

        return await super().send(request, **kwargs)


class AnthropicClient(BaseClient):

    def __init__(self,
                 options: AnthropicOptions,
                 token_counter: Optional[TokenCounter] = None):
        self.logger = logging.getLogger(__name__)

        self.bedrock = options.provider == 'bedrock'
        self.aws_access_key = getattr(options, 'aws_access_key', None)
        self.aws_secret_key = getattr(options, 'aws_secret_key', None)
        self.aws_region = getattr(options, 'aws_region', None)
        self.model = options.model
        self.api_key = options.api_key
        self.base_url = str(options.base_url) if options.base_url is not None else None
        self.proxy_url = str(options.proxy_url) if options.proxy_url is not None else None
        self.override_url = str(options.override_url) if options.override_url is not None else None
        self.thinking = options.thinking
        self.computer_use = options.computer_use
        self.temperature = options.temperature
        self.parallel_tool_calls = options.parallel_tool_calls
        self.max_thinking_tokens = options.max_thinking_tokens
        self.max_output_tokens = options.max_output_tokens
        self.max_retries = options.max_retries
        self.max_images = options.max_images
        self.image_size = options.image_size
        self.extra_body = options.extra_body
        self.extra_headers = options.extra_headers
        self.insecure = options.insecure
        self.token_counter = token_counter

        self._client: Optional[Union[AsyncAnthropic, AsyncAnthropicBedrock]] = None

    @cached_property
    def betas(self) -> list[AnthropicBetaParam]:
        betas = []

        if self.thinking:
            betas.append('interleaved-thinking-2025-05-14')

        if self.computer_use:
            betas.append('computer-use-2025-01-24')

        return betas

    async def _get_client(self) -> Union[AsyncAnthropic, AsyncAnthropicBedrock]:
        if self._client is None:
            timeout = Timeout(None, connect=5.0)
            http_client = CustomAsyncClient(
                model=self.model,
                api_key=self.api_key.get_secret_value() if self.api_key else None,
                override_url=self.override_url,
                http2=True,
                proxy=self.proxy_url,
                timeout=timeout,
                verify=not self.insecure
            )

            if self.bedrock:
                self._client = AsyncAnthropicBedrock(
                    aws_secret_key=self.aws_secret_key.get_secret_value(),
                    aws_access_key=self.aws_access_key.get_secret_value(),
                    aws_region=self.aws_region,
                    base_url=self.base_url,
                    timeout=timeout,
                    max_retries=self.max_retries,
                    http_client=http_client,
                )
            else:
                self._client = AsyncAnthropic(
                    api_key=self.api_key.get_secret_value() if self.api_key else '_not_needed_',
                    base_url=self.base_url,
                    timeout=timeout,
                    max_retries=self.max_retries,
                    http_client=http_client,
                )

        return self._client

    async def get_model_name(self) -> str:
        if self.model is not None:
            return self.model

        raise ValueError('model name must be set')

    async def query(self,
                    messages: list[MessageRecord],
                    tools: Optional[list[FunctionDefinition]] = None,
                    cache_key: Optional[str] = None) -> list[MessageRecord]:
        client = await self._get_client()
        model = await self.get_model_name()

        # modify prompts to add ephemeral cache control
        system = MessageRecord.convert_all(messages, to='anthropic_message_system', ignore_errors=True)
        if system:
            system[-1]['cache_control'] = CacheControlEphemeralParam(type='ephemeral')
        messages = MessageRecord.convert_all(messages, to='anthropic_message_input', ignore_errors=True)
        for message in messages:
            if message.get('role') == 'assistant':
                message['cache_control'] = CacheControlEphemeralParam(type='ephemeral')
        tools = FunctionDefinition.convert_all(tools, to='anthropic_message')
        if tools:
            tools[-1]['cache_control'] = CacheControlEphemeralParam(type='ephemeral')

        # apply message filter if set
        if self.max_images is not None:
            messages = trim_images(messages, self.max_images)
        if self.image_size is not None:
            messages = resize_images(messages, self.image_size)

        response = await client.beta.messages.create(
            model=model,
            system=system,
            messages=messages,
            max_tokens=self.max_output_tokens,
            temperature=self.temperature if not self.thinking else 1.0,
            thinking=ThinkingConfigEnabledParam(
                type='enabled',
                budget_tokens=self.max_thinking_tokens
            ) if self.thinking else ThinkingConfigDisabledParam(
                type='disabled'
            ),
            tool_choice=ToolChoiceAutoParam(
                type='auto',
                disable_parallel_tool_use=self.parallel_tool_calls is not True
            ) if tools else None,
            tools=tools,
            betas=self.betas,
            extra_body=self.extra_body,
            extra_headers=self.extra_headers
        )

        if self.token_counter and hasattr(response, 'usage') and response.usage:
            self.token_counter.add_from_usage(response.usage)

        for error_key in {'error', 'message', 'error_message'}:
            if getattr(response, error_key, None):
                self.logger.debug('API response: %s', response)
                raise RuntimeError(f'API error: {getattr(response, error_key)}')

        if response.stop_reason == 'refusal':
            raise RuntimeError('Content filter triggered')

        return [AnthropicMessageOutputMessageRecord(response)]

    async def close(self):
        if self._client is not None:
            await self._client.close()
            self._client = None

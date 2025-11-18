from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING

from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam

from .anthropic import AnthropicClient, AnthropicOptions, BedrockOptions
from .dummy import DummyClient, DummyOptions
from .openai import OpenAIClient, OpenAIOptions
from ..convert import OpenAIChatCompletionInputMessageRecord

if TYPE_CHECKING:
    from ._base import BaseClient
    from ..session.tokens import TokenCounter

LOG = logging.getLogger(__name__)


async def create_client(provider: str,
                        settings: dict,
                        token_counter: Optional[TokenCounter] = None) -> BaseClient:
    if provider == 'openai':
        model_client_cls = OpenAIClient
        model_options_cls = OpenAIOptions
    elif provider == 'anthropic':
        model_client_cls = AnthropicClient
        model_options_cls = AnthropicOptions
    elif provider == 'bedrock':
        model_client_cls = AnthropicClient
        model_options_cls = BedrockOptions
    elif provider == 'dummy':
        model_client_cls = DummyClient
        model_options_cls = DummyOptions
    else:
        raise ValueError(f'unsupported model client: {provider}')

    options = model_options_cls.model_validate(settings)
    client = model_client_cls(options, token_counter=token_counter)
    model = await client.get_model_name()

    LOG.info('initialized %s client with model="%s" and options=%s',
             provider, model, options.model_dump(exclude_defaults=True))

    # run a basic sanity check with a simple request
    LOG.debug('running sanity check request for provider="%s", model="%s"', provider, model)
    await client.query([
        OpenAIChatCompletionInputMessageRecord([
            ChatCompletionSystemMessageParam(
                role='system',
                content='You are a helpful assistant.'
            )
        ]),
        OpenAIChatCompletionInputMessageRecord([
            ChatCompletionUserMessageParam(
                role='user',
                content='Say hi.'
            )
        ])
    ])

    return client

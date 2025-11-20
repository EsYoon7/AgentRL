from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Union, Literal, Any, Optional, Sequence

from anthropic.types import (Base64ImageSourceParam,
                             ImageBlockParam,
                             Message,
                             MessageParam,
                             TextBlockParam,
                             ToolResultBlockParam,
                             ToolUseBlockParam,
                             URLImageSourceParam)
from anthropic.types.beta import BetaMessage, BetaMessageParam, BetaTextBlockParam
from anthropic.types.tool_result_block_param import Content
from openai.types.chat import (ChatCompletionAssistantMessageParam,
                               ChatCompletionContentPartImageParam,
                               ChatCompletionContentPartParam,
                               ChatCompletionContentPartTextParam,
                               ChatCompletionMessage,
                               ChatCompletionMessageFunctionToolCallParam,
                               ChatCompletionMessageParam,
                               ChatCompletionSystemMessageParam,
                               ChatCompletionToolMessageParam,
                               ChatCompletionUserMessageParam)
from openai.types.chat.chat_completion_content_part_image_param import ImageURL
from openai.types.chat.chat_completion_message_function_tool_call_param import Function
from openai.types.responses import (EasyInputMessageParam,
                                    ResponseFunctionCallOutputItemListParam,
                                    ResponseFunctionToolCallParam,
                                    ResponseInputImageContentParam,
                                    ResponseInputImageParam,
                                    ResponseInputItemParam,
                                    ResponseInputMessageContentListParam,
                                    ResponseInputTextContentParam,
                                    ResponseInputTextParam,
                                    ResponseOutputItem)
from openai.types.responses.response_input_item_param import Message as ResponseMessage, FunctionCallOutput

from ._base import ConversionError, Convertible
from ..utils import model_dump


class MessageRecord(Convertible):

    def to_openai_chat_completion_input(self) -> list[ChatCompletionMessageParam]:
        return self.content

    def to_openai_response_input(self) -> list[ResponseInputItemParam]:
        return self.content

    def to_anthropic_message_input(self) -> Union[BetaMessageParam, MessageParam]:
        return self.content

    def to_anthropic_message_system(self) -> list[Union[BetaTextBlockParam, TextBlockParam]]:
        return self.content


class OpenAIChatCompletionInputMessageRecord(MessageRecord):
    content: list[ChatCompletionMessageParam]

    def to_openai_response_input(self) -> list[ResponseInputItemParam]:
        result: list[ResponseInputItemParam] = []
        for message in self.content:
            role = message['role']
            content = message['content']

            if role == 'tool':
                parts: ResponseFunctionCallOutputItemListParam = []
                if isinstance(content, str):
                    parts.append(ResponseInputTextContentParam(
                        type='input_text',
                        text=content
                    ))
                elif isinstance(content, Iterable):
                    for part in content:
                        if part.get('type') == 'text' and part.get('text'):
                            parts.append(ResponseInputTextContentParam(
                                type='input_text',
                                text=part['text']
                            ))
                        elif part.get('type') == 'image_url' and part.get('image_url', {}).get('url'):
                            parts.append(ResponseInputImageContentParam(
                                type='input_image',
                                image_url=part['image_url']['url'],
                                detail=part['image_url'].get('detail', 'auto')
                            ))
                result.append(FunctionCallOutput(
                    type='function_call_output',
                    call_id=message['tool_call_id'],
                    output=parts
                ))

            else:
                parts: ResponseInputMessageContentListParam = []
                if isinstance(content, str):
                    parts.append(ResponseInputTextParam(
                        type='input_text',
                        text=content
                    ))
                elif isinstance(content, Iterable):
                    for part in content:
                        if part.get('type') == 'text' and part.get('text'):
                            parts.append(ResponseInputTextParam(
                                type='input_text',
                                text=part['text']
                            ))
                        elif part.get('type') == 'image_url' and part.get('image_url', {}).get('url'):
                            parts.append(ResponseInputImageParam(
                                type='input_image',
                                image_url=part['image_url']['url'],
                                detail=part['image_url'].get('detail', 'auto')
                            ))
                if role == 'assistant':
                    for part in parts:
                        part['type'] = part['type'].replace('input_', 'output_')
                result.append(EasyInputMessageParam(
                    content=parts,
                    role=role,
                    type='message'
                ))

                if message.get('tool_calls'):
                    result.extend([
                        ResponseFunctionToolCallParam(
                            type='function_call',
                            call_id=tool_call.get('id'),
                            name=tool_call.get('function', {}).get('name', ''),
                            arguments=tool_call.get('function', {}).get('arguments') or '{}',
                        )
                        for tool_call in message['tool_calls']
                    ])

        return result

    def to_anthropic_message_input(self) -> Union[BetaMessageParam, MessageParam]:
        anthropic_role: Optional[Literal['assistant', 'user']] = None
        blocks = []

        for message in self.content:
            role = message['role']
            if role == 'system' or role == 'developer':
                raise ConversionError('cannot convert system or developer message to anthropic input message')
            new_anthropic_role: Literal['assistant', 'user'] = 'assistant' if role == 'assistant' else 'user'
            if anthropic_role is None:
                anthropic_role = new_anthropic_role
            elif anthropic_role != new_anthropic_role:
                raise ConversionError('cannot convert mixed roles to anthropic input message')

            content = message['content']
            if role == 'tool':
                parts: list[Content] = []
                if isinstance(content, str):
                    parts.append(TextBlockParam(
                        type='text',
                        text=content
                    ))
                elif isinstance(content, Iterable):
                    for part in content:
                        if part.get('type') == 'text' and part.get('text'):
                            parts.append(TextBlockParam(
                                type='text',
                                text=part['text']
                            ))
                        elif part.get('type') == 'image_url' and part.get('image_url', {}).get('url'):
                            if part['image_url']['url'].startswith('data:'):
                                media_type = part['image_url']['url'].split(';', maxsplit=1)[0][5:]
                                if media_type not in {'image/jpeg', 'image/png', 'image/gif', 'image/webp'}:
                                    raise ConversionError(f'unsupported media type for base64 image: {media_type}')
                                parts.append(ImageBlockParam(
                                    type='image',
                                    source=Base64ImageSourceParam(
                                        type='base64',
                                        media_type=media_type,
                                        data=part['image_url']['url'].split(',', maxsplit=1)[1]
                                    )
                                ))
                            else:
                                parts.append(ImageBlockParam(
                                    type='image',
                                    source=URLImageSourceParam(
                                        type='url',
                                        url=part['image_url']['url']
                                    )
                                ))
                blocks.append(ToolResultBlockParam(
                    type='tool_result',
                    tool_use_id=message['tool_call_id'],
                    content=parts
                ))
            else:
                if isinstance(content, str):
                    blocks.append(TextBlockParam(
                        type='text',
                        text=content
                    ))
                elif isinstance(content, Iterable):
                    for part in content:
                        if part.get('type') == 'text' and part.get('text'):
                            blocks.append(TextBlockParam(
                                type='text',
                                text=part['text']
                            ))
                        elif part.get('type') == 'image_url' and part.get('image_url', {}).get('url'):
                            if part['image_url']['url'].startswith('data:'):
                                media_type = part['image_url']['url'].split(';', maxsplit=1)[0][5:]
                                if media_type not in {'image/jpeg', 'image/png', 'image/gif', 'image/webp'}:
                                    raise ConversionError(f'unsupported media type for base64 image: {media_type}')
                                blocks.append(ImageBlockParam(
                                    type='image',
                                    source=Base64ImageSourceParam(
                                        type='base64',
                                        media_type=media_type,
                                        data=part['image_url']['url'].split(',', maxsplit=1)[1]
                                    )
                                ))
                            else:
                                blocks.append(ImageBlockParam(
                                    type='image',
                                    source=URLImageSourceParam(
                                        type='url',
                                        url=part['image_url']['url']
                                    )
                                ))
                if message.get('tool_calls'):
                    for tool_call in message['tool_calls']:
                        blocks.append(ToolUseBlockParam(
                            type='tool_use',
                            id=tool_call.get('id'),
                            name=tool_call.get('function', {}).get('name'),
                            input=json.loads(tool_call.get('function', {}).get('arguments') or '{}')
                        ))

        if anthropic_role is None or not blocks:
            raise ConversionError('cannot convert message to anthropic input message')

        return MessageParam(
            role=anthropic_role,
            content=blocks
        )

    def to_anthropic_message_system(self) -> list[Union[BetaTextBlockParam, TextBlockParam]]:
        result: list[Union[BetaTextBlockParam, TextBlockParam]] = []

        for message in self.content:
            if message['role'] == 'system' or message['role'] == 'developer':
                content = message['content']
                if isinstance(content, str):
                    result.append(TextBlockParam(
                        type='text',
                        text=content
                    ))
                elif isinstance(content, Iterable):
                    for part in content:
                        if isinstance(part, dict) and part.get('text'):
                            result.append(TextBlockParam(
                                type='text',
                                text=part['text']
                            ))

        if result:
            return result
        raise ConversionError('cannot convert to anthropic system message: no system prompt found')


class OpenAIChatCompletionOutputMessageRecord(MessageRecord):
    content: Sequence[ChatCompletionMessage]

    def to_openai_chat_completion_input(self) -> list[ChatCompletionMessageParam]:
        return [model_dump(i) for i in self.content]

    def to_openai_response_input(self) -> list[ResponseInputItemParam]:
        return (OpenAIChatCompletionInputMessageRecord(self.to_openai_chat_completion_input())
                .to_openai_response_input())

    def to_anthropic_message_input(self) -> Union[BetaMessageParam, MessageParam]:
        return (OpenAIChatCompletionInputMessageRecord(self.to_openai_chat_completion_input())
                .to_anthropic_message_input())

    def to_anthropic_message_system(self) -> list[Union[BetaTextBlockParam, TextBlockParam]]:
        return (OpenAIChatCompletionInputMessageRecord(self.to_openai_chat_completion_input())
                .to_anthropic_message_system())


class OpenAIResponseInputMessageRecord(MessageRecord):
    content: list[ResponseInputItemParam]

    def to_openai_chat_completion_input(self) -> list[ChatCompletionMessageParam]:
        result: list[ChatCompletionMessageParam] = []
        pending: dict[str, Any] = {}

        for message in self.content:
            if message['type'] == 'message' or message['type'] == 'function_call_output':
                content = message['content'] if message['type'] == 'message' else message['output']
                parts: Union[str, list[ChatCompletionContentPartParam]] = []
                if isinstance(content, str):
                    parts = content
                elif isinstance(content, Iterable):
                    for part in content:
                        if part.get('type') == 'input_text' or part.get('type') == 'output_text' and part.get('text'):
                            parts.append(ChatCompletionContentPartTextParam(
                                type='text',
                                text=part['text']
                            ))
                        elif part.get('type') == 'input_image' and part.get('image_url'):
                            parts.append(ChatCompletionContentPartImageParam(
                                type='image_url',
                                image_url=ImageURL(
                                    url=part['image_url'],
                                    detail=part.get('detail', 'auto')
                                )
                            ))

                # flatten if only one text part is present, for better compatibility
                if isinstance(parts, list) and len(parts) == 1 and parts[0]['type'] == 'text':
                    parts = parts[0]['text']

                if message['type'] == 'message':
                    # noinspection PyTypeChecker
                    result.append({
                        'role': message['role'],
                        'content': parts,
                        **pending
                    })
                    pending = {}
                else:
                    result.append(ChatCompletionToolMessageParam(
                        role='tool',
                        tool_call_id=message['call_id'],
                        content=parts
                    ))

            elif message['type'] == 'function_call':
                tool_calls = pending.get('tool_calls', [])
                tool_calls.append(ChatCompletionMessageFunctionToolCallParam(
                    id=message['call_id'],
                    type='function',
                    function=Function(
                        name=message['name'],
                        arguments=message['arguments']
                    )
                ))
                pending['tool_calls'] = tool_calls

        if pending:
            if len(result) > 0 and result[-1]['role'] != 'tool':
                result[-1].update(pending)
            else:
                result.append(ChatCompletionAssistantMessageParam(
                    role='assistant',
                    content='',
                    **pending
                ))

        return result

    def to_anthropic_message_input(self) -> Union[BetaMessageParam, MessageParam]:
        raise NotImplementedError  # not needed

    def to_anthropic_message_system(self) -> list[Union[BetaTextBlockParam, TextBlockParam]]:
        raise NotImplementedError  # not needed


class OpenAIResponseOutputMessageRecord(MessageRecord):
    content: Sequence[ResponseOutputItem]

    def to_openai_chat_completion_input(self) -> list[ChatCompletionMessageParam]:
        return (OpenAIResponseInputMessageRecord(self.to_openai_response_input())
                .to_openai_chat_completion_input())

    def to_openai_response_input(self) -> list[ResponseInputItemParam]:
        return [model_dump(i) for i in self.content]

    def to_anthropic_message_input(self) -> Union[BetaMessageParam, MessageParam]:
        return (OpenAIResponseInputMessageRecord(self.to_openai_response_input())
                .to_anthropic_message_input())

    def to_anthropic_message_system(self) -> list[Union[BetaTextBlockParam, TextBlockParam]]:
        return (OpenAIResponseInputMessageRecord(self.to_openai_response_input())
                .to_anthropic_message_system())


class AnthropicMessageInputMessageRecord(MessageRecord):
    content: Union[BetaMessageParam, MessageParam]

    def to_openai_chat_completion_input(self) -> list[ChatCompletionMessageParam]:
        result: list[ChatCompletionMessageParam] = []
        parts: Union[str, list[ChatCompletionContentPartParam]] = []
        tool_calls: list[ChatCompletionMessageFunctionToolCallParam] = []

        for block in self.content['content']:
            if block['type'] == 'tool_result':
                content_parts: Union[str, list[ChatCompletionContentPartParam]] = []
                for content_block in block['content']:
                    if content_block['type'] == 'text':
                        content_parts.append(ChatCompletionContentPartTextParam(
                            type='text',
                            text=content_block['text']
                        ))
                    elif content_block['type'] == 'image':
                        if content_block['source']['type'] == 'base64':
                            media_type = content_block['source']['media_type']
                            data = content_block['source']['data']
                            content_parts.append(ChatCompletionContentPartImageParam(
                                type='image_url',
                                image_url=ImageURL(
                                    url=f'data:{media_type};base64,{data}',
                                    detail='auto'
                                )
                            ))
                        elif content_block['source']['type'] == 'url':
                            content_parts.append(ChatCompletionContentPartImageParam(
                                type='image_url',
                                image_url=ImageURL(
                                    url=content_block['source']['url'],
                                    detail='auto'
                                )
                            ))

                # flatten if only one text part is present, for better compatibility
                if isinstance(content_parts, list) and len(content_parts) == 1 and content_parts[0]['type'] == 'text':
                    content_parts = content_parts[0]['text']

                result.append(ChatCompletionToolMessageParam(
                    role='tool',
                    tool_call_id=block['tool_use_id'],
                    content=content_parts
                ))
                continue

            if block['type'] == 'text':
                parts.append(ChatCompletionContentPartTextParam(
                    type='text',
                    text=block['text']
                ))
            elif block['type'] == 'image':
                if block['source']['type'] == 'base64':
                    media_type = block['source']['media_type']
                    data = block['source']['data']
                    parts.append(ChatCompletionContentPartImageParam(
                        type='image_url',
                        image_url=ImageURL(
                            url=f'data:{media_type};base64,{data}',
                            detail='auto'
                        )
                    ))
                elif block['source']['type'] == 'url':
                    parts.append(ChatCompletionContentPartImageParam(
                        type='image_url',
                        image_url=ImageURL(
                            url=block['source']['url'],
                            detail='auto'
                        )
                    ))
            elif self.content['role'] == 'assistant' and block['type'] == 'tool_use':
                tool_calls.append(ChatCompletionMessageFunctionToolCallParam(
                    id=block['id'],
                    type='function',
                    function=Function(
                        name=block['name'],
                        arguments=json.dumps(block['input'])
                    )
                ))

        # flatten if only one text part is present, for better compatibility
        if isinstance(parts, list) and len(parts) == 1 and parts[0]['type'] == 'text':
            parts = parts[0]['text']

        if self.content['role'] == 'assistant':
            result.append(ChatCompletionAssistantMessageParam(
                role='assistant',
                content=parts if parts or not tool_calls else None,
                tool_calls=tool_calls
            ))
        else:
            result.append(ChatCompletionUserMessageParam(
                role='user',
                content=parts
            ))

        return result

    def to_openai_response_input(self) -> list[ResponseInputItemParam]:
        raise NotImplementedError  # not needed

    def to_anthropic_message_system(self) -> list[Union[BetaTextBlockParam, TextBlockParam]]:
        raise ConversionError('cannot convert anthropic input message to system message')


class AnthropicMessageOutputMessageRecord(MessageRecord):
    content: Union[BetaMessage, Message]

    def to_openai_chat_completion_input(self) -> list[ChatCompletionMessageParam]:
        return (AnthropicMessageInputMessageRecord(self.to_anthropic_message_input())
                .to_openai_chat_completion_input())

    def to_openai_response_input(self) -> list[ResponseInputItemParam]:
        return (AnthropicMessageInputMessageRecord(self.to_anthropic_message_input())
                .to_openai_response_input())

    def to_anthropic_message_input(self) -> Union[BetaMessageParam, MessageParam]:
        if isinstance(self.content, BetaMessage):
            return BetaMessageParam(
                content=[model_dump(block) for block in self.content.content],
                role=self.content.role
            )
        return MessageParam(
            content=[model_dump(block) for block in self.content.content],
            role=self.content.role
        )

    def to_anthropic_message_system(self) -> list[Union[BetaTextBlockParam, TextBlockParam]]:
        return (AnthropicMessageInputMessageRecord(self.to_anthropic_message_input())
                .to_anthropic_message_system())


class AnthropicMessageSystemMessageRecord(MessageRecord):
    content: list[Union[BetaTextBlockParam, TextBlockParam]]

    def to_openai_chat_completion_input(self) -> list[ChatCompletionMessageParam]:
        return [ChatCompletionSystemMessageParam(
            role='system',
            content=[
                ChatCompletionContentPartTextParam(
                    type='text',
                    text=block['text']
                )
                for block in self.content
            ]
        )]

    def to_openai_response_input(self) -> list[ResponseInputItemParam]:
        return [ResponseMessage(
            type='message',
            role='system',
            content=[
                ResponseInputTextParam(
                    type='input_text',
                    text=block['text']
                )
                for block in self.content
            ]
        )]

    def to_anthropic_message_input(self) -> Union[BetaMessageParam, MessageParam]:
        raise ConversionError('cannot convert anthropic system message to input message')

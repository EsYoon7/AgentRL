from __future__ import annotations

import json

import pytest
from anthropic.types import (Base64ImageSourceParam,
                             ImageBlockParam,
                             Message,
                             MessageParam,
                             TextBlock,
                             TextBlockParam,
                             ThinkingBlockParam,
                             ToolResultBlockParam,
                             ToolUseBlockParam,
                             URLImageSourceParam,
                             Usage)
from anthropic.types.beta import (BetaImageBlockParam,
                                  BetaMessage,
                                  BetaMessageParam,
                                  BetaRedactedThinkingBlockParam,
                                  BetaTextBlock,
                                  BetaTextBlockParam,
                                  BetaThinkingBlockParam,
                                  BetaToolResultBlockParam,
                                  BetaToolUseBlockParam,
                                  BetaUsage)
from openai.types.chat import (ChatCompletionMessage,
                               ChatCompletionMessageFunctionToolCall)
from openai.types.chat.chat_completion_message_function_tool_call import Function
from openai.types.responses import (EasyInputMessageParam,
                                    ResponseFunctionToolCallParam,
                                    ResponseInputImageContentParam,
                                    ResponseInputImageParam,
                                    ResponseInputTextContentParam,
                                    ResponseInputTextParam,
                                    ResponseOutputMessageParam,
                                    ResponseOutputTextParam,
                                    ResponseReasoningItemParam)
from openai.types.responses.response_input_item_param import FunctionCallOutput
from openai.types.responses.response_reasoning_item_param import Content, Summary

from agentrl.eval.convert import (AnthropicMessageInputMessageRecord,
                                  AnthropicMessageOutputMessageRecord,
                                  AnthropicMessageSystemMessageRecord,
                                  ConversionError,
                                  MessageRecord,
                                  OpenAIChatCompletionInputMessageRecord,
                                  OpenAIChatCompletionOutputMessageRecord,
                                  OpenAIResponseInputMessageRecord,
                                  OpenAIResponseOutputMessageRecord)
from agentrl.eval.utils import model_dump


def _dump_sequence(sequence):
    return [model_dump(item) for item in sequence]


@pytest.mark.parametrize(
    ('content', 'expected'),
    [
        ('hello agent', [{'type': 'input_text', 'text': 'hello agent'}]),
        (
            [{'type': 'text', 'text': 'part 1'}, {'type': 'text', 'text': 'part 2'}],
            [
                {'type': 'input_text', 'text': 'part 1'},
                {'type': 'input_text', 'text': 'part 2'},
            ],
        ),
        (
            [{'type': 'image_url', 'image_url': {'url': 'https://example.org/img.png', 'detail': 'high'}}],
            [{'type': 'input_image', 'image_url': 'https://example.org/img.png', 'detail': 'high'}],
        ),
    ],
    ids=['string-content', 'multi-text', 'image'],
)
def test_chat_input_to_response_user_variants(content, expected):
    record = OpenAIChatCompletionInputMessageRecord([{'role': 'user', 'content': content}])

    response_items = _dump_sequence(record.to_openai_response_input())

    assert response_items == [
        {
            'type': 'message',
            'role': 'user',
            'content': expected,
        }
    ]


@pytest.mark.parametrize(
    ('content', 'expected_output'),
    [
        ('raw tool payload', [{'type': 'input_text', 'text': 'raw tool payload'}]),
        (
            [
                {'type': 'text', 'text': 'structured'},
                {'type': 'image_url', 'image_url': {'url': 'https://example.org/tool.png', 'detail': 'low'}},
            ],
            [
                {'type': 'input_text', 'text': 'structured'},
                {'type': 'input_image', 'image_url': 'https://example.org/tool.png', 'detail': 'low'},
            ],
        ),
    ],
    ids=['string', 'multi-part'],
)
def test_chat_input_to_response_tool_variants(content, expected_output):
    record = OpenAIChatCompletionInputMessageRecord([
        {
            'role': 'tool',
            'tool_call_id': 'tool-call',
            'content': content,
        }
    ])

    response_items = _dump_sequence(record.to_openai_response_input())

    assert response_items == [
        {
            'type': 'function_call_output',
            'call_id': 'tool-call',
            'output': expected_output,
        }
    ]


def test_chat_input_to_response_assistant_with_tool_calls():
    record = OpenAIChatCompletionInputMessageRecord([
        {
            'role': 'assistant',
            'content': [
                {'type': 'text', 'text': 'Computation ready'},
                {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,QUJD'}},
            ],
            'tool_calls': [
                {
                    'id': 'call-1',
                    'type': 'function',
                    'function': {'name': 'compute', 'arguments': '{"x": 1}'},
                },
                {
                    'id': 'call-2',
                    'type': 'function',
                    'function': {'name': 'explain', 'arguments': ''},
                },
            ],
        }
    ])

    response_items = _dump_sequence(record.to_openai_response_input())

    assert len(response_items) == 3
    assistant_message, first_call, second_call = response_items
    assert assistant_message['role'] == 'assistant'
    assert assistant_message['content'][0] == {'type': 'output_text', 'text': 'Computation ready'}
    assert assistant_message['content'][1]['type'] == 'output_image'
    assert first_call == {
        'type': 'function_call',
        'call_id': 'call-1',
        'name': 'compute',
        'arguments': '{"x": 1}',
    }
    assert second_call == {
        'type': 'function_call',
        'call_id': 'call-2',
        'name': 'explain',
        'arguments': '{}',
    }


def test_chat_input_to_anthropic_assistant_blocks():
    record = OpenAIChatCompletionInputMessageRecord([
        {
            'role': 'assistant',
            'content': [
                {'type': 'text', 'text': 'Observation'},
                {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,UVdF'}},
            ],
            'tool_calls': [
                {
                    'id': 'call-1',
                    'type': 'function',
                    'function': {'name': 'solve', 'arguments': '{"answer": 42}'},
                }
            ],
        }
    ])

    anthropic = model_dump(record.to_anthropic_message_input())

    assert anthropic['role'] == 'assistant'
    content = anthropic['content']
    assert content[0] == {'type': 'text', 'text': 'Observation'}
    assert content[1]['type'] == 'image'
    assert content[1]['source'] == {
        'type': 'base64',
        'media_type': 'image/png',
        'data': 'UVdF',
    }
    tool_use = content[2]
    assert tool_use['type'] == 'tool_use'
    assert tool_use['id'] == 'call-1'
    assert tool_use['name'] == 'solve'
    assert tool_use['input'] == {'answer': 42}


def test_chat_input_to_anthropic_tool_result():
    record = OpenAIChatCompletionInputMessageRecord([
        {
            'role': 'tool',
            'tool_call_id': 'call-1',
            'content': [
                {'type': 'text', 'text': 'result text'},
                {'type': 'image_url', 'image_url': {'url': 'https://example.org/result.png'}},
            ],
        }
    ])

    anthropic = model_dump(record.to_anthropic_message_input())

    assert anthropic['role'] == 'user'
    assert len(anthropic['content']) == 1
    tool_result = anthropic['content'][0]
    assert tool_result['type'] == 'tool_result'
    assert tool_result['tool_use_id'] == 'call-1'
    result_content = tool_result['content']
    assert result_content[0] == {'type': 'text', 'text': 'result text'}
    assert result_content[1]['type'] == 'image'
    assert result_content[1]['source'] == {'type': 'url', 'url': 'https://example.org/result.png'}


def test_chat_input_to_anthropic_system_messages():
    record = OpenAIChatCompletionInputMessageRecord([
        {'role': 'system', 'content': 'primary system'},
        {'role': 'developer', 'content': [{'type': 'text', 'text': 'secondary system'}]},
    ])

    blocks = _dump_sequence(record.to_anthropic_message_system())

    assert blocks == [
        {'type': 'text', 'text': 'primary system'},
        {'type': 'text', 'text': 'secondary system'},
    ]


def test_chat_input_to_anthropic_invalid_media_type():
    record = OpenAIChatCompletionInputMessageRecord([
        {
            'role': 'assistant',
            'content': [{'type': 'image_url', 'image_url': {'url': 'data:image/tiff;base64,AAAA'}}],
        }
    ])

    with pytest.raises(ConversionError):
        record.to_anthropic_message_input()


def test_chat_input_to_anthropic_disallows_mixed_roles():
    record = OpenAIChatCompletionInputMessageRecord([
        {'role': 'user', 'content': 'hello'},
        {'role': 'assistant', 'content': 'world'},
    ])

    with pytest.raises(ConversionError):
        record.to_anthropic_message_input()


def test_chat_output_record_roundtrip():
    tool_call = ChatCompletionMessageFunctionToolCall(
        id='call-7',
        type='function',
        function=Function(name='lookup', arguments='{"term": "pytest"}'),
    )
    message = ChatCompletionMessage(
        role='assistant',
        content='Answer goes here',
        tool_calls=[tool_call],
    )
    record = OpenAIChatCompletionOutputMessageRecord([message])

    chat_completion_input = _dump_sequence(record.to_openai_chat_completion_input())
    assert chat_completion_input[0]['role'] == 'assistant'
    assert chat_completion_input[0]['tool_calls'][0]['id'] == 'call-7'

    response_input = _dump_sequence(record.to_openai_response_input())
    assert response_input[0]['role'] == 'assistant'
    assert response_input[1]['type'] == 'function_call'
    assert response_input[1]['call_id'] == 'call-7'

    anthropic = model_dump(record.to_anthropic_message_input())
    assert anthropic['role'] == 'assistant'
    assert anthropic['content'][0]['text'] == 'Answer goes here'

    with pytest.raises(ConversionError):
        record.to_anthropic_message_system()


def test_response_input_to_chat_completion_user_string():
    record = OpenAIResponseInputMessageRecord([
        EasyInputMessageParam(
            type='message',
            role='user',
            content='Plain user string',
        )
    ])

    messages = record.to_openai_chat_completion_input()
    dumped = _dump_sequence(messages)

    assert dumped == [{'role': 'user', 'content': 'Plain user string'}]


def test_response_input_to_chat_completion_user_parts():
    record = OpenAIResponseInputMessageRecord([
        EasyInputMessageParam(
            type='message',
            role='assistant',
            content=[
                ResponseInputTextParam(type='input_text', text='part 1'),
                ResponseInputImageParam(type='input_image', image_url='https://example.org/out.png', detail='high'),
            ],
        ),
        ResponseFunctionToolCallParam(
            type='function_call',
            call_id='call-abc',
            name='summarize',
            arguments='{"q": "info"}',
        ),
    ])

    messages = record.to_openai_chat_completion_input()
    dumped = _dump_sequence(messages)

    assert len(dumped) == 1
    assistant = dumped[0]
    assert assistant['role'] == 'assistant'
    assert assistant['content'][0] == {'type': 'text', 'text': 'part 1'}
    assert assistant['content'][1]['type'] == 'image_url'
    assert json.loads(assistant['tool_calls'][0]['function']['arguments']) == {'q': 'info'}


def test_response_input_to_chat_completion_tool_output():
    record = OpenAIResponseInputMessageRecord([
        FunctionCallOutput(
            type='function_call_output',
            call_id='tool-1',
            output=[
                ResponseInputTextContentParam(type='input_text', text='output'),
                ResponseInputImageContentParam(
                    type='input_image',
                    image_url='https://example.org/tool.png',
                    detail='auto',
                ),
            ],
        )
    ])

    messages = record.to_openai_chat_completion_input()
    dumped = _dump_sequence(messages)

    assert len(dumped) == 1
    tool_message = dumped[0]
    assert tool_message['role'] == 'tool'
    assert tool_message['tool_call_id'] == 'tool-1'
    assert tool_message['content'][0] == {'type': 'text', 'text': 'output'}
    assert tool_message['content'][1]['image_url']['url'] == 'https://example.org/tool.png'


def test_response_input_to_chat_completion_inserts_fallback_for_pending_tool_calls():
    record = OpenAIResponseInputMessageRecord([
        ResponseFunctionToolCallParam(
            type='function_call',
            call_id='pending-1',
            name='do_work',
            arguments='{}',
        )
    ])

    messages = record.to_openai_chat_completion_input()
    dumped = _dump_sequence(messages)

    assert dumped == [
        {
            'role': 'assistant',
            'content': '',
            'tool_calls': [
                {
                    'id': 'pending-1',
                    'type': 'function',
                    'function': {'name': 'do_work', 'arguments': '{}'},
                }
            ],
        }
    ]


def test_response_input_to_chat_completion_appends_tool_calls_after_tool_output():
    record = OpenAIResponseInputMessageRecord([
        FunctionCallOutput(
            type='function_call_output',
            call_id='tool-1',
            output=[ResponseInputTextContentParam(type='input_text', text='done')],
        ),
        ResponseFunctionToolCallParam(
            type='function_call',
            call_id='tool-1',
            name='finalize',
            arguments='{}',
        ),
    ])

    messages = record.to_openai_chat_completion_input()
    dumped = _dump_sequence(messages)

    assert len(dumped) == 2
    tool_message, assistant = dumped
    assert tool_message['role'] == 'tool'
    assert assistant['role'] == 'assistant'
    assert assistant['tool_calls'][0]['function']['name'] == 'finalize'


def test_response_output_message_record_routes_via_input_conversion():
    record = OpenAIResponseOutputMessageRecord([
        ResponseOutputMessageParam(
            id='out-1',
            role='assistant',
            type='message',
            status='completed',
            content=[
                ResponseOutputTextParam(
                    type='output_text',
                    text='Final output',
                    annotations=[],
                    logprobs=[],
                )
            ],
        )
    ])

    chat_messages = record.to_openai_chat_completion_input()
    dumped_chat = _dump_sequence(chat_messages)
    assert dumped_chat[0]['role'] == 'assistant'

    response_items = _dump_sequence(record.to_openai_response_input())
    assert response_items[0]['role'] == 'assistant'

    with pytest.raises(NotImplementedError):
        record.to_anthropic_message_input()
    with pytest.raises(NotImplementedError):
        record.to_anthropic_message_system()


def test_anthropic_input_to_chat_completion_maps_all_blocks():
    message = MessageParam(
        role='assistant',
        content=[
            TextBlockParam(type='text', text='Response text'),
            ThinkingBlockParam(type='thinking', thinking='Step-by-step plan', signature='sig-1'),
            ImageBlockParam(
                type='image',
                source=Base64ImageSourceParam(type='base64', media_type='image/png', data='QUJD'),
            ),
            ToolUseBlockParam(type='tool_use', id='call-1', name='compute', input={'value': 9}),
            ToolResultBlockParam(
                type='tool_result',
                tool_use_id='call-1',
                content=[
                    TextBlockParam(type='text', text='Tool result'),
                    ImageBlockParam(
                        type='image',
                        source=URLImageSourceParam(type='url', url='https://example.org/tool.png'),
                    ),
                ],
            ),
        ],
    )

    record = AnthropicMessageInputMessageRecord(message)
    messages = record.to_openai_chat_completion_input()
    dumped = _dump_sequence(messages)

    assert len(dumped) == 2
    tool_message = dumped[0]
    assert tool_message['role'] == 'tool'
    assert tool_message['tool_call_id'] == 'call-1'
    assert tool_message['content'][0] == {'type': 'text', 'text': 'Tool result'}
    assistant_message = dumped[1]
    assert assistant_message['role'] == 'assistant'
    text_parts = [part['text'] for part in assistant_message['content'] if part['type'] == 'text']
    assert any('Response text' in part for part in text_parts)
    images = [part for part in assistant_message['content'] if part['type'] == 'image_url']
    assert images and images[0]['image_url']['url'].startswith('data:image/png;base64,')
    tool_calls = assistant_message['tool_calls']
    assert tool_calls[0]['id'] == 'call-1'
    assert json.loads(tool_calls[0]['function']['arguments']) == {'value': 9}


def test_anthropic_input_tool_use_without_text_preserves_empty_content():
    message = MessageParam(
        role='assistant',
        content=[
            ToolUseBlockParam(type='tool_use', id='call-1', name='execute', input={'a': 1}),
        ],
    )

    record = AnthropicMessageInputMessageRecord(message)
    messages = record.to_openai_chat_completion_input()
    dumped = _dump_sequence(messages)

    assert dumped == [
        {
            'role': 'assistant',
            'content': None,
            'tool_calls': [
                {
                    'id': 'call-1',
                    'type': 'function',
                    'function': {'name': 'execute', 'arguments': json.dumps({'a': 1})},
                }
            ],
        }
    ]


def test_anthropic_input_user_message_with_images():
    message = MessageParam(
        role='user',
        content=[
            TextBlockParam(type='text', text='User text'),
            ImageBlockParam(
                type='image',
                source=URLImageSourceParam(type='url', url='https://example.org/image.png'),
            ),
        ],
    )

    record = AnthropicMessageInputMessageRecord(message)
    messages = record.to_openai_chat_completion_input()
    dumped = _dump_sequence(messages)

    assert dumped == [
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': 'User text'},
                {'type': 'image_url', 'image_url': {'url': 'https://example.org/image.png', 'detail': 'auto'}},
            ],
        }
    ]


def test_anthropic_input_system_conversion_not_supported():
    message = MessageParam(role='assistant', content=[TextBlockParam(type='text', text='Hello')])
    record = AnthropicMessageInputMessageRecord(message)

    with pytest.raises(ConversionError):
        record.to_anthropic_message_system()


def test_anthropic_output_message_record_handles_standard_and_beta():
    standard_record = AnthropicMessageOutputMessageRecord(Message(
        id='msg-1',
        role='assistant',
        type='message',
        model='claude-3-opus-20240229',
        usage=Usage(input_tokens=1, output_tokens=1),
        content=[TextBlock(type='text', text='Standard response')],
    ))
    beta_record = AnthropicMessageOutputMessageRecord(BetaMessage(
        id='beta-msg',
        role='assistant',
        type='message',
        model='claude-3-7-sonnet-latest',
        usage=BetaUsage(input_tokens=2, output_tokens=3),
        content=[BetaTextBlock(type='text', text='Beta response')],
    ))

    standard_param = model_dump(standard_record.to_anthropic_message_input())
    beta_param = model_dump(beta_record.to_anthropic_message_input())

    assert standard_param['content'][0]['text'] == 'Standard response'
    assert beta_param['content'][0]['text'] == 'Beta response'


def test_anthropic_system_message_record_conversions():
    record = AnthropicMessageSystemMessageRecord([
        TextBlockParam(type='text', text='Primary guidance'),
        BetaTextBlockParam(type='text', text='Beta guidance'),
    ])

    chat_messages = _dump_sequence(record.to_openai_chat_completion_input())
    assert chat_messages[0]['role'] == 'system'
    assert [part['text'] for part in chat_messages[0]['content']] == ['Primary guidance', 'Beta guidance']

    response_items = _dump_sequence(record.to_openai_response_input())
    assert response_items[0]['role'] == 'system'
    assert [item['text'] for item in response_items[0]['content']] == ['Primary guidance', 'Beta guidance']

    with pytest.raises(ConversionError):
        record.to_anthropic_message_input()


def test_message_record_dump_all_and_convert_all_roundtrip():
    message = ChatCompletionMessage(
        role='assistant',
        content='Hello world',
    )
    record = OpenAIChatCompletionOutputMessageRecord([message])

    dumped = MessageRecord.dump_all([record])
    assert dumped[0]['role'] == 'assistant'

    converted = MessageRecord.convert_all([record], to='openai-response-input')
    converted_dump = _dump_sequence(converted)
    assert converted_dump[0]['role'] == 'assistant'

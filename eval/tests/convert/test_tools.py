from agentrl.eval.convert.tools import (AnthropicMessageFunctionDefinition,
                                            FunctionDefinition,
                                            OpenAIChatCompletionFunctionDefinition,
                                            OpenAIResponseFunctionDefinition)


def test_tool_definition_conversions():
    openai_tool = OpenAIChatCompletionFunctionDefinition({
        'type': 'function',
        'function': {
            'name': 'search',
            'description': 'Search web',
            'parameters': {'type': 'object'},
        },
    })
    response_tool = openai_tool.to_openai_response()
    anthropic_tool = openai_tool.to_anthropic_message()

    assert response_tool['name'] == 'search'
    assert anthropic_tool['name'] == 'search'
    assert anthropic_tool['input_schema'] == {'type': 'object'}

    response_definition = OpenAIResponseFunctionDefinition(response_tool)
    chat_tool = response_definition.to_openai_chat_completion()
    anthropic_from_response = response_definition.to_anthropic_message()

    assert chat_tool['function']['name'] == 'search'
    assert anthropic_from_response['input_schema'] == {'type': 'object'}

    anthropic_definition = AnthropicMessageFunctionDefinition(anthropic_tool)
    chat_from_anthropic = anthropic_definition.to_openai_chat_completion()
    response_from_anthropic = anthropic_definition.to_openai_response()

    assert chat_from_anthropic['function']['name'] == 'search'
    assert response_from_anthropic['parameters'] == {'type': 'object'}

    base_definition = FunctionDefinition(anthropic_tool)
    assert base_definition.to_anthropic_message() == anthropic_tool

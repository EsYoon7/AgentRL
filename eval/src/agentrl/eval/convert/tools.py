from __future__ import annotations

from anthropic.types import ToolParam
from openai.types.chat import ChatCompletionFunctionToolParam
from openai.types.responses import FunctionToolParam
from openai.types.shared_params import FunctionDefinition as OpenAIFunctionDefinition

from ._base import Convertible


class FunctionDefinition(Convertible):

    def to_openai_chat_completion(self) -> ChatCompletionFunctionToolParam:
        return self.content

    def to_openai_response(self) -> FunctionToolParam:
        return self.content

    def to_anthropic_message(self) -> ToolParam:
        return self.content


class OpenAIChatCompletionFunctionDefinition(FunctionDefinition):
    content: ChatCompletionFunctionToolParam

    def to_openai_response(self) -> FunctionToolParam:
        return FunctionToolParam(
            name=self.content['function']['name'],
            parameters=self.content['function']['parameters'],
            strict=False,
            type='function',
            description=self.content['function'].get('description', '')
        )

    def to_anthropic_message(self) -> ToolParam:
        return ToolParam(
            input_schema=self.content['function']['parameters'],
            name=self.content['function']['name'],
            description=self.content['function'].get('description', ''),
            type='custom'
        )


class OpenAIResponseFunctionDefinition(FunctionDefinition):
    content: FunctionToolParam

    def to_openai_chat_completion(self) -> ChatCompletionFunctionToolParam:
        return ChatCompletionFunctionToolParam(
            type='function',
            function=OpenAIFunctionDefinition(
                name=self.content['name'],
                description=self.content.get('description', ''),
                parameters=self.content['parameters'],
                strict=False
            )
        )

    def to_anthropic_message(self) -> ToolParam:
        return ToolParam(
            input_schema=self.content['parameters'],
            name=self.content['name'],
            description=self.content.get('description', ''),
            type='custom'
        )


class AnthropicMessageFunctionDefinition(FunctionDefinition):
    content: ToolParam

    def to_openai_chat_completion(self) -> ChatCompletionFunctionToolParam:
        return ChatCompletionFunctionToolParam(
            type='function',
            function=OpenAIFunctionDefinition(
                name=self.content['name'],
                description=self.content.get('description', ''),
                parameters=self.content['input_schema'],
                strict=False
            )
        )

    def to_openai_response(self) -> FunctionToolParam:
        return FunctionToolParam(
            name=self.content['name'],
            parameters=self.content['input_schema'],
            strict=False,
            type='function',
            description=self.content.get('description', '')
        )

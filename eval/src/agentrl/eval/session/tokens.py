from __future__ import annotations

from typing import Optional, Union, TYPE_CHECKING

from anthropic.types import Usage
from anthropic.types.beta import BetaUsage
from openai.types import CompletionUsage
from openai.types.responses import ResponseUsage

from ..event.types import TokenUsageSummaryEvent
from ..utils import format_number

if TYPE_CHECKING:
    from ..event.bus import EventBus


class TokenCounter:

    def __init__(self, event_bus: Optional[EventBus] = None):
        self.event_bus = event_bus

        self.input = 0
        self.input_cached = 0
        self.input_image = 0
        self.output = 0
        self.output_thinking = 0

    def add_from_usage(self, usage: Union[CompletionUsage, ResponseUsage, Usage, BetaUsage]):
        if hasattr(usage, 'input_tokens') and usage.input_tokens:
            self.input += usage.input_tokens
        elif hasattr(usage, 'prompt_tokens') and usage.prompt_tokens:
            self.input += usage.prompt_tokens

        if hasattr(usage, 'prompt_tokens_details') and usage.prompt_tokens_details:
            if hasattr(usage.prompt_tokens_details, 'cached_tokens') and usage.prompt_tokens_details.cached_tokens:
                self.input_cached += usage.prompt_tokens_details.cached_tokens
            if hasattr(usage.prompt_tokens_details, 'image_tokens') and usage.prompt_tokens_details.image_tokens:
                self.input_image += usage.prompt_tokens_details.image_tokens
        elif hasattr(usage, 'input_token_details') and usage.input_tokens_details:
            if hasattr(usage.input_tokens_details, 'cached_tokens') and usage.input_tokens_details.cached_tokens:
                self.input_cached += usage.input_tokens_details.cached_tokens
            if hasattr(usage.input_tokens_details, 'image_tokens') and usage.input_tokens_details.image_tokens:
                self.input_image += usage.input_tokens_details.image_tokens
        elif hasattr(usage, 'cache_read_input_tokens') and usage.cache_read_input_tokens:
            self.input_cached += usage.cache_read_input_tokens

        if hasattr(usage, 'output_tokens') and usage.output_tokens:
            self.output += usage.output_tokens
        elif hasattr(usage, 'completion_tokens') and usage.completion_tokens:
            self.output += usage.completion_tokens

        if hasattr(usage, 'completion_tokens_details') and usage.completion_tokens_details:
            if hasattr(usage.completion_tokens_details, 'reasoning_tokens') and usage.completion_tokens_details.reasoning_tokens:
                self.output_thinking += usage.completion_tokens_details.reasoning_tokens
        elif hasattr(usage, 'output_tokens_details') and usage.output_tokens_details:
            if hasattr(usage.output_tokens_details, 'reasoning_tokens') and usage.output_tokens_details.reasoning_tokens:
                self.output_thinking += usage.output_tokens_details.reasoning_tokens

        if self.event_bus is not None:
            self.event_bus.publish_sync(TokenUsageSummaryEvent(summary=self.summary()))

    def total(self) -> dict[str, int]:
        return {
            'input_total': self.input,
            'input_cached': self.input_cached,
            'input_image': self.input_image,
            'output_total': self.output,
            'output_thinking': self.output_thinking,
            'total': self.input + self.output
        }

    def summary(self) -> str:
        total = self.total()
        result = f'{format_number(total["total"])} token'
        if total['total'] != 1:
            result += 's'
        result += f' used ({format_number(total["input_total"])} input'
        if total['input_cached'] > 0:
            result += f', {format_number(total["input_cached"])} cached'
        if total['input_image'] > 0:
            result += f', {format_number(total["input_image"])} image'
        result += f'; {format_number(total["output_total"])} output'
        if total['output_thinking'] > 0:
            result += f', {format_number(total["output_thinking"])} thinking'
        result += ')'
        return result

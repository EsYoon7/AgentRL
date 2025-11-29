from __future__ import annotations

from typing import Union

from anthropic.types import MessageParam
from anthropic.types.beta import BetaMessageParam
from openai.types.chat import ChatCompletionMessageParam
from openai.types.responses import ResponseInputItemParam


def trim_images(
    messages: list[Union[ChatCompletionMessageParam, ResponseInputItemParam, BetaMessageParam, MessageParam]],
    max_images: int
) -> list[Union[ChatCompletionMessageParam, ResponseInputItemParam, BetaMessageParam, MessageParam]]:
    images = 0

    messages_reverse = []  # process messages in reverse order
    for message in reversed(messages):
        if isinstance(message, dict):
            for content_key in ('content', 'input', 'output'):
                if not isinstance(message.get(content_key), list):
                    continue
                new_content = []  # process content blocks in reverse order
                for block in reversed(message[content_key]):
                    if isinstance(block, dict):
                        # anthropic tool result block
                        if isinstance(block.get('content'), list):
                            new_block_content = []  # process content blocks in reverse order
                            for sub_block in reversed(block['content']):
                                if sub_block.get('type') in ('image', 'image_url', 'input_image', 'output_image'):
                                    if images < max_images:
                                        images += 1
                                        new_block_content.append(sub_block)
                                else:
                                    new_block_content.append(sub_block)
                            block['content'] = list(reversed(new_block_content))  # reverse back
                            new_content.append(block)
                            continue

                        # ordinary image block
                        if block.get('type') in ('image', 'image_url', 'input_image', 'output_image'):
                            if images < max_images:
                                images += 1
                                new_content.append(block)
                            continue

                    new_content.append(block)
                message[content_key] = list(reversed(new_content))  # reverse back
        messages_reverse.append(message)

    return list(reversed(messages_reverse))  # reverse back

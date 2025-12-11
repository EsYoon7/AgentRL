from __future__ import annotations

import logging
import re
from base64 import b64decode, b64encode
from copy import deepcopy
from io import BytesIO
from typing import Any, Optional, Union

from PIL import Image, ImageFile
from anthropic.types import MessageParam
from anthropic.types.beta import BetaMessageParam
from openai.types.chat import ChatCompletionMessageParam
from openai.types.responses import ResponseInputItemParam

from .data_url import DataUrlUtil

ImageFile.LOAD_TRUNCATED_IMAGES = True

LOG = logging.getLogger(__name__)


def trim_images(
    messages: list[Union[ChatCompletionMessageParam, ResponseInputItemParam, BetaMessageParam, MessageParam]],
    max_images: int
) -> list[Union[ChatCompletionMessageParam, ResponseInputItemParam, BetaMessageParam, MessageParam]]:
    images = 0

    messages_copy = [deepcopy(message) for message in messages]
    messages_reverse = []  # process messages in reverse order
    for message in reversed(messages_copy):
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


def resize_images(
    messages: list[Union[ChatCompletionMessageParam, ResponseInputItemParam, BetaMessageParam, MessageParam]],
    image_size: str
) -> list[Union[ChatCompletionMessageParam, ResponseInputItemParam, BetaMessageParam, MessageParam]]:
    size_spec = image_size.strip()

    if not size_spec:
        raise ValueError('image_size must be a non-empty string')

    size_mode: str
    size_value: Any

    min_match = re.fullmatch(r'(\d+)\+', size_spec)
    max_match = re.fullmatch(r'(\d+)-', size_spec)
    exact_match = re.fullmatch(r'(\d+)[xX](\d+)', size_spec)

    if min_match:
        size_mode = 'min'
        size_value = int(min_match.group(1))
    elif max_match:
        size_mode = 'max'
        size_value = int(max_match.group(1))
    elif exact_match:
        size_mode = 'exact'
        width = int(exact_match.group(1))
        height = int(exact_match.group(2))
        if width <= 0 or height <= 0:
            raise ValueError('image_size dimensions must be positive integers')
        size_value = (width, height)
    else:
        raise ValueError('image_size must be formatted like 512+, 1024-, or 800x600')

    if size_mode in ('min', 'max') and size_value <= 0:
        raise ValueError('image_size must use a positive integer dimension')

    resample_filter = Image.Resampling.LANCZOS

    def compute_target_size(width: int, height: int) -> tuple[int, int]:
        if width <= 0 or height <= 0:
            return width, height
        if size_mode == 'exact':
            return size_value
        if size_mode == 'min':
            current = min(width, height)
            scale = size_value / current
        else:  # max
            current = max(width, height)
            scale = size_value / current
        if abs(scale - 1.0) < 1e-9:
            return width, height
        new_w = max(1, int(round(width * scale)))
        new_h = max(1, int(round(height * scale)))
        return new_w, new_h

    def resize_payload(mime: str, payload: bytes) -> Optional[bytes]:
        try:
            with Image.open(BytesIO(payload)) as img:
                img.load()
                target_width, target_height = compute_target_size(*img.size)
                if (target_width, target_height) == img.size:
                    return None
                resized = img.resize((target_width, target_height), resample=resample_filter)
                format_source = img.format or mime.split('/')[-1] if mime else (img.format or 'PNG')
                format_name = format_source.upper()
                if format_name == 'JPG':
                    format_name = 'JPEG'
                if format_name == 'JPEG' and resized.mode not in ('RGB', 'L'):
                    resized = resized.convert('RGB')
                buffer = BytesIO()
                save_kwargs = {'optimize': True} if format_name == 'JPEG' else {}
                resized.save(buffer, format=format_name, **save_kwargs)
                return buffer.getvalue()
        except Exception:
            LOG.warning('failed to resize image, skipping')
            return None

    def handle_base64_dict(value: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        if value.get('type') != 'base64':
            return False, value
        media_type = value.get('media_type')
        data_str = value.get('data')
        if not isinstance(media_type, str) or not isinstance(data_str, str):
            return True, value
        mime = media_type.lower()
        if not mime.startswith('image/'):
            return True, value
        try:
            payload = b64decode(data_str, validate=False)
        except Exception as e:
            LOG.warning(f'failed to decode anthropic base64 payload: {e}; leaving unmodified')
            return True, value
        resized_payload = resize_payload(mime, payload)
        if resized_payload is None:
            return True, value
        new_value = dict(value)
        new_value['data'] = b64encode(resized_payload).decode('ascii')
        return True, new_value

    def maybe_resize(value: Any) -> Any:
        if isinstance(value, str):
            parsed = DataUrlUtil.parse(value)
            if parsed is None:
                return value
            mime, params, payload, _ = parsed
            if not mime.startswith('image/'):
                return value
            resized_payload = resize_payload(mime, payload)
            if resized_payload is None:
                return value
            return DataUrlUtil.build(mime, resized_payload, params)
        if isinstance(value, dict):
            handled, updated = handle_base64_dict(value)
            if handled:
                return updated
            return {k: maybe_resize(v) for k, v in value.items()}
        if isinstance(value, list):
            return [maybe_resize(v) for v in value]
        if isinstance(value, tuple):
            return tuple(maybe_resize(v) for v in value)
        if isinstance(value, set):
            return {maybe_resize(v) for v in value}
        return value

    messages_copy = [deepcopy(message) for message in messages]
    return maybe_resize(messages_copy)

from __future__ import annotations

import base64
from io import BytesIO

import pytest
from PIL import Image

from agentrl.eval.utils import DataUrlUtil, resize_images


def _make_image_url(size: tuple[int, int], color: tuple[int, int, int] = (255, 0, 0)) -> str:
    img = Image.new('RGB', size, color)
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    encoded = base64.b64encode(buffer.getvalue()).decode('ascii')
    return f'data:image/png;base64,{encoded}'


def _make_base64_source(size: tuple[int, int]) -> dict:
    data_url = _make_image_url(size)
    header, data = data_url.split(',', 1)
    media_type = header.split(';', 1)[0][5:]
    return {'type': 'base64', 'media_type': media_type, 'data': data}


def _get_image_size(data_url: str) -> tuple[int, int]:
    parsed = DataUrlUtil.parse(data_url)
    assert parsed is not None, 'expected valid data url'
    _, _, payload, _ = parsed
    with Image.open(BytesIO(payload)) as img:
        return img.size


def _get_base64_source_size(source: dict) -> tuple[int, int]:
    payload = base64.b64decode(source['data'])
    with Image.open(BytesIO(payload)) as img:
        return img.size


def test_resize_images_exact_size_applies_new_dimensions():
    original = _make_image_url((4, 6))
    messages = [
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': 'hello'},
                {'type': 'image_url', 'image_url': {'url': original}},
            ],
        }
    ]

    resized = resize_images(messages, '8x2')

    new_url = resized[0]['content'][1]['image_url']['url']
    assert _get_image_size(new_url) == (8, 2)


def test_resize_images_min_dimension_handles_nested_structures():
    portrait = _make_image_url((4, 8), color=(0, 255, 0))
    landscape = _make_image_url((6, 3), color=(0, 0, 255))
    text_url = 'data:text/plain;base64,' + base64.b64encode(b'notes').decode('ascii')

    messages = [
        {
            'role': 'user',
            'content': [
                {'type': 'image_url', 'image_url': {'url': portrait}},
                {
                    'type': 'tool_result',
                    'content': [
                        {'type': 'output_image', 'image': {'url': landscape}},
                        'ok',
                        {'nested': landscape},
                    ],
                },
            ],
            'metadata': {
                'attachments': [portrait, {'inner': landscape}],
                'tuple': (portrait,),
                'set': {landscape, 'anchor'},
                'notes': text_url,
            },
        }
    ]

    resized = resize_images(messages, '10+')

    top_url = resized[0]['content'][0]['image_url']['url']
    tool_url = resized[0]['content'][1]['content'][0]['image']['url']

    assert _get_image_size(top_url) == (10, 20)
    assert _get_image_size(tool_url) == (20, 10)
    assert resized[0]['metadata']['notes'] == text_url

    attachment_url = resized[0]['metadata']['attachments'][0]
    tuple_url = resized[0]['metadata']['tuple'][0]
    for candidate in (attachment_url, tuple_url):
        assert _get_image_size(candidate) == (10, 20)

    set_payloads = {
        value for value in resized[0]['metadata']['set'] if isinstance(value, str) and value.startswith('data:image/png')
    }
    assert len(set_payloads) == 1
    assert _get_image_size(next(iter(set_payloads))) == (20, 10)


def test_resize_images_min_dimension_downsizes_when_needed():
    img_url = _make_image_url((50, 100))
    messages = [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': img_url}}]}]

    resized = resize_images(messages, '20+')

    new_url = resized[0]['content'][0]['image_url']['url']
    assert _get_image_size(new_url) == (20, 40)


def test_resize_images_max_dimension_skips_non_images():
    wide = _make_image_url((200, 50))
    text_url = 'data:text/plain;base64,' + base64.b64encode(b'skip').decode('ascii')

    messages = [
        {
            'role': 'assistant',
            'content': [
                {'type': 'text', 'text': 'here'},
                {'type': 'image_url', 'image_url': {'url': wide}},
            ],
            'extra': text_url,
        }
    ]

    resized = resize_images(messages, '60-')

    new_url = resized[0]['content'][1]['image_url']['url']
    assert _get_image_size(new_url) == (60, 15)
    assert resized[0]['extra'] == text_url


def test_resize_images_max_dimension_upscales_when_smaller():
    img_url = _make_image_url((30, 20))
    messages = [{'role': 'assistant', 'content': [{'type': 'image_url', 'image_url': {'url': img_url}}]}]

    resized = resize_images(messages, '120-')

    new_url = resized[0]['content'][0]['image_url']['url']
    assert _get_image_size(new_url) == (120, 80)


def test_resize_images_handles_anthropic_base64_sources():
    source = _make_base64_source((12, 4))
    messages = [
        {
            'role': 'assistant',
            'content': [
                {'type': 'image', 'source': source},
            ],
        }
    ]

    resized = resize_images(messages, '6-')

    new_source = resized[0]['content'][0]['source']
    assert new_source['type'] == 'base64'
    assert _get_base64_source_size(new_source) == (6, 2)


def test_resize_images_leaves_non_image_base64_sources():
    source = {
        'type': 'base64',
        'media_type': 'text/plain',
        'data': base64.b64encode(b'notes').decode('ascii'),
    }
    messages = [{'role': 'assistant', 'content': [{'type': 'image', 'source': source}]}]

    resized = resize_images(messages, '50+')

    assert resized[0]['content'][0]['source']['data'] == source['data']


def test_resize_images_handles_invalid_base64_data_gracefully():
    source = {'type': 'base64', 'media_type': 'image/png', 'data': 'not-base64'}
    messages = [{'role': 'assistant', 'content': [{'type': 'image', 'source': source}]}]

    resized = resize_images(messages, '80+')

    assert resized[0]['content'][0]['source']['data'] == source['data']


def test_resize_images_does_not_mutate_original_messages():
    original_url = _make_image_url((10, 5))
    message = {
        'role': 'user',
        'content': [{'type': 'image_url', 'image_url': {'url': original_url}}],
    }

    resized = resize_images([message], '20+')

    assert resized[0] is not message
    assert message['content'][0]['image_url']['url'] == original_url
    assert _get_image_size(resized[0]['content'][0]['image_url']['url']) == (40, 20)


@pytest.mark.parametrize('spec', ['', 'foo', '100', '10*10', 'x600', '0+', '-5+', '10x0', '0x10'])
def test_resize_images_invalid_size_spec(spec):
    with pytest.raises(ValueError):
        resize_images([], spec)

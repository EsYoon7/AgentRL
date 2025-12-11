from agentrl.eval.utils.messages import trim_images

IMAGE_TYPES = {'image', 'image_url', 'input_image', 'output_image'}


def _count_image_blocks(message):
    total = 0
    for key in ('content', 'input', 'output'):
        for block in message.get(key, []) or []:
            total += _count_in_block(block)
    return total


def _count_in_block(block):
    if not isinstance(block, dict):
        return 0
    total = 0
    if block.get('type') in IMAGE_TYPES:
        total += 1
    if isinstance(block.get('content'), list):
        for sub_block in block['content']:
            total += _count_in_block(sub_block)
    return total


def test_trim_images_zero_limit_removes_all_images():
    user_message = {
        'role': 'user',
        'content': [
            {'type': 'text', 'text': 'hello'},
            {'type': 'image', 'url': 'old'},
        ],
        'input': [
            {'type': 'input_image', 'url': 'input-old'},
        ],
    }

    result = trim_images([user_message], max_images=0)

    assert _count_image_blocks(result[0]) == 0
    assert result[0]['content'] == [{'type': 'text', 'text': 'hello'}]
    assert result[0]['input'] == []


def test_trim_images_prefers_recent_messages():
    older = {
        'role': 'user',
        'content': [
            {'type': 'image', 'url': 'old-1'},
            {'type': 'text', 'text': 'older text'},
            {'type': 'image', 'url': 'old-2'},
        ],
    }
    newer = {
        'role': 'assistant',
        'content': [
            {'type': 'text', 'text': 'reply'},
        ],
        'output': [
            {'type': 'output_image', 'url': 'recent'},
        ],
    }

    result = trim_images([older, newer], max_images=1)

    assert _count_image_blocks(result[1]) == 1
    assert _count_image_blocks(result[0]) == 0
    assert result[0]['content'][0]['type'] == 'text'
    assert result[1]['output'][0]['url'] == 'recent'


def test_trim_images_handles_nested_tool_blocks():
    tool_message = {
        'role': 'system',
        'content': [
            {
                'type': 'tool_result',
                'content': [
                    {'type': 'image', 'url': 'nested-old'},
                    {'type': 'text', 'text': 'nested desc'},
                    {'type': 'image', 'url': 'nested-recent'},
                ],
            }
        ],
    }

    result = trim_images([tool_message], max_images=1)
    nested_content = result[0]['content'][0]['content']

    assert _count_image_blocks(result[0]) == 1
    assert nested_content == [
        {'type': 'text', 'text': 'nested desc'},
        {'type': 'image', 'url': 'nested-recent'},
    ]


def test_trim_images_does_not_mutate_original_messages():
    message = {
        'role': 'user',
        'content': [
            {'type': 'text', 'text': 'keep'},
            {'type': 'image', 'url': 'original'},
        ],
    }

    result = trim_images([message], max_images=0)

    assert result[0] is not message
    assert message['content'][1] == {'type': 'image', 'url': 'original'}
    assert len(result[0]['content']) == 1
    assert result[0]['content'][0]['type'] == 'text'

from typing import Optional

import pytest

from agentrl.eval.utils import normalize_model_name


@pytest.mark.parametrize(
    ('raw_name', 'expected'),
    [
        ('GPT-4o', 'gpt-4o'),
        ('openai:gpt-4o', 'gpt-4o'),
        ('openai/gpt-4.1-mini', 'gpt-4.1-mini'),
        ('/path/model-a', 'model-a'),
        ('self-hosted:models/model-b/global_step_42', 'model-b-step42'),
        ('/nested/path/model-c/global_step_123/', 'model-c-step123')
    ]
)
def test_normalize_model_name_happy_path(raw_name: str, expected: str) -> None:
    assert normalize_model_name(raw_name) == expected


@pytest.mark.parametrize(
    ('raw_name', 'thinking', 'expected'),
    [
        ('claude-4-5', True, 'claude-4-5-thinking'),
        ('claude-4-5', False, 'claude-4-5'),
        ('claude-4-5', None, 'claude-4-5'),
        ('anthropic:claude-4-5', True, 'claude-4-5-thinking'),
        ('anthropic:claude-4-5', False, 'claude-4-5'),
        ('anthropic:claude-4-5', None, 'claude-4-5'),
        ('gpt-4o', None, 'gpt-4o'),
        ('o4-mini', None, 'o4-mini')
    ]
)
def test_normalize_model_name_thinking_suffix(
    raw_name: str,
    thinking: Optional[bool],
    expected: str
) -> None:
    assert normalize_model_name(raw_name, thinking=thinking) == expected

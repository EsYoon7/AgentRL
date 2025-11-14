from __future__ import annotations

from types import SimpleNamespace

from agentrl.eval.session.tokens import TokenCounter


def _make_usage(**kwargs):
    return SimpleNamespace(**kwargs)


def test_token_counter_with_prompt_details_and_completion_details() -> None:
    counter = TokenCounter()
    usage = _make_usage(
        input_tokens=100,
        prompt_tokens_details=SimpleNamespace(cached_tokens=12, image_tokens=5),
        output_tokens=80,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=20),
    )

    counter.add_from_usage(usage)

    assert counter.total() == {
        'input_total': 100,
        'input_cached': 12,
        'input_image': 5,
        'output_total': 80,
        'output_thinking': 20,
        'total': 180,
    }


def test_token_counter_accumulates_multiple_usage_sources() -> None:
    counter = TokenCounter()
    first_usage = _make_usage(
        input_tokens=60,
        prompt_tokens_details=SimpleNamespace(cached_tokens=4, image_tokens=0),
        output_tokens=30,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=6),
    )
    second_usage = _make_usage(
        prompt_tokens=40,
        cache_read_input_tokens=7,
        completion_tokens=20,
        output_tokens_details=SimpleNamespace(reasoning_tokens=3),
    )

    counter.add_from_usage(first_usage)
    counter.add_from_usage(second_usage)

    assert counter.input == 100
    assert counter.input_cached == 11
    assert counter.input_image == 0
    assert counter.output == 50
    assert counter.output_thinking == 9
    assert counter.total() == {
        'input_total': 100,
        'input_cached': 11,
        'input_image': 0,
        'output_total': 50,
        'output_thinking': 9,
        'total': 150,
    }


def test_token_counter_uses_input_token_details_branch() -> None:
    counter = TokenCounter()
    token_details = SimpleNamespace(cached_tokens=8, image_tokens=2)
    usage = _make_usage(
        input_tokens=None,
        prompt_tokens=None,
        input_token_details=token_details,
        input_tokens_details=token_details,
        output_tokens=None,
        completion_tokens=0,
    )

    counter.add_from_usage(usage)

    assert counter.input == 0
    assert counter.input_cached == 8
    assert counter.input_image == 2
    assert counter.output == 0
    assert counter.output_thinking == 0
    assert counter.total() == {
        'input_total': 0,
        'input_cached': 8,
        'input_image': 2,
        'output_total': 0,
        'output_thinking': 0,
        'total': 0,
    }


def test_token_counter_summary_includes_all_sections() -> None:
    counter = TokenCounter()
    usage = _make_usage(
        input_tokens=150,
        prompt_tokens_details=SimpleNamespace(cached_tokens=50, image_tokens=25),
        output_tokens=350,
        completion_tokens_details=SimpleNamespace(reasoning_tokens=100),
    )

    counter.add_from_usage(usage)

    assert counter.summary() == '500 tokens used (150 input, 50 cached, 25 image; 350 output, 100 thinking)'


def test_token_counter_summary_handles_singular_case() -> None:
    counter = TokenCounter()
    usage = _make_usage(input_tokens=1, output_tokens=0)

    counter.add_from_usage(usage)

    assert counter.summary() == '1 token used (1 input; 0 output)'

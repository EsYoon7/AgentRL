from __future__ import annotations

import pytest

from agentrl.eval.utils import format_number


@pytest.mark.parametrize(
    ('value', 'expected'),
    [
        (0, '0'),
        (42.7, '43'),
        (-12.2, '-12'),
    ],
)
def test_format_number_handles_values_below_one_thousand(value: float, expected: str) -> None:
    assert format_number(value) == expected


def test_format_number_scales_large_numbers_and_trims_suffixes() -> None:
    assert format_number(1500) == '1.5k'
    assert format_number(2_000_000) == '2m'
    assert format_number(3_450_000_000, digits=2) == '3.45b'

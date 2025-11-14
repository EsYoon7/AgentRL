from pathlib import Path

import pytest

from agentrl.eval.store.list import ResultList


@pytest.mark.parametrize(
    ('model', 'task', 'view_only', 'expected'),
    [
        ('gpt-4o', 'demo-task', True, 'gpt-4o-demo-task-'),
        ('gpt-4o', 'demo-task', False, 'gpt-4o-demo-task-'),
        ('gpt-4o', None, True, 'gpt-4o-'),
        ('gpt-4o', None, False, 'gpt-4o-multi-task-'),
        (None, 'demo-task', True, 'multi-model-demo-task-'),
        (None, 'demo-task', False, 'multi-model-demo-task-'),
        (None, None, True, ''),
        (None, None, False, 'multi-model-multi-task-')
    ]
)
def test_list_init_prefix(model, task, view_only, expected):
    assert ResultList(
        base_dir=Path(''),
        model=model,
        task=task,
        view_only=view_only
    ).prefix == expected

from __future__ import annotations

import re
from typing import Any, Union, Optional


def format_number(n: Union[float, int], digits: int = 1) -> str:
    units = ['', 'k', 'm', 'b', 't']
    sign = '-' if n < 0 else ''
    n = abs(float(n))

    idx = 0
    if n < 1000:
        s = f'{n:.0f}'
    else:
        while n >= 1000 and idx < len(units) - 1:
            n /= 1000.0
            idx += 1
        s = f'{n:.{digits}f}'.rstrip('0').rstrip('.')

    return f'{sign}{s}{units[idx]}'


def model_dump(model: Any) -> Any:
    if hasattr(model, 'model_dump'):
        return model.model_dump(mode='json', exclude_unset=True)
    elif hasattr(model, 'to_dict'):
        return model.to_dict()
    elif isinstance(model, (list, tuple, set)):
        return [model_dump(item) for item in model]
    elif isinstance(model, dict):
        return {k: model_dump(v) for k, v in model.items()}
    return model


def normalize_model_name(name: str, thinking: Optional[bool] = None) -> str:
    name = name.lower()

    # remove provider prefix
    if ':' in name:
        name = name.split(':', maxsplit=1)[1]

    # match model name with path
    match = re.search(r'/?([^/]+)(?:/global_step_(\d+)/?)?$', name)
    if match is not None:
        name, step = match.groups()
        name = f'{name}-step{step}' if step else name

    # add thinking suffix if requested
    if thinking and 'thinking' not in name:
        name += '-thinking'

    return name

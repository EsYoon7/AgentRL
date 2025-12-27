import re
from typing import Union


def parse_duration(duration_str: str,
                   default_unit: str = 's',
                   return_seconds: bool = False) -> Union[int, float]:
    """
    Parse a duration string (e.g., '5s', '10ms', '2us', or '12' with default unit)
    and return nanoseconds (or seconds if return_seconds is set) as an integer.

    Supports the units:
      ns (nanoseconds), us (microseconds), µs (microseconds), ms (milliseconds),
      s (seconds), m (minutes), h (hours).

    If the unit is missing, uses default_unit (default is 's').
    """
    if not isinstance(duration_str, str):
        duration_str = str(duration_str)

    multipliers = {
        'ns': 1,
        'us': 1_000,
        'µs': 1_000,  # microsecond sometimes written as µs
        'ms': 1_000_000,
        's': 1_000_000_000,
        'm': 60 * 1_000_000_000,
        'h': 3600 * 1_000_000_000,
    }
    if default_unit not in multipliers:
        raise ValueError(f"Unknown default unit: '{default_unit}'")

    pattern = re.compile(r'^\s*(\d+(?:\.\d+)?)([a-zA-Zµ]*)\s*$')
    match = pattern.fullmatch(duration_str)
    if not match:
        raise ValueError(f"Invalid duration format: '{duration_str}'")
    value, unit = match.groups()
    unit = unit or default_unit
    if unit not in multipliers:
        raise ValueError(f"Unknown duration unit: '{unit}'")
    ns = float(value) * multipliers[unit]
    if return_seconds:
        return ns / multipliers['s']
    return int(ns)

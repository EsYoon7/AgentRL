from __future__ import annotations

import json
from typing import Any, Sequence

from ..utils import model_dump


class Convertible:

    def __init__(self, content: Any):
        self.content = content

    def __repr__(self):
        return self.content.__repr__()

    def __str__(self):
        return json.dumps(self.native(), ensure_ascii=False, indent=None, sort_keys=True)

    def native(self) -> Any:
        return model_dump(self.content)

    @staticmethod
    def dump_all(items: Sequence[Convertible],
                 flatten: bool = True) -> list[Any]:
        if not items:
            return []

        result: list[Any] = []
        for item in items:
            dumped = item.native()
            if flatten and isinstance(dumped, list):
                result.extend(dumped)
            else:
                result.append(dumped)

        return result

    @staticmethod
    def convert_all(items: Sequence[Convertible],
                    to: str,
                    flatten: bool = True,
                    ignore_errors: bool = False) -> list[Any]:
        if not items:
            return []

        result: list[Any] = []
        convert_fn = f'to_{to.replace("-", "_").lower()}'
        for item in items:
            try:
                converted = getattr(item, convert_fn)()
                if flatten and isinstance(converted, list):
                    result.extend(converted)
                else:
                    result.append(converted)
            except ConversionError:
                if not ignore_errors:
                    raise

        return result


class ConversionError(Exception):
    pass

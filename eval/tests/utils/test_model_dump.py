from __future__ import annotations

from agentrl.eval.utils import model_dump


class _ModelDumpOnly:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def model_dump(self, *, mode: str, **_) -> dict[str, str]:
        self.calls.append({'mode': mode})
        return {'dumped': 'yes', 'mode': mode}


class _ToDictOnly:
    def __init__(self) -> None:
        self.calls: int = 0

    def to_dict(self) -> dict[str, str]:
        self.calls += 1
        return {'dumped': 'yes'}


def test_model_dump_prefers_model_dump_method() -> None:
    obj = _ModelDumpOnly()

    result = model_dump(obj)

    assert result == {'dumped': 'yes', 'mode': 'json'}
    assert obj.calls == [{'mode': 'json'}]


def test_model_dump_falls_back_to_to_dict() -> None:
    obj = _ToDictOnly()

    result = model_dump(obj)

    assert result == {'dumped': 'yes'}
    assert obj.calls == 1


def test_model_dump_returns_original_when_no_conversion() -> None:
    obj = {'plain': 'object'}

    result = model_dump(obj)

    assert result == obj


def test_model_dump_converts_iterable_list() -> None:
    class _Dumpable:
        def __init__(self, value: int) -> None:
            self.value = value

        def to_dict(self) -> dict[str, int]:
            return {'value': self.value}

    dumped = model_dump([_Dumpable(1), _Dumpable(2)])

    assert dumped == [{'value': 1}, {'value': 2}]


def test_model_dump_converts_iterable_tuple() -> None:
    class _Dumpable:
        def __init__(self, text: str) -> None:
            self.text = text

        def to_dict(self) -> dict[str, str]:
            return {'text': self.text}

    dumped = model_dump((_Dumpable('first'), _Dumpable('second')))

    assert dumped == [{'text': 'first'}, {'text': 'second'}]


def test_model_dump_converts_iterable_set() -> None:
    class _Dumpable:
        def __init__(self, label: str) -> None:
            self.label = label

        def to_dict(self) -> dict[str, str]:
            return {'label': self.label}

    dumped = model_dump({_Dumpable('a'), _Dumpable('b')})

    assert sorted(dumped, key=lambda item: item['label']) == [
        {'label': 'a'},
        {'label': 'b'},
    ]


def test_model_dump_converts_iterable_dict() -> None:
    class _Dumpable:
        def __init__(self, number: int) -> None:
            self.number = number

        def to_dict(self) -> dict[str, int]:
            return {'number': self.number}

    payload = {
        'alpha': _Dumpable(1),
        'beta': [_Dumpable(2)],
    }

    dumped = model_dump(payload)

    assert dumped['alpha'] == {'number': 1}
    assert dumped['beta'] == [{'number': 2}]

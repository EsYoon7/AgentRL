import asyncio
from datetime import datetime, UTC
from pathlib import Path
from typing import Union, Optional

import pytest

from agentrl.eval.session.metric import Metric
from agentrl.eval.session.types import MetricType, RunResult, SampleStatus
from agentrl.eval.store.store import ResultStore


def _make_store(tmp_path: Path) -> ResultStore:
    store_dir = tmp_path / 'store'
    store_dir.mkdir()
    return ResultStore(store_dir)


def _build_result(
    *,
    index: Union[str, int],
    status: SampleStatus,
    reward: float,
    score: float,
    session_id: int = 1,
    run: int = 1,
    model: str = 'concurrency-model',
    task: str = 'concurrency-task'
) -> RunResult:
    return RunResult(
        model=model,
        session_id=session_id,
        run=run,
        task=task,
        index=index,
        status=status,
        reward=reward,
        score=score,
        metrics=None,
        result=None,
        task_trace=None,
        raw_trace=None,
        time_start=datetime(2025, 1, 1, 0, 0, 0, tzinfo=UTC),
        time_end=datetime(2025, 1, 1, 0, 1, 0, tzinfo=UTC)
    )


def _index_value(row: dict) -> Optional[Union[str, int]]:
    index_field = row['index']
    if index_field is None:
        return None
    if index_field['int_value'] is not None:
        return index_field['int_value']
    return index_field['str_value']


async def _record_with_delay(store: ResultStore, result: RunResult, delay: float):
    await asyncio.sleep(delay)
    await store.record(result)


async def _record_all(store: ResultStore, results: list[RunResult]):
    tasks = [asyncio.create_task(store.record(result)) for result in results]
    await asyncio.gather(*tasks)
    await store.save(force=True)


def _snapshot_results(store: ResultStore) -> list[tuple]:
    return sorted(
        (
            (
                row['session_id'],
                row['run'],
                row['task'],
                _index_value(row),
                row['status'],
                row['metric_reward'],
                row['metric_score']
            )
            for row in store.results.to_dicts()
        )
    )


def test_high_concurrency_inserts(tmp_path: Path):
    store = _make_store(tmp_path)
    total = 40
    results = [
        _build_result(
            index=i,
            status=SampleStatus.COMPLETED,
            reward=float(i) * 0.1,
            score=float(i) * 0.3,
            session_id=i + 1
        )
        for i in range(total)
    ]

    asyncio.run(_record_all(store, results))

    rows = store.results.to_dicts()
    assert len(rows) == total
    assert {row['session_id'] for row in rows} == set(range(1, total + 1))
    assert { _index_value(row) for row in rows } == set(range(total))


def test_racing_updates_same_key(tmp_path: Path):
    store = _make_store(tmp_path)
    base_result = _build_result(
        index='race',
        status=SampleStatus.RUNNING,
        reward=0.25,
        score=0.1,
        session_id=1
    )
    final_result = _build_result(
        index='race',
        status=SampleStatus.COMPLETED,
        reward=0.75,
        score=0.85,
        session_id=1
    )

    async def runner():
        await asyncio.gather(
            _record_with_delay(store, base_result, 0.01),
            _record_with_delay(store, final_result, 0.02)
        )
        await store.save(force=True)

    asyncio.run(runner())

    rows = store.results.to_dicts()
    assert len(rows) == 1
    row = rows[0]
    assert row['status'] == SampleStatus.COMPLETED
    assert row['metric_reward'] == pytest.approx(final_result.reward)
    assert row['metric_score'] == pytest.approx(final_result.score)


def test_metrics_after_concurrent_updates(tmp_path: Path):
    store = _make_store(tmp_path)
    results = []
    for i in range(10):
        status = SampleStatus.COMPLETED if i % 2 == 0 else SampleStatus.RUNNING
        results.append(
            _build_result(
                index=str(i),
                status=status,
                reward=1.0 + i,
                score=0.5 + i
            )
        )

    asyncio.run(_record_all(store, results))

    reward_metric = Metric(MetricType.REWARD)
    metrics_items = asyncio.run(store.metrics(reward_metric))

    assert len(metrics_items) == 1
    item = metrics_items[0]
    completed_rewards = [result.reward for result in results if result.status.is_completed()]
    expected_avg = sum(completed_rewards) / len(completed_rewards)

    assert item.valid == len(completed_rewards)
    assert item.avg == pytest.approx(expected_avg)
    assert item.bon == pytest.approx(expected_avg)


def test_completed_metrics_save_no_side_effects(tmp_path: Path):
    store = _make_store(tmp_path)
    results = [
        _build_result(
            index=i,
            status=SampleStatus.COMPLETED if i % 2 == 0 else SampleStatus.RUNNING,
            reward=0.4 + i,
            score=0.6 + i,
            session_id=i + 10
        )
        for i in range(6)
    ]

    asyncio.run(_record_all(store, results))
    before = _snapshot_results(store)

    asyncio.run(store.completed())
    assert before == _snapshot_results(store)

    reward_metric = Metric(MetricType.REWARD)
    asyncio.run(store.metrics(reward_metric))
    assert before == _snapshot_results(store)

    asyncio.run(store.save(force=True))
    assert before == _snapshot_results(store)

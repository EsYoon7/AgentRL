from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

import polars as pl

from .types import MetricsListItem
from ..session.metric import Metric
from ..session.types import RunResult, RunSpec, SampleStatus
from ..utils import DataUrlUtil

SCHEMA: pl.Schema = pl.Schema({
    'model': pl.Categorical(),
    'session_id': pl.Int64(),
    'run': pl.Int64(),
    'task': pl.Categorical(),
    'index': pl.Struct({
        'int_value': pl.Int64(),
        'str_value': pl.String()
    }),
    'status': pl.Enum(SampleStatus),
    'metric_reward': pl.Float64(),
    'metric_score': pl.Float64(),
    'metric_success_rate': pl.Float64(),
    'metrics': pl.String(),
    'result': pl.String(),
    'task_trace': pl.String(),
    'raw_trace': pl.String(),
    'ts_start': pl.Datetime(time_unit='ms', time_zone='UTC'),
    'ts_end': pl.Datetime(time_unit='ms', time_zone='UTC'),
})


class ResultStore:

    def __init__(self,
                 path: Path,
                 *,
                 resume: bool = False,
                 multi_model: bool = True,
                 multi_task: bool = True):
        self.logger = logging.getLogger(__name__)

        self.path = path
        if not self.path.is_dir():
            if resume:
                raise NotADirectoryError(f'path "{self.path}" does not exist or is not a directory.')
            elif self.path.exists():
                raise NotADirectoryError(f'path "{self.path}" exists but is not a directory.')
        self.multi_model = multi_model
        self.multi_task = multi_task

        # try to read existing results if exists
        self.results = pl.DataFrame(schema=SCHEMA)
        self.results_path = self.path / 'results.jsonl'
        if self.results_path.is_file():
            try:
                self.logger.info('reading existing results from "%s"', self.results_path)
                self.load_sync()
            except Exception:
                self.logger.warning(f'failed to load existing results from "%s"',
                                    self.results_path, exc_info=True)

        self._lock = asyncio.Lock()
        self._results_last_save = 0.0

    def log_path(self) -> Path:
        return self.path / 'run.log'

    def session_path(self, spec: RunSpec, session_id: int, ts: int) -> Path:
        path = self.path / '-'.join([str(i) for i in [
            spec.model if self.multi_model else None,
            spec.task if self.multi_task else None,
            spec.index,
            spec.run,
            session_id,
            ts
        ] if i is not None]).replace('/', '_')

        if path.exists() and not path.is_dir():
            raise NotADirectoryError(f'session path "{path}" exists and is not a directory.')

        return path

    async def completed(self) -> dict[str, RunResult]:
        async with self._lock:
            df = await (
                self.results.lazy()
                .filter(pl.col('status').is_in(SampleStatus.completed_statuses()))
                .select([
                    'model', 'run', 'task', 'index',
                    'session_id', 'status',
                    'metric_reward', 'metric_score',
                    'ts_start', 'ts_end'
                ])
                .collect_async()
            )

        result: dict[str, RunResult] = {}
        for row in df.iter_rows():
            spec = RunSpec(
                model=row[0],
                run=row[1],
                task=row[2],
                index=(
                    row[3]['int_value'] if row[3]['int_value'] is not None else row[3]['str_value']
                    if row[3] is not None else None
                ),
                custom_params=None
            )
            result[spec.run_key()] = RunResult(
                model=spec.model,
                session_id=row[4],
                run=spec.run,
                task=spec.task,
                index=spec.index,
                status=row[5],
                reward=row[6],
                score=row[7],
                metrics=None,
                result=None,
                task_trace=None,
                raw_trace=None,
                time_start=row[8] if row[8] is not None else None,
                time_end=row[9] if row[9] is not None else None,
            )

        return result

    async def record(self, result: RunResult):
        # pre-process: extract trace data to separate files
        session_path = self.session_path(result.spec(), result.session_id, int(time.time()))

        files_seen = {}
        next_index = 1

        if result.task_trace is not None:
            trace, next_index = DataUrlUtil.extract(
                obj=result.task_trace,
                base_path=session_path,
                start_index=next_index,
                _seen=files_seen
            )
            task_trace_path = session_path / 'task_trace.json'
            session_path.mkdir(parents=True, exist_ok=True)
            with open(task_trace_path, 'w', encoding='utf-8') as f:
                json.dump(trace, f, ensure_ascii=False, indent=2)

        if result.raw_trace is not None:
            trace, next_index = DataUrlUtil.extract(
                obj=result.raw_trace,
                base_path=session_path,
                start_index=next_index,
                _seen=files_seen
            )
            raw_trace_path = session_path / 'trace.json'
            session_path.mkdir(parents=True, exist_ok=True)
            with open(raw_trace_path, 'w', encoding='utf-8') as f:
                json.dump(trace, f, ensure_ascii=False, indent=2)

        # upsert the new row into the results DataFrame
        async with self._lock:
            self.results = await self.results.lazy().update(
                pl.LazyFrame({
                    'model': [result.model],
                    'session_id': [result.session_id],
                    'run': [result.run],
                    'task': [result.task],
                    'index': [{
                        'int_value': result.index if isinstance(result.index, int) else None,
                        'str_value': result.index if isinstance(result.index, str) else None,
                    } if result.index is not None else None],
                    'status': [result.status],
                    **{f'metric_{k.value}': [v] for k, v in Metric.all(result).items()},
                    'result': [json.dumps(result.result, ensure_ascii=False) if result.result is not None else None],
                    'metrics': [json.dumps(result.metrics, ensure_ascii=False) if result.metrics is not None else None],
                    'task_trace': [str(task_trace_path.relative_to(self.path)) if result.task_trace is not None else None],
                    'raw_trace': [str(raw_trace_path.relative_to(self.path)) if result.raw_trace is not None else None],
                    'ts_start': [result.time_start],
                    'ts_end': [result.time_end],
                }, schema=SCHEMA),
                on=['model', 'run', 'task', 'index'],
                how='full',
                include_nulls=True
            ).collect_async()

        await self.save()

    async def metrics(self, metric: Metric) -> list[MetricsListItem]:
        col = f'metric_{metric.type.value}'

        async with self._lock:
            metric_values = pl.col(col).drop_nans().drop_nulls()
            run_level = (
                self.results.lazy()
                .filter(pl.col('status').is_in(SampleStatus.completed_statuses()))
                .group_by(['model', 'task', 'run'])
                .agg([
                    metric_values.count().alias('run_valid'),
                    metric_values.sum().alias('run_sum'),
                    metric_values.mean().alias('run_avg')
                ])
            )

            total_valid = pl.col('run_valid').sum()
            total_sum = pl.col('run_sum').sum()
            run_avg = pl.col('run_avg').drop_nulls()

            df = await (
                run_level
                .group_by(['model', 'task'])
                .agg([
                    total_valid.alias('valid'),
                    pl.when(total_valid > 0)
                        .then(total_sum / total_valid)
                        .otherwise(None)
                        .alias('avg'),
                    pl.when(run_avg.count() > 1)
                        .then(run_avg.std())
                        .otherwise(None)
                        .alias('std'),
                    pl.when(run_avg.count() > 0)
                        .then(run_avg.max())
                        .otherwise(None)
                        .alias('bon')
                ])
                .filter(pl.col('valid') > 0)
                .collect_async()
            )

        return [MetricsListItem.model_validate(m) for m in df.to_dicts()]

    def load_sync(self):
        try:
            self.results = pl.read_ndjson(self.results_path, schema=SCHEMA)
        except Exception as e:
            self.results = pl.read_ndjson(self.results_path, schema=SCHEMA, ignore_errors=True)
            self.logger.warning(f'error parsing saved results: {e}, ignoring these values')

    async def load(self):
        async with self._lock:
            try:
                self.results = await pl.scan_ndjson(self.results_path, schema=SCHEMA).collect_async()
            except Exception as e:
                self.results = await pl.scan_ndjson(self.results_path, schema=SCHEMA, ignore_errors=True).collect_async()
                self.logger.warning(f'error parsing saved results: {e}, ignoring these values')

    async def save(self, force: bool = False):
        """
        Save the data to a jsonl file.
        Debounced if force is False.
        """
        async with self._lock:
            if not force and (time.time() - self._results_last_save) < 10.0:
                return

            self.results_path.parent.mkdir(parents=True, exist_ok=True)
            await self.results.lazy().sink_ndjson(self.results_path, lazy=True).collect_async()
            self._results_last_save = time.time()

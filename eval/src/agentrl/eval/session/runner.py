from __future__ import annotations

import asyncio
import logging
import random
from pathlib import Path
from typing import Any, Optional, Sequence, TYPE_CHECKING

from .types import RunSpec, TaskIndex
from .metric import Metric
from .workflow import SampleWorkflow
from ..event.bus import EventBus
from ..event.types import (MetricsEvent,
                           ResultListEvent,
                           SessionCompletedEvent,
                           SessionStartedEvent,
                           SpecsEvent,
                           WorkflowCompletedEvent,
                           WorkflowStartEvent)
from ..store.list import ResultList
from ..utils import normalize_model_name, setup_file_logging

if TYPE_CHECKING:
    from .controller import ControllerClient
    from .tokens import TokenCounter
    from ..client import BaseClient
    from ..store.store import ResultStore


class EvaluationRunner:

    def __init__(self,
                 *,
                 event_bus: EventBus,
                 metric: Metric,
                 output_dir: Path,
                 concurrency: Optional[int] = None,
                 controller: Optional[ControllerClient] = None,
                 controller_renew: bool = False,
                 cross_sample: bool = False,
                 custom_params: Optional[dict[str, Any]] = None,
                 indices: Optional[set[TaskIndex]] = None,
                 interactive: bool = False,
                 models: Optional[Sequence[BaseClient]] = None,
                 resume: Optional[Path] = None,
                 runs: Optional[int] = None,
                 start_sample: bool = False,
                 tasks: Optional[set[str]] = None,
                 token_counter: Optional[TokenCounter] = None,
                 view_only: bool = False):
        self.logger = logging.getLogger(__name__)

        self.concurrency = concurrency
        self.controller = controller
        self.controller_renew = controller_renew
        self.cross_sample = cross_sample
        self.custom_params = custom_params or {}
        self.event_bus = event_bus
        self.indices = indices
        self.interactive = interactive
        self.metric = metric
        self.models = models or []
        self.output_dir = output_dir
        self.resume = resume
        self.runs = runs
        self.start_sample = start_sample
        self.tasks = tasks
        self.token_counter = token_counter
        self.view_only = view_only

        self._result_list: Optional[ResultList] = None
        self._model_clients: dict[str, Sequence[BaseClient]] = {}
        self._specs_queue: Optional[asyncio.Queue[RunSpec]] = None

    async def gather_models(self) -> list[str]:
        if self.start_sample:
            return ['start-sample']

        if not self.models:
            return []

        if not self._model_clients:
            model_name_futures = [
                asyncio.create_task(client.get_model_name())
                for client in self.models
            ]
            await asyncio.gather(*model_name_futures)

            self._model_clients: dict[str, Sequence[BaseClient]] = {
                normalize_model_name(
                    name=future.result(),
                    thinking=getattr(client, 'thinking', None)  # thinking suffix for anthropic clients
                ): [client]
                for future, client in zip(model_name_futures, self.models)
            }

        model_names = list(self._model_clients.keys())

        if self.cross_sample:
            joined_model_name = '-'.join(model_names)
            self._model_clients[joined_model_name] = self.models
            return [joined_model_name]

        return model_names

    async def gather_tasks(self) -> list[RunSpec]:
        if self.controller is None or self.runs is None or self.tasks is None:
            self.logger.error('both controller, runs and tasks must be specified to gather tasks.')
            return []

        model_names = await self.gather_models()

        indices_futures = {
            task_name: asyncio.create_task(self.controller.get_indices(task_name))
            for task_name in self.tasks
        }
        await asyncio.gather(*indices_futures.values())

        indices = [
            (task, index)
            for task, indices_future in indices_futures.items()
            for index in indices_future.result()
            if index != -1 and (not self.indices or index in self.indices)
        ]
        self.logger.debug('indices: %s', indices)
        self.logger.info('gathered %d %s to evaluate %d model%s for %d run%s',
                         len(indices), 'indices' if len(indices) != 1 else 'index',
                         len(model_names), 's' if len(model_names) != 1 else '',
                         self.runs, 's' if self.runs != 1 else '')

        # evaluation order:
        # 1. complete all tasks for one run, then next run
        # 2. randomize order of tasks within each run
        specs = []
        for run in range(self.runs):
            random.shuffle(indices)
            for task, index in indices:
                for model in model_names:
                    specs.append(RunSpec(
                        model=model,
                        run=run,
                        task=task,
                        index=index,
                        custom_params=self.custom_params
                    ))
        return specs

    async def get_result_list(self) -> ResultList:
        if self._result_list is None:
            models = await self.gather_models()
            tasks = list(self.tasks) if self.tasks else []

            self._result_list = ResultList(
                base_dir=self.output_dir,
                model=models[0] if len(models) == 1 else None,
                task=tasks[0] if len(tasks) == 1 else None,
                view_only=self.view_only
            )

        return self._result_list

    async def run(self):
        result_list = await self.get_result_list()

        if self.resume:
            # resume path provided, force using it as the result store
            store = result_list.path(self.resume, resume=True)
        elif self.start_sample:
            # start-sample only mode, create a new result store
            store = result_list.create()
        else:
            result_list_items = result_list.list()
            if len(result_list_items) > 0 and self.interactive:
                # more than one historical results, push to ui and let user decide whether to resume
                await self.event_bus.publish(ResultListEvent(items=result_list_items))
                e = await self.event_bus.wait_for(WorkflowStartEvent)
                if e is not None and e.path is not None:
                    store = result_list.path(e.path)
                else:
                    store = result_list.create()
            else:
                store = result_list.create()

        # setup file logging in the store
        setup_file_logging(store.log_path())

        try:
            specs = await self.gather_tasks()
            self.event_bus.publish_defer(SpecsEvent(items=specs))
            asyncio.create_task(self._push_metrics(store))

            completed = await store.completed()
            for result in completed.values():
                self.event_bus.publish_defer(SessionCompletedEvent(
                    spec=result.spec(),
                    session_id=result.session_id,
                    result=result,
                    metric=self.metric(result)
                ))

            self._specs_queue = asyncio.Queue()
            [self._specs_queue.put_nowait(spec) for spec in specs if spec.run_key() not in completed]

            worker_tasks = [
                asyncio.create_task(self._worker(store=store))
                for _ in range(self.concurrency)
            ]
            try:
                await asyncio.gather(*worker_tasks)
            except Exception:
                for task in worker_tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*worker_tasks, return_exceptions=True)
                raise
        finally:
            self.event_bus.publish_defer(WorkflowCompletedEvent())
            await store.save(force=True)
            self.logger.info('evaluation results:\n%s', await store.metrics(self.metric))
            if self.token_counter is not None:
                self.logger.info(self.token_counter.summary())

    async def _worker(self, *, store: ResultStore):
        while not self._specs_queue.empty():
            spec = await self._specs_queue.get()
            await self._run_spec(
                store=store,
                spec=spec
            )
            self._specs_queue.task_done()

    async def _run_spec(self, *, store: ResultStore, spec: RunSpec):
        session_id: Optional[int] = None

        async def _handle_started(started_session_id: int):
            nonlocal session_id
            session_id = started_session_id
            self.event_bus.publish_defer(SessionStartedEvent(
                spec=spec,
                session_id=session_id
            ))

        result = await (SampleWorkflow(
            controller=self.controller,
            models=self._model_clients.get(spec.model),
            spec=spec,
            controller_renew=self.controller_renew,
            session_started_callback=_handle_started,
            start_sample=self.start_sample
        )())

        self.event_bus.publish_defer(SessionCompletedEvent(
            spec=spec,
            session_id=session_id,
            result=result,
            metric=self.metric(result)
        ))

        await store.record(result)
        asyncio.create_task(self._push_metrics(store))

    async def _push_metrics(self, store: ResultStore):
        if self.start_sample:
            return  # no metrics in start-sample mode

        try:
            await self.event_bus.publish(MetricsEvent(
                metric_type=self.metric.type,
                items=await store.metrics(self.metric)
            ))
        except Exception:
            self.logger.exception('failed to push metrics')

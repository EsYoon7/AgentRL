"""
task_manager_mm.py  -> trainer/src/agentrl/trainer/components/

The base DistributedTaskManager._worker does:  item.update(result); buffer.add(item)
i.e. it assumes task_fn returns ONE dict. Our multimodal task returns a LIST of
per-turn items. This module provides a tiny adapter so a list result is added as
multiple buffer entries, without editing the upstream manager.

Two options are provided:

1. wrap_list_task_fn(task_fn): wrap your task_fn so that, instead of returning a
   list, it... still returns a list -- the real handling must live in the worker
   loop. So we instead subclass the managers to handle list results. Prefer (2).

2. DistributedTaskManagerMM / LocalTaskManagerMM: subclasses whose worker loop
   adds each element of a list result to the buffer. This is the clean path.

IMPORTANT for GRPO grouping: Buffer(strict_group) releases a group only once it
has `group_size` items. With per-turn items, a "group" is no longer n
trajectories but n*<turns> items, and turns-per-trajectory varies. So set
buffer_group_size=1 for the MM managers and do grouping via `group_id` inside
compute_advantage (which groups by group_id regardless of buffer batching).
Alternatively keep strict grouping but make group completion trajectory-based;
that needs a group-by-trajectory buffer, which is more invasive. We take the
simpler route: buffer_group_size=1 here.
"""

from __future__ import annotations

import asyncio
import traceback

import ray
from ray.util.queue import Queue as RayQueue

from agentrl.trainer.components.task_manager import (
    LocalTaskManager, DistributedTaskManager,
)


def _normalize_results(item, result):
    """Return a list of buffer-ready dicts from a task_fn result.

    - None        -> []           (dropped)
    - dict        -> [merged dict] (legacy single-item behavior)
    - list[dict]  -> [merged dicts] (per-turn items; each merged with `item`)
    """
    if result is None:
        return []
    if isinstance(result, dict):
        merged = dict(item)
        merged.update(result)
        return [merged]
    if isinstance(result, list):
        out = []
        for r in result:
            merged = dict(item)
            merged.update(r)
            out.append(merged)
        return out
    raise TypeError(f"task_fn must return None | dict | list[dict], got {type(result)}")


class LocalTaskManagerMM(LocalTaskManager):
    async def _worker(self):
        while not self._stop_signal.is_set():
            item = await self.queue.get()
            try:
                result = await self.task_fn(item)
            except Exception:
                traceback.print_exc()
                self._pending_count -= 1
                continue
            entries = _normalize_results(item, result)
            if not entries:
                self._pending_count -= 1
                continue
            for entry in entries:
                await self.buffer.add(entry)
            # pending counts trajectories submitted, not turns; decrement once.
            self._pending_count -= 1


@ray.remote
def distributed_worker_mm(queue, buffer, task_fn):
    async def process():
        while True:
            try:
                item = await queue.get_async()
                try:
                    result = await task_fn(item)
                except Exception:
                    traceback.print_exc()
                    continue
                entries = _normalize_results(item, result)
                for entry in entries:
                    await buffer.add.remote(entry)
            except Exception:
                traceback.print_exc()
                continue

    asyncio.run(process())


class DistributedTaskManagerMM(DistributedTaskManager):
    def start(self):
        if self._started:
            return
        for _ in range(self.num_workers):
            worker = distributed_worker_mm.remote(self.queue, self.buffer, self.task_fn)
            self.workers.append(worker)
        self._started = True
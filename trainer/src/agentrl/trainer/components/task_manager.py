import asyncio
import traceback
from asyncio import Queue, Task
from functools import partial
from typing import Callable, Any, Optional

import ray
from ray.util.queue import Queue as RayQueue

from ..agentic.loops import retry_openai_chat_agent_loop
from ..agentic.tasks import openai_chat_start, openai_chat_obs, openai_chat_end
from ..components.buffer import Buffer


class AbstractTaskManager:
    async def put(self, item: dict) -> None: ...
    async def get(self, *args, **kwargs) -> Any: ...
    async def get_all(self) -> Any: ...

    def start(self) -> None: ...
    async def stop(self) -> None: ...


class LocalTaskManager(AbstractTaskManager):
    def __init__(
        self,
        task_fn: Callable[[dict], Task[dict | None]],
        max_queue_size: int = 100,
        max_buffer_size: int = 100,
        buffer_group_size: int = 1,
        num_workers: int = 1,
        event_loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        self.task_fn = task_fn
        self.queue = Queue(maxsize=max_queue_size)
        self.buffer = Buffer(max_buffer_size, group_size=buffer_group_size)
        self.workers = []
        self.num_workers = num_workers
        self.event_loop = event_loop
        self._started = False
        self._stop_signal = asyncio.Event()
        self._pending_count = 0

    async def _worker(self):
        while not self._stop_signal.is_set():
            item = await self.queue.get()
            try:
                result = await self.task_fn(item)
            except Exception:
                traceback.print_exc()
                self._pending_count -= 1
                continue
            if result is None:
                self._pending_count -= 1
                continue
            item.update(result)
            await self.buffer.add(item)
            self._pending_count -= 1

    def start(self):
        if self._started:
            return
        self._stop_signal.clear()
        for _ in range(self.num_workers):
            if self.event_loop:
                worker = self.event_loop.create_task(self._worker())
            else:
                worker = asyncio.create_task(self._worker())
            self.workers.append(worker)
        self._started = True

    async def put(self, item: dict) -> None:
        self._pending_count += 1
        await self.queue.put(item)

    def put_nowait(self, item: dict) -> None:
        self._pending_count += 1
        self.queue.put_nowait(item)

    async def get(self, minimum: int, multiple: int = 1) -> list[dict]:
        results = await self.buffer.get(minimum, multiple)
        self._pending_count -= len(results)
        return results

    async def stop(self):
        self._stop_signal.set()
        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers = []
        self._started = False

    async def get_all(self) -> list[dict]:
        results = await self.buffer.get(self._pending_count, 1)
        return results

    @property
    def queue_size(self) -> int:
        return self.queue.qsize()

    @property
    def queue_maxsize(self) -> int:
        return self.queue.maxsize


@ray.remote
def distributed_worker(queue: RayQueue, buffer, task_fn):
    async def process():
        while True:
            try:
                item = await queue.get_async()
                try:
                    result = await task_fn(item)
                except Exception:
                    traceback.print_exc()
                    continue
                if result is None:
                    continue
                item.update(result)
                await buffer.add.remote(item)
            except:
                traceback.print_exc()
                continue

    asyncio.run(process())


class DistributedTaskManager(AbstractTaskManager):
    def __init__(
        self,
        task_fn: Callable[[dict], Task[dict | None]],
        max_queue_size: int = 100,
        max_buffer_size: int = 100,
        buffer_group_size: int = 1,
        num_workers: int = 1,
        event_loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        self.task_fn = task_fn
        self.num_workers = num_workers
        self.event_loop = event_loop
        self._started = False

        self.queue = RayQueue(maxsize=max_queue_size, actor_options={"max_concurrency": 16384})
        self.buffer = ray.remote(Buffer).remote(max_buffer_size, buffer_group_size)
        self.workers = []
        self._pending_count = 0

    def start(self):
        if self._started:
            return

        for i in range(self.num_workers):
            worker = distributed_worker.remote(self.queue, self.buffer, self.task_fn)
            self.workers.append(worker)
        self._started = True

    async def put(self, item: dict) -> None:
        self._pending_count += 1
        self.queue.put(item)

    def put_nowait(self, item: dict) -> None:
        self._pending_count += 1
        self.queue.put(item)

    async def get(self, minimum: int, multiple: int = 1) -> list[dict]:
        results = await self.buffer.get.remote(minimum, multiple)
        self._pending_count -= len(results)
        return results

    async def get_all(self) -> list[dict]:
        results = await self.buffer.get.remote(self._pending_count, 1)
        self._pending_count = 0
        return results

    async def stop(self):
        self.workers = []
        self._started = False

    @property
    def queue_size(self) -> int:
        return self.queue.qsize()

    @property
    def queue_maxsize(self) -> int:
        return self.queue.maxsize


def openai_chat_task(item, config, tokenizer, gen_fn):
    # choose function set
    url = config["base_url"]
    loop_fn = partial(
        retry_openai_chat_agent_loop,
        incomplete_punishment=config.get("incomplete_punishment", 0),
        tool_call_parser=config.get("tool_call_parser", "qwen25"),
        max_retries=config.get("max_retries", 10),
    )
    start_fn = partial(openai_chat_start, url=url)
    obs_fn = partial(openai_chat_obs, url=url)
    end_fn = partial(openai_chat_end, url=url)

    item = {k: v for k, v in item.items() if k in config["loop_start_args"]}

    return asyncio.create_task(
        loop_fn(
            start_args=item,
            start_fn=start_fn,
            gen_fn=gen_fn,
            obs_fn=obs_fn,
            end_fn=end_fn,
            max_turns=config["max_turns"],
            max_length=config["max_total_len"] - 10,
            tokenizer=tokenizer,
        )
    )

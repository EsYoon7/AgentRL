from copy import copy
from functools import partial
from typing import TypeVar, Generic, Any, Callable

import ray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from .abstract import BaseWorker, AsyncBaseWorker
from ..utils import is_async_cls

T = TypeVar("T", bound=BaseWorker)


class CollectiveHandle(Generic[T]):
    workers: list = []
    world_size: int = 0
    dispatch_target: list | Any  = []

    def __init__(self, cls: T, placement, num_gpus, *args, **kwargs):
        self.workers = []
        self.world_size = len(placement.bundle_specs)
        concurrency = 1
        if is_async_cls(cls):
            concurrency = 16384 # ray default 1000 is not enough
            assert issubclass(cls, AsyncBaseWorker), "async workers should base on AsyncBaseWorker"

        for i in range(self.world_size):
            worker = ray.remote(
                num_cpus=0.01,
                num_gpus=num_gpus,
                runtime_env={
                    "env_vars": {
                        "WORLD_SIZE": str(self.world_size),
                        "RANK": str(i),
                    }
                },
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=placement,
                    placement_group_bundle_index=i,
                ),
            )(cls).options(max_concurrency=concurrency).remote(*args, **kwargs)
            self.workers.append(worker)

        self.dispatch_target = self.workers

        addr, port = ray.get(self.workers[0].get_addr_and_port.remote())
        self.init_distributed(addr, port)

    def __getattribute__(self, item):
        if item.startswith("_") or item in CollectiveHandle.__dict__:
            return super().__getattribute__(item)
        def wrapped_func(*args, **kwargs):
            if isinstance(self.dispatch_target, list):
                results = []
                for worker in self.dispatch_target:
                    results.append(getattr(worker, item).remote(*args, **kwargs))
            else:
                results = getattr(self.dispatch_target, item).remote(*args, **kwargs)
            return results

        return wrapped_func

    def dispatch_rank0(self):
        return self.dispatch_rank(0)

    def dispatch_rank(self, rank: int | list[int]) -> T | "CollectiveHandle[T]":
        if rank < 0 or rank >= len(self.workers):
            raise IndexError(f"Rank {rank} out of range for {len(self.workers)} workers.")
        new = copy(self)
        if isinstance(rank, list):
            new.dispatch_target = [self.workers[r] for r in rank]
        else:
            new.dispatch_target = self.workers[rank]
        return new

def spawn(cls: type[T], placement: PlacementGroup, num_gpus=1) -> type[T] | Callable[[Any], CollectiveHandle[T]]:
    return partial(CollectiveHandle, cls, placement, num_gpus)

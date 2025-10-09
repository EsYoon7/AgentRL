import functools
import gc
import inspect
import uuid
from collections import defaultdict
from copy import deepcopy
from typing import Any, Iterable, Generator

import ray
import torch
import torch.distributed as dist


def append_to_dict(data: dict, new_data: dict):
    """Append values from new_data to lists in data.

    For each key in new_data, this function appends the corresponding value to a list
    stored under the same key in data. If the key doesn't exist in data, a new list is created.

    Args:
        data (Dict): The target dictionary containing lists as values.
        new_data (Dict): The source dictionary with values to append.

    Returns:
        None: The function modifies data in-place.
    """
    for key, val in new_data.items():
        if key not in data:
            data[key] = []
        data[key].append(val)


def reduce_dict(data: dict[Any, list | Any]) -> dict:
    new_data = {}
    for key, val in data.items():
        if isinstance(val, list):
            if not val:
                continue
            val = sum(val) / len(val)
        if isinstance(val, torch.Tensor):
            val = val.item()
        new_data[key] = val
    return new_data


def append_with_prefix(data: dict, prefix: str, new_data: dict):
    for key, val in new_data.items():
        data[f"{prefix}{key}"] = val


def recursive_apply(func):
    def wrapped(item, *args, **kwargs):
        item = func(item, *args, **kwargs)
        if isinstance(item, dict):
            return {k: wrapped(v, *args, **kwargs) for k, v in item.items()}
        if isinstance(item, list):
            return [wrapped(v, *args, **kwargs) for v in item]
        return item
    return wrapped


@recursive_apply
def to_device(item, device="cuda"):
    if isinstance(item, ray.ObjectRef):
        item = ray.get(item)
    if isinstance(item, torch.Tensor):
        return item.to(device, non_blocking=True)
    return item


@recursive_apply
def to_plasma(item):
    if isinstance(item, torch.Tensor):
        return ray.put(item.cpu())
    return item


def is_async_cls(cls):
    for name, member in inspect.getmembers(cls, predicate=inspect.isfunction):
        if inspect.iscoroutinefunction(member):
            return True
    return False


# Copied from Sglang
# Copy from pytorch and OpenRLHF to allow creating multiple main groups.
# https://github.com/pytorch/pytorch/blob/main/torch/distributed/distributed_c10d.py
# https://github.com/OpenRLHF/OpenRLHF/blob/main/openrlhf/utils/distributed_util.py
def init_custom_process_group(
    backend=None,
    init_method=None,
    timeout=None,
    world_size=-1,
    rank=-1,
    store=None,
    group_name=None,
    pg_options=None,
    device_id=None,
):
    from torch.distributed.distributed_c10d import (
        Backend,
        PrefixStore,
        _new_process_group_helper,
        _world,
        default_pg_timeout,
        rendezvous,
    )

    assert (store is None) or (
        init_method is None
    ), "Cannot specify both init_method and store."

    if store is not None:
        assert world_size > 0, "world_size must be positive if using store"
        assert rank >= 0, "rank must be non-negative if using store"
    elif init_method is None:
        init_method = "env://"

    if backend:
        backend = Backend(backend)
    else:
        backend = Backend("undefined")

    if timeout is None:
        timeout = default_pg_timeout

    # backward compatible API
    if store is None:
        rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
        store, rank, world_size = next(rendezvous_iterator)
        store.set_timeout(timeout)

        # Use a PrefixStore to avoid accidental overrides of keys used by
        # different systems (e.g. RPC) in case the store is multi-tenant.
        store = PrefixStore(group_name, store)

    # NOTE: The pg_options parameter was renamed into backend_options in PyTorch 2.6.0
    # https://github.com/pytorch/pytorch/commit/a0c7029a75628cd5fa8df83c0de0ea98ee7fd844
    # We need to determine the appropriate parameter name based on PyTorch version
    pg_options_param_name = (
        "backend_options" if str(torch.__version__) >= "2.6" else "pg_options"
    )
    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        backend,
        store,
        group_name=group_name,
        **{pg_options_param_name: pg_options},
        timeout=timeout,
        device_id=device_id,
    )

    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}

    return pg


def pretty_print_metrics(metrics: dict):
    def tree():
        return defaultdict(tree)

    def insert(d, keys, value):
        if len(keys) == 1:
            d[keys[0]] = value
        else:
            insert(d[keys[0]], keys[1:], value)

    def print_tree(d, indent=0):
        for key in sorted(d.keys()):
            if isinstance(d[key], dict):
                print("  " * indent + str(key) + "/")
                print_tree(d[key], indent + 1)
            else:
                if isinstance(d[key], float):
                    val_str = f"{d[key]:.6f}"
                else:
                    val_str = str(d[key])
                print("  " * indent + f"{key}: {val_str}")

    root = tree()
    for k, v in metrics.items():
        parts = k.split("/")
        insert(root, parts, v)

    print_tree(root)


def repeat(it: Iterable, n: int) -> Generator[dict, Any, None]:
    for item in it:
        item["group_id"] = str(uuid.uuid4())
        for _ in range(n):
            item = deepcopy(item)
            item["uid"] = str(uuid.uuid4())
            yield item


def interleave(*its: Iterable) -> Generator:
    its = [iter(it) for it in its]
    while its:
        for it in its:
            yield next(it)


def _get_current_mem_info(unit: str = "GB", precision: int = 2):
    """Get current memory usage."""
    assert unit in ["GB", "MB", "KB"]
    divisor = 1024**3 if unit == "GB" else 1024**2 if unit == "MB" else 1024
    mem_allocated = torch.cuda.memory_allocated()
    mem_reserved = torch.cuda.memory_reserved()
    # use get_torch_device().mem_get_info to profile device memory
    # since vllm's sleep mode works below pytorch
    # see https://github.com/vllm-project/vllm/pull/11743#issuecomment-2754338119
    mem_free, mem_total = torch.cuda.mem_get_info()
    mem_used = mem_total - mem_free
    mem_allocated = f"{mem_allocated / divisor:.{precision}f}"
    mem_reserved = f"{mem_reserved / divisor:.{precision}f}"
    mem_used = f"{mem_used / divisor:.{precision}f}"
    mem_total = f"{mem_total / divisor:.{precision}f}"
    return mem_allocated, mem_reserved, mem_used, mem_total


def log_gpu_memory_usage(head: str, rank: int = 0):
    if (not dist.is_initialized()) or (rank is None) or (dist.get_rank() == rank):
        mem_allocated, mem_reserved, mem_used, mem_total = _get_current_mem_info()
        message = f"{head}, memory allocated (GB): {mem_allocated}, memory reserved (GB): {mem_reserved}, device memory used/total (GB): {mem_used}/{mem_total}"
        print(message)


def clean_cuda(func):
    """Decorator to clean CUDA memory after function execution."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        finally:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
    return wrapper

import os
import socket
from pathlib import Path
from typing import Callable, Iterator

import ray
import torch
from torch import Tensor
from torch.profiler import profile, ProfilerActivity

LossFnType = Callable[[dict, dict], tuple[torch.Tensor, dict]]


class BaseWorker:
    plugins: dict = {}
    profiler: profile | None = None

    @property
    def rank(self) -> int:
        return int(os.environ["RANK"])

    @property
    def world_size(self) -> int:
        return int(os.environ["WORLD_SIZE"])

    def init_distributed(self, addr, port): ...

    def register_plugin(self, name: str, plugin_cls, *args, **kwargs):
        plugin = plugin_cls(self, *args, **kwargs)
        self.plugins[name] = plugin

    def call_plugin(self, name: str, method: str, *args, **kwargs):
        plugin = self.plugins.get(name)
        if plugin is None:
            raise AttributeError(f"Plugin '{name}' not found.")
        return getattr(plugin, method)(*args, **kwargs)

    def get_addr_and_port(self) -> tuple[str, int]:
        addr = ray._private.services.get_node_ip_address()
        with socket.socket() as sock:
            sock.bind(("", 0))
            port = sock.getsockname()[1]
        return addr, port

    def start_profiler(self, *args, **kwargs):
        self.profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
            with_flops=True,
            *args, **kwargs,
        )
        self.profiler.start()

    def stop_profiler(self, path: str, tag: str = ""):
        self.profiler.stop()
        self.profiler.export_chrome_trace(str(Path(path) / f"trace_{self.__class__.__name__}_{tag}_r{self.rank}.json"))
        print(f"Profiler trace saved to {path}")
        self.profiler = None

    def no_op(self):
        """Acting as a barrier for ray ops"""
        pass


class AsyncBaseWorker(BaseWorker):
    async def async_call_plugin(self, name: str, method: str, *args, **kwargs):
        plugin = self.plugins.get(name)
        if plugin is None:
            raise AttributeError(f"Plugin '{name}' not found.")
        return await getattr(plugin, method)(*args, **kwargs)


class AbstractTrainWorker(BaseWorker):
    def build_model(self, path: str): ...
    def build_optimizer(self): ...
    def build_checkpoint_manager(self): ...

    def param_generator(self) -> Iterator[tuple[str, Tensor]]: ...

    def forward_backward(self, mini_batch: list[dict], loss_fn: LossFnType, forward_only: bool = False, unpack: bool = False) -> dict: ...
    def step(self) -> torch.Tensor: ...

    def save_checkpoint(self, path: str): ...
    def load_checkpoint(self, path: str): ...


class AbstractAsyncRolloutWorker(AsyncBaseWorker):
    def build_engine(self, path: str): ...

    async def update_params(self, params: list[tuple[str, torch.Tensor]]): ...

    async def generate(self, *args, **kwargs) -> tuple[str, list]: ...

    async def async_acquire_writer_lock(self): ...
    async def async_release_writer_lock(self): ...

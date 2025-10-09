import asyncio
import os
import random
import time
from copy import deepcopy

import sglang as sgl
import torch
from sglang.srt.aio_rwlock import RWLock
from sglang.srt.managers.io_struct import UpdateWeightsFromTensorReqInput
from sglang.srt.model_executor.model_runner import LocalSerializedTensor
from sglang.srt.utils import MultiprocessingSerializer

from .abstract import AbstractAsyncRolloutWorker
from ..utils import to_device
from ..utils.aio_rwlock import WriteEnforceRWLock
from ..utils.sglang_patch import apply_patch

apply_patch()


class AsyncSglangWorker(AbstractAsyncRolloutWorker):
    def __init__(self, config):
        super().__init__()
        # configs
        self.config = config
        self.tp_size = config.get("tp_size", 1)
        assert torch.cuda.device_count() == self.tp_size, f"{torch.cuda.device_count()} != {self.tp_size}"
        self.base_sampling_params: dict = config.get("sampling_params", {})
        self.base_sampling_params.update({
            "skip_special_tokens": False,
        })
        self.rw_lock = WriteEnforceRWLock() if self.config.get("use_force_cancel", False) else RWLock()
        self.event_loop = asyncio.get_event_loop()

        self.engine = None

    def build_engine(self, model_path):
        rng = random.Random(time.time())
        os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
        self.engine = sgl.Engine(
            model_path=model_path,
            port=40000 + rng.randint(0, 1000),
            dtype=self.config.get("dtype", "bfloat16"),
            tp_size=self.tp_size,
            **self.config.get("server_args", {}),
        )
        self.engine.tokenizer_manager.auto_create_handle_loop()

    async def generate(self, **kwargs):
        sampling_params = deepcopy(self.base_sampling_params)
        sampling_params.update(kwargs.get("sampling_params", {}))
        kwargs["sampling_params"] = sampling_params
        while True:
            try:
                async with self.rw_lock.reader_lock:
                    ret = await self.engine.async_generate(
                        **to_device(kwargs),
                        return_logprob=True,
                    )
                    break
            except asyncio.CancelledError:
                print("gen chat cancelled")

        log_probs = ret["meta_info"]["output_token_logprobs"]

        text = ret["text"]
        return text, log_probs

    async def update_params(self, tensor_list):
        dup_list = [
            (k, [v.to(f"cuda:{i}", non_blocking=True) for i in range(self.tp_size)])
            for k, v in tensor_list
        ]
        serialized_tensor_list = [
            (k, LocalSerializedTensor(values=[MultiprocessingSerializer.serialize(v) for v in vs]))
            for k, vs in dup_list
        ]
        await self.engine.tokenizer_manager.update_weights_from_tensor(
            UpdateWeightsFromTensorReqInput(
                serialized_named_tensors=[
                    MultiprocessingSerializer.serialize(serialized_tensor_list)
                    for _ in range(self.tp_size)
                ],
                load_format=None,
                flush_cache=True,
            )
        )
        del dup_list, serialized_tensor_list
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

    async def async_acquire_writer_lock(self):
        if self.config.get("forget_lock"):
            return
        await self.rw_lock.acquire_writer()

        while True:
            print("flushing cache...")
            result = await self.engine.tokenizer_manager.flush_cache()
            print(f"flush cache done {result.success=}")
            if result.success:
                break
            else:
                print(f"flush cache failed, retrying...")
                await asyncio.sleep(0.5)


    async def async_release_writer_lock(self):
        if self.config.get("forget_lock"):
            return
        await self.rw_lock.release_writer()

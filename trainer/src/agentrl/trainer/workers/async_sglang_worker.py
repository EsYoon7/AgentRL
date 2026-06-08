import asyncio
import os
import random
import time
from copy import deepcopy

import sglang as sgl
import torch
from sglang.srt.utils.aio_rwlock import RWLock
from sglang.srt.managers.io_struct import (
    InitWeightsUpdateGroupReqInput,
    UpdateWeightsFromDistributedReqInput,
)

from .abstract import AbstractAsyncRolloutWorker
from ..utils import to_device
from ..utils.aio_rwlock import WriteEnforceRWLock
from ..utils.sglang_patch import apply_patch

apply_patch()


class AsyncSglangWorker(AbstractAsyncRolloutWorker):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tp_size = config.get("tp_size", 1)
        assert torch.cuda.device_count() == self.tp_size, f"{torch.cuda.device_count()} != {self.tp_size}"
        self.base_sampling_params: dict = config.get("sampling_params", {})
        self.base_sampling_params.update({"skip_special_tokens": False})
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
        if isinstance(ret, list):
            ret = ret[0]
        log_probs = ret["meta_info"]["output_token_logprobs"]
        text = ret["text"]
        return text, log_probs

    # ---- distributed weight sync (NCCL broadcast; IPC 없음 → ptrace 무관, TP 무관) ----
    async def init_weight_update_group(
        self, master_address, master_port, rank_offset, world_size,
        group_name="weight_update_group",
    ):
        """weight broadcast용 NCCL group에 SGLang을 receiver로 합류시킨다.
        SGLang의 각 tp rank는 rank = rank_offset + tp_rank를 갖고,
        trainer rank0(=0)이 broadcast source다."""
        return await self.engine.tokenizer_manager.init_weights_update_group(
            InitWeightsUpdateGroupReqInput(
                master_address=master_address, master_port=master_port,
                rank_offset=rank_offset, world_size=world_size,
                group_name=group_name, backend="nccl",
            ), None,
        )

    async def update_params(self, names, dtypes, shapes, group_name="weight_update_group"):
        """NCCL broadcast(src=trainer rank0)로 가중치를 엔진에 주입한다.
        주입 동안 generation을 멈추고(writer lock) KV 캐시를 flush한다."""
        await self.async_acquire_writer_lock()
        try:
            await self.engine.tokenizer_manager.update_weights_from_distributed(
                UpdateWeightsFromDistributedReqInput(
                    names=names, dtypes=dtypes, shapes=shapes,
                    group_name=group_name, flush_cache=True,
                ), None,
            )
        finally:
            await self.async_release_writer_lock()

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
            print("flush cache failed, retrying...")
            await asyncio.sleep(0.5)

    async def async_release_writer_lock(self):
        if self.config.get("forget_lock"):
            return
        await self.rw_lock.release_writer()
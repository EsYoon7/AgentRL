"""
async_sglang_worker_mm.py  -> goes in trainer/src/agentrl/trainer/workers/

Thin subclass of AsyncSglangWorker that adds a generation method preserving
output token ids (and aligned rollout logprobs). The base worker is untouched,
so apply_patch(), update_params() (NCCL weight sync), and the RWLock all carry
over unchanged.

NOTE on naming: the repo already has utils/sglang_patch.py which monkey-patches
the sglang *library* (flush_cache, weight update, cancellation). That is
unrelated to this file. This file extends the *worker*.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy

from agentrl.trainer.workers.async_sglang_worker import AsyncSglangWorker
from agentrl.trainer.utils import to_device


def extract_generation(ret: dict):
    """(text, output_ids, rollout_logprobs) from a native sglang result.

    Confirmed for this repo's sglang version via probe:
      meta_info['output_token_logprobs'] is a list of (logprob, token_id, ...)
    We extract ids from index [1] and logprobs from index [0]. If a future
    version also exposes top-level 'output_ids', prefer it.
    """
    if isinstance(ret, list):
        ret = ret[0]

    text = ret.get("text", "")
    meta = ret["meta_info"]
    otlp = meta.get("output_token_logprobs")

    if "output_ids" in ret and ret["output_ids"] is not None:
        output_ids = list(ret["output_ids"])
        rollout_logprobs = (
            [float(x[0]) for x in otlp] if otlp is not None else []
        )
        if rollout_logprobs and len(rollout_logprobs) != len(output_ids):
            output_ids = output_ids[-len(rollout_logprobs):]
    elif otlp is not None:
        output_ids = [int(x[1]) for x in otlp]
        rollout_logprobs = [float(x[0]) for x in otlp]
    else:
        raise RuntimeError(
            "No output ids in sglang result; inspect ret.keys() / "
            "ret['meta_info'].keys() for this sglang version."
        )

    return text, output_ids, rollout_logprobs


class AsyncSglangWorkerMM(AsyncSglangWorker):
    """Adds generate_with_ids; everything else inherited unchanged."""

    async def generate_with_ids(self, **kwargs):
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

        return extract_generation(ret)
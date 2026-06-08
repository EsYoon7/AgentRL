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
    """(text, output_ids, rollout_logprobs, prompt_ids) from a native sglang result.

    Method B: SGLang tokenizes the prompt (text + image_data) itself, so we read
    BOTH prompt ids and output ids back:
      meta_info['input_token_logprobs']  -> prompt ids   (logprob, token_id, ...)
      meta_info['output_token_logprobs'] -> output ids   (logprob, token_id, ...)
    Requires async_generate(..., return_logprob=True, logprob_start_len=0) so the
    full prompt's input_token_logprobs are returned. prompt_ids is None if the
    field is absent (then method B isn't available on this sglang version).
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

    # prompt ids (method B): the prompt SGLang actually tokenized, incl. expanded
    # image placeholders. Use these as the training prompt ids (tokenize-once).
    itlp = meta.get("input_token_logprobs")
    prompt_ids = None
    if itlp is not None:
        try:
            prompt_ids = [int(x[1]) for x in itlp]
        except Exception:
            prompt_ids = None

    return text, output_ids, rollout_logprobs, prompt_ids


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
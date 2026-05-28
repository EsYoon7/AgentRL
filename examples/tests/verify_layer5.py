"""
LAYER 5-6 verification (GPU + model required). Run on a rollout-capable node.

Layer 5: drive ONE real generation through AsyncSglangWorkerMM and confirm the
         emitted item matches what FSDPWorker.collate / ppo_loss expect.
Layer 6: THE MONEY CHECK. Forward the stored (prompt+output) ids through the
         SAME model and compare the recomputed logprobs to the stored
         rollout_log_prob. Small diff = backend precision (expected). Large diff
         = a real bug (positions / placeholders / retokenization).

These are written to run pieces independently so you can bisect. They avoid Ray;
they talk to the engine + an HF model directly. Adapt MODEL_PATH and the model
constants (placeholder id, vision pad ids) to your model.
"""

import asyncio
import numpy as np
import torch


# --- inlined from async_sglang_worker_mm.py to avoid import-path issues ---
def extract_generation(ret: dict):
    """(text, output_ids, rollout_logprobs) from a native sglang result."""
    if isinstance(ret, list):
        ret = ret[0]
    text = ret.get("text", "")
    meta = ret["meta_info"]
    otlp = meta.get("output_token_logprobs")
    if "output_ids" in ret and ret["output_ids"] is not None:
        output_ids = list(ret["output_ids"])
        rollout_logprobs = [float(x[0]) for x in otlp] if otlp is not None else []
        if rollout_logprobs and len(rollout_logprobs) != len(output_ids):
            output_ids = output_ids[-len(rollout_logprobs):]
    elif otlp is not None:
        output_ids = [int(x[1]) for x in otlp]
        rollout_logprobs = [float(x[0]) for x in otlp]
    else:
        raise RuntimeError(
            "No output ids in sglang result; inspect ret.keys()/meta_info.keys()."
        )
    return text, output_ids, rollout_logprobs


# ---- fill these for your model -------------------------------------------
MODEL_PATH = "/mnt/home/justiwag/esyoon/models/Qwen3.5-9B"
IMAGE_PLACEHOLDER_ID = 248056          # int
IMAGE_PAD_OPEN_IDS = [248053]              # e.g. [vision_start]
IMAGE_PAD_CLOSE_IDS = [248054]             # e.g. [vision_end]
COLLAPSE_MARKER_IDS = [1919, 34993, 682, 978, 27350, 13]             # exact ids used in training
ASSISTANT_HEADER_IDS = [248045, 74455, 198]
ASSISTANT_EOT_IDS = [248053, ..., 248054]

# ===========================================================================
# LAYER 5 — one real generation + item-shape validation
# ===========================================================================
async def layer5_one_turn():
    import sglang as sgl
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(MODEL_PATH)
    engine = sgl.Engine(model_path=MODEL_PATH, dtype="bfloat16", tp_size=1)
    engine.tokenizer_manager.auto_create_handle_loop()

    # build a trivial text-only prompt as input_ids (no image first, to isolate)
    prompt_text = "What is 2 + 2? Answer in one word."
    prompt_ids = proc.tokenizer(prompt_text, add_special_tokens=True)["input_ids"]

    ret = await engine.async_generate(
        input_ids=prompt_ids,
        sampling_params={"max_new_tokens": 8, "temperature": 0.0},
        return_logprob=True,
    )
    text, output_ids, rollout_lp = extract_generation(ret)

    print("=== LAYER 5 ===")
    print("prompt_ids len :", len(prompt_ids))
    print("output_ids     :", output_ids)
    print("rollout_lp len :", len(rollout_lp), "(must == len(output_ids))")
    assert len(rollout_lp) == len(output_ids), "logprob/id length mismatch!"

    full = prompt_ids + output_ids
    loss_mask = [0] * len(prompt_ids) + [1] * len(output_ids)
    print("full len       :", len(full), "loss tokens:", sum(loss_mask))

    # item-shape checks mirroring FSDPWorker.collate expectations
    input_ids = torch.tensor(full).unsqueeze(0)
    assert input_ids.shape[1] == len(full)
    assert loss_mask[0] == 0, "first token must have no loss (ppo_loss asserts this)"
    print("item-shape checks OK")
    return proc, full, prompt_ids, output_ids, rollout_lp


# ===========================================================================
# LAYER 6 — rollout vs train logprob alignment (the decisive check)
# ===========================================================================
@torch.no_grad()
def layer6_logprob_match(model_path, full_ids, prompt_len, rollout_lp,
                         position_ids=None, multi_modal_inputs=None,
                         temperature=1.0):
    """Forward full_ids through an HF model; compare to stored rollout_lp.

    Replicates the ppo_loss logprob computation:
      logits/temperature -> logprobs_from_logits(labels=roll(input_ids,-1))
      -> roll(+1). We compare on the OUTPUT region only.
    """
    from transformers import AutoModelForImageTextToText, AutoModelForCausalLM

    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model = model.cuda().eval()

    input_ids = torch.tensor(full_ids).unsqueeze(0).cuda()
    kwargs = {"input_ids": input_ids}
    if position_ids is not None:
        kwargs["position_ids"] = torch.as_tensor(position_ids).cuda()
    if multi_modal_inputs:
        for k, v in multi_modal_inputs.items():
            kwargs[k] = v.cuda()

    out = model(**kwargs)
    logits = out.logits / temperature

    # logprob of each token given the previous (causal). Align like ppo_loss:
    labels = torch.roll(input_ids, shifts=-1, dims=1)
    logp_all = torch.log_softmax(logits.float(), dim=-1)
    tok_logp = logp_all.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    tok_logp = torch.roll(tok_logp, shifts=1, dims=1)  # [1, L]

    # output region = positions [prompt_len : L]; compare to rollout_lp
    recomputed = tok_logp[0, prompt_len:].cpu().numpy()
    stored = np.array(rollout_lp, dtype=np.float64)

    n = min(len(recomputed), len(stored))
    recomputed, stored = recomputed[:n], stored[:n]
    diff = np.abs(recomputed - stored)

    print("=== LAYER 6 ===")
    print(f"compared {n} output tokens")
    print(f"mean |diff| = {diff.mean():.5f}")
    print(f"max  |diff| = {diff.max():.5f}")
    print(f"per-token diff: {np.round(diff, 4)}")
    if diff.mean() < 0.01:
        print("VERDICT: aligned (residual = backend precision). GOOD.")
    elif diff.mean() < 0.1:
        print("VERDICT: borderline. Check temperature match and fp casting.")
    else:
        print("VERDICT: LARGE mismatch -> BUG. Suspect order:")
        print("  1) position_ids: are you passing the SAME mrope positions?")
        print("  2) image placeholder count vs pixel grid (multimodal)")
        print("  3) retokenization somewhere (ids not preserved)")
        print("  diagnostic: if diff is uniform -> precision; if it starts after")
        print("  an image block -> position/placeholder; if a single spike ->")
        print("  a boundary token.")
    return diff


def _save_layer5(path, full, prompt_ids, output_ids, rollout_lp):
    import json
    with open(path, "w") as f:
        json.dump({
            "full": full, "prompt_len": len(prompt_ids),
            "output_ids": output_ids, "rollout_lp": rollout_lp,
        }, f)
    print(f"[saved Layer 5 output to {path}]")


def _load_layer5(path):
    import json
    with open(path) as f:
        d = json.load(f)
    return d["full"], d["prompt_len"], d["output_ids"], d["rollout_lp"]


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "both"
    cache = "layer5_out.json"

    # Run Layer 5 and Layer 6 SEPARATELY to avoid holding the sglang engine and
    # the HF model in GPU memory at the same time (that caused the OOM).
    #   python verify_layer56.py layer5   # rollout only, saves ids, frees GPU
    #   python verify_layer56.py layer6   # loads ids, runs train-forward compare
    #   python verify_layer56.py both     # (only if the GPU has room for both)

    if mode in ("layer5", "both"):
        proc, full, prompt_ids, output_ids, rollout_lp = asyncio.run(layer5_one_turn())
        _save_layer5(cache, full, prompt_ids, output_ids, rollout_lp)
        if mode == "layer5":
            print("[Layer 5 done. Now run: python verify_layer56.py layer6]")
            sys.exit(0)

    if mode in ("layer6", "both"):
        full, prompt_len, output_ids, rollout_lp = _load_layer5(cache)
        layer6_logprob_match(
            MODEL_PATH, full, prompt_len, rollout_lp,
            position_ids=None, multi_modal_inputs=None, temperature=1.0,
        )
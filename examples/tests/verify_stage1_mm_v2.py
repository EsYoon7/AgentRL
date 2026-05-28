"""
STAGE 1 (v2) — PROCESSOR-DRIVEN multimodal verification with ONE dummy image.

Matches the new design: prompt is built by apply_chat_template + processor (NOT
hand-assembled). This tests the ACTUAL rollout path now.

  python verify_stage1_v2.py layer5img   # build via chat_template, rollout, save
  python verify_stage1_v2.py layer6img   # train-forward compare (logprob match)

What it checks:
  - processor builds input_ids with correct placeholders (no IndexError)
  - sglang accepts processor-built input_ids + image_data
  - rollout output_ids / logprob align
  - (layer6) image-sequence logprob matches between rollout and train forward,
    using position_ids from the model's get_rope_index (the real train path)
"""

import asyncio
import io
import json

import numpy as np
import torch
from PIL import Image

MODEL_PATH = "/mnt/home/justiwag/esyoon/models/Qwen3.5-9B"
CACHE = "stage1v2_out.json"
DUMMY_HW = (224, 224)


def _dummy_img():
    return Image.new("RGB", (DUMMY_HW[1], DUMMY_HW[0]), (120, 120, 120))


def _build_prompt(processor):
    """Build input_ids exactly like multimodal_task does: messages -> template."""
    img = _dummy_img()
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe the image in one word."},
        ],
    }]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    enc = processor(text=[text], images=[img], return_tensors="pt")
    return enc, img


# ===========================================================================
async def layer5img():
    import sglang as sgl
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    enc, img = _build_prompt(proc)
    prompt_ids = enc["input_ids"][0].tolist()
    print("prompt len:", len(prompt_ids),
          "image_grid_thw:", enc["image_grid_thw"].tolist())

    import base64
    buf = io.BytesIO(); img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    engine = sgl.Engine(model_path=MODEL_PATH, dtype="bfloat16", tp_size=1)
    engine.tokenizer_manager.auto_create_handle_loop()
    ret = await engine.async_generate(
        input_ids=prompt_ids,
        image_data=[img_b64],
        sampling_params={"max_new_tokens": 8, "temperature": 0.0},
        return_logprob=True,
    )
    if isinstance(ret, list):
        ret = ret[0]
    otlp = ret["meta_info"]["output_token_logprobs"]
    output_ids = [int(x[1]) for x in otlp]
    rollout_lp = [float(x[0]) for x in otlp]

    with open(CACHE, "w") as f:
        json.dump({
            "prompt_ids": prompt_ids,
            "output_ids": output_ids,
            "rollout_lp": rollout_lp,
            "image_grid_thw": enc["image_grid_thw"].tolist(),
        }, f)
    print("=== STAGE1 v2 layer5img ===")
    print("output_ids:", output_ids)
    print("rollout_lp len:", len(rollout_lp), "== output", len(output_ids))
    assert len(rollout_lp) == len(output_ids)
    print(f"[saved -> {CACHE}; now: python verify_stage1_v2.py layer6img]")


@torch.no_grad()
def layer6img():
    from transformers import AutoModelForImageTextToText, AutoProcessor
    with open(CACHE) as f:
        d = json.load(f)
    prompt_ids = d["prompt_ids"]; output_ids = d["output_ids"]
    rollout_lp = d["rollout_lp"]
    grid = torch.tensor(d["image_grid_thw"])

    proc = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    enc, img = _build_prompt(proc)
    pixel_values = enc["pixel_values"].cuda()
    image_grid_thw = enc["image_grid_thw"].cuda()

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()

    full = prompt_ids + output_ids
    input_ids = torch.tensor(full).unsqueeze(0).cuda()

    # position_ids via the model's own get_rope_index (the REAL train path)
    fn = getattr(model, "get_rope_index", None) or getattr(
        getattr(model, "model", model), "get_rope_index", None)
    if fn is not None:
        pos, _ = fn(input_ids, image_grid_thw=image_grid_thw)
        position_ids = pos if pos.dim() == 3 else pos.unsqueeze(1)
        print("position_ids from get_rope_index, shape:", tuple(position_ids.shape))
    else:
        position_ids = None
        print("get_rope_index not found; letting model compute internally")

    fwd = dict(input_ids=input_ids, pixel_values=pixel_values,
               image_grid_thw=image_grid_thw)
    if position_ids is not None:
        fwd["position_ids"] = position_ids
    o = model(**fwd)
    logits = o.logits.float()

    labels = torch.roll(input_ids, shifts=-1, dims=1)
    logp = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    logp = torch.roll(logp, shifts=1, dims=1)
    recomputed = logp[0, len(prompt_ids):].cpu().numpy()
    stored = np.array(rollout_lp)

    n = min(len(recomputed), len(stored))
    diff = np.abs(recomputed[:n] - stored[:n])
    print("=== STAGE1 v2 layer6img ===")
    print(f"compared {n} tokens | mean|diff|={diff.mean():.5f} max={diff.max():.5f}")
    print("per-token:", np.round(diff, 4))
    if diff.mean() < 0.02:
        print("VERDICT: image path aligned (processor-driven). Ready for OSWorld.")
    else:
        print("VERDICT: mismatch. If diff starts after the image block -> position;"
              " if uniform small -> precision; single spike -> boundary token.")


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "layer5img"
    if mode == "layer5img":
        asyncio.run(layer5img())
    elif mode == "layer6img":
        layer6img()
    else:
        print("modes: layer5img | layer6img")
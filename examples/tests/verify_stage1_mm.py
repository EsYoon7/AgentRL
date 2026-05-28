"""
STAGE 1 — multimodal path verification with ONE dummy image (no OSWorld).

Run BEFORE wiring real screenshots. Isolates the image-handling correctness:
  1A. mrope cross-check: does OUR compute_mrope_position_ids match the model's
      own get_rope_index for the same windowed layout? (the #1 risk)
  1B. placeholder/grid alignment: do our placeholder counts == processor grids?
  1C. image-sequence logprob match (Layer 6 with an image): rollout vs train.

Split into layer5img / layer6img like before to avoid holding sglang + HF model
in GPU memory simultaneously.

  python verify_stage1_mm.py mrope     # 1A + 1B, CPU-friendly (loads model on CPU for get_rope_index only if needed)
  python verify_stage1_mm.py layer5img # rollout one image turn, save ids+image
  python verify_stage1_mm.py layer6img # train-forward compare with image

Fill MODEL_PATH and the constants from extract_model_constants.py output.
"""

import asyncio
import io
import json

import numpy as np
import torch
from PIL import Image


# ---- from extract_model_constants.py (your Qwen3.5-9B values) -------------
MODEL_PATH = "/mnt/home/justiwag/esyoon/models/Qwen3.5-9B"
IMAGE_PLACEHOLDER_ID = 248056
IMAGE_PAD_OPEN_IDS = [248053]
IMAGE_PAD_CLOSE_IDS = [248054]
ASSISTANT_HEADER_IDS = [248045, 74455, 198]
ASSISTANT_EOT_IDS = [248046]
# text collapse marker ids: paste your resolved block (e.g. for
# "This screenshot has been collapsed.") -- not needed for stage 1 single image.
COLLAPSE_MARKER_IDS = [1919, 34993, 682, 978, 27350, 13]    

CACHE = "stage1_out.json"

# a small dummy image; pick a size that is a clean multiple of patch*merge=32
DUMMY_HW = (224, 224)   # -> grid (224/16/2)=7 x 7 = 49 placeholders


def _make_dummy_png(h, w):
    img = Image.new("RGB", (w, h), (120, 120, 120))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ===========================================================================
# 1A + 1B : mrope cross-check and placeholder/grid alignment
# ===========================================================================
def stage1_mrope_check():
    from transformers import AutoProcessor
    from mrope_mm import compute_mrope_position_ids   # adjust import path if needed

    proc = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    ip = proc.image_processor
    patch, merge = ip.patch_size, ip.merge_size

    h, w = DUMMY_HW
    png = _make_dummy_png(h, w)
    img = Image.open(io.BytesIO(png)).convert("RGB")

    # processor's own grid + placeholder count
    out = proc.image_processor(images=img, size={
        "shortest_edge": h * w, "longest_edge": h * w})
    grid_thw = out["image_grid_thw"][0]
    gt, gh, gw = int(grid_thw[0]), int(grid_thw[1]), int(grid_thw[2])
    # post-merge placeholder count
    real_ph = (gt * gh * gw) // (merge * merge)
    print("=== 1B placeholder/grid ===")
    print(f"processor grid_thw (pre-merge) = ({gt},{gh},{gw}); placeholders = {real_ph}")

    # our post-merge grid
    our_gh = (h // patch) // merge
    our_gw = (w // patch) // merge
    our_ph = 1 * our_gh * our_gw
    print(f"our post-merge grid = (1,{our_gh},{our_gw}); placeholders = {our_ph}")
    assert our_ph == real_ph, f"PLACEHOLDER MISMATCH {our_ph} != {real_ph}"
    print("placeholder count MATCH\n")

    # build a tiny sequence: 2 text, image block, 2 text (with vision pads)
    pre = [101, 102]
    img_block = IMAGE_PAD_OPEN_IDS + [IMAGE_PLACEHOLDER_ID] * our_ph + IMAGE_PAD_CLOSE_IDS
    post = [201, 202]
    full = pre + img_block + post

    our_pos = compute_mrope_position_ids(
        full, image_placeholder_id=IMAGE_PLACEHOLDER_ID,
        image_grids=[(1, our_gh, our_gw)],
    )  # [3, L]

    # model's get_rope_index (the ground truth)
    print("=== 1A mrope cross-check ===")
    try:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_PATH, dtype=torch.bfloat16)
        input_ids_t = torch.tensor(full).unsqueeze(0)
        image_grid_thw = torch.tensor([[gt, gh, gw]])
        # most Qwen-VL models: model.get_rope_index or model.model.get_rope_index
        fn = getattr(model, "get_rope_index", None) or \
             getattr(getattr(model, "model", model), "get_rope_index", None)
        if fn is None:
            print("get_rope_index not found on model; skipping exact cross-check.")
            print("our positions[:, :8] =\n", our_pos[:, :8])
            return
        ref_pos, _ = fn(input_ids_t, image_grid_thw=image_grid_thw)
        ref = ref_pos.squeeze(1).cpu().numpy() if ref_pos.dim() == 3 else ref_pos.cpu().numpy()
        print("ref shape:", ref.shape, "ours:", our_pos.shape)
        match = np.array_equal(ref[:, :our_pos.shape[1]], our_pos)
        print("MROPE EXACT MATCH:", match)
        if not match:
            print("ref :\n", ref[:, :12])
            print("ours:\n", our_pos[:, :12])
            print(">>> fix mrope_mm to match the model before training.")
    except Exception as e:
        print("cross-check skipped:", e)
        print("our positions[:, :8] =\n", our_pos[:, :8])


# ===========================================================================
# 1C : image-sequence rollout (layer5img) and train compare (layer6img)
# ===========================================================================
async def stage1_layer5img():
    import sglang as sgl
    from transformers import AutoProcessor
    from agentrl.trainer.components.mrope_mm import compute_mrope_position_ids

    proc = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    patch, merge = proc.image_processor.patch_size, proc.image_processor.merge_size

    h, w = DUMMY_HW
    png = _make_dummy_png(h, w)
    gh, gw = (h // patch) // merge, (w // patch) // merge
    n_ph = gh * gw

    # build prompt: <vision_start><image_pad>*n<vision_end> + question text
    import base64
    q_ids = proc.tokenizer("Describe the image in one word.",
                           add_special_tokens=True)["input_ids"]
    prompt_ids = (IMAGE_PAD_OPEN_IDS + [IMAGE_PLACEHOLDER_ID] * n_ph
                  + IMAGE_PAD_CLOSE_IDS + q_ids + ASSISTANT_HEADER_IDS)

    # SGLang expects image_data entries as STRINGS (base64/path/url), not dicts.
    # To control resolution per-image (windowing/downsampling), we PRE-RESIZE the
    # PNG to the exact target size, then send it. The placeholder count we put in
    # prompt_ids must match this resized size.
    img_resized = Image.open(io.BytesIO(png)).convert("RGB").resize((w, h))
    buf = io.BytesIO(); img_resized.save(buf, format="PNG")
    png_resized_b64 = base64.b64encode(buf.getvalue()).decode()

    engine = sgl.Engine(model_path=MODEL_PATH, dtype="bfloat16", tp_size=1)
    engine.tokenizer_manager.auto_create_handle_loop()
    ret = await engine.async_generate(
        input_ids=prompt_ids,
        image_data=[png_resized_b64],
        sampling_params={"max_new_tokens": 8, "temperature": 0.0},
        return_logprob=True,
    )
    if isinstance(ret, list):
        ret = ret[0]
    otlp = ret["meta_info"]["output_token_logprobs"]
    output_ids = [int(x[1]) for x in otlp]
    rollout_lp = [float(x[0]) for x in otlp]

    full = prompt_ids + output_ids
    pos = compute_mrope_position_ids(
        full, image_placeholder_id=IMAGE_PLACEHOLDER_ID,
        image_grids=[(1, gh, gw)])

    with open(CACHE, "w") as f:
        json.dump({
            "full": full, "prompt_len": len(prompt_ids),
            "output_ids": output_ids, "rollout_lp": rollout_lp,
            "position_ids": pos.tolist(),
            "img_h": h, "img_w": w,
        }, f)
    print("=== STAGE1 Layer5img ===")
    print("n_placeholders:", n_ph, "output_ids:", output_ids)
    print("rollout_lp len:", len(rollout_lp), "== output", len(output_ids))
    print(f"[saved -> {CACHE}; now run: python verify_stage1_mm.py layer6img]")


@torch.no_grad()
def stage1_layer6img():
    from transformers import AutoModelForImageTextToText, AutoProcessor
    with open(CACHE) as f:
        d = json.load(f)
    full = d["full"]; prompt_len = d["prompt_len"]
    rollout_lp = d["rollout_lp"]; pos = np.array(d["position_ids"])
    h, w = d["img_h"], d["img_w"]

    proc = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    png = _make_dummy_png(h, w)
    img = Image.open(io.BytesIO(png)).convert("RGB")
    out = proc.image_processor(images=img, size={
        "shortest_edge": h * w, "longest_edge": h * w})
    pixel_values = torch.as_tensor(out["pixel_values"]).cuda()
    image_grid_thw = torch.as_tensor(out["image_grid_thw"]).cuda()

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()

    input_ids = torch.tensor(full).unsqueeze(0).cuda()
    position_ids = torch.as_tensor(pos).unsqueeze(1).cuda()  # [3,1,L]

    o = model(input_ids=input_ids, position_ids=position_ids,
              pixel_values=pixel_values, image_grid_thw=image_grid_thw)
    logits = o.logits.float()

    labels = torch.roll(input_ids, shifts=-1, dims=1)
    logp = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    logp = torch.roll(logp, shifts=1, dims=1)
    recomputed = logp[0, prompt_len:].cpu().numpy()
    stored = np.array(rollout_lp)

    n = min(len(recomputed), len(stored))
    diff = np.abs(recomputed[:n] - stored[:n])
    print("=== STAGE1 Layer6img (image-sequence logprob) ===")
    print(f"compared {n} tokens | mean|diff|={diff.mean():.5f} max|diff|={diff.max():.5f}")
    print("per-token:", np.round(diff, 4))
    if diff.mean() < 0.02:
        print("VERDICT: image path aligned. Ready to wire real screenshots.")
    else:
        print("VERDICT: mismatch. Suspect mrope (run `mrope` mode) or pixel/grid.")


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "mrope"
    if mode == "mrope":
        stage1_mrope_check()
    elif mode == "layer5img":
        asyncio.run(stage1_layer5img())
    elif mode == "layer6img":
        stage1_layer6img()
    else:
        print("modes: mrope | layer5img | layer6img")
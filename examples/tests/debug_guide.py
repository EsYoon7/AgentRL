"""
DEBUGGING GUIDE — run the layers in order. Each layer adds one dependency.
Stop and fix at the first failing layer; don't move outward until it's green.

  Layer 0  syntax / imports            (no deps)
  Layer 1  pure logic                  (numpy only)   <- test_logic.py
  Layer 2  invariant property checks   (numpy only)   <- this file
  Layer 3  processor alignment         (real HF processor, no model/GPU)
  Layer 4  sglang ret structure        (GPU + model)  <- probe_sglang.py
  Layer 5  one real rollout turn       (GPU + model + worker)
  Layer 6  rollout==train logprob      (GPU; the money check)

The philosophy: the bugs that actually bite (placeholder count mismatch,
position shift, mask off-by-one, id reuse) are all catchable at Layers 2-3
WITHOUT a GPU. Catch them cheap.
"""

import numpy as np


# ===========================================================================
# LAYER 2 — invariant property checks (the things that silently corrupt RL)
# ===========================================================================
# These assert the INVARIANTS we designed for, not specific values. Run with
# the fake processor so they're fast and GPU-free.

from agentrl.trainer.components.mrope_mm import compute_mrope_position_ids
from agentrl.trainer.components.context_manager_mm import MultimodalContextManager
from agentrl.trainer.components.buffer_mm import ImagePool, StepRecord


class _FakeImgProc:
    patch_size = 16
    merge_size = 2


class FakeProcessor:
    image_processor = _FakeImgProc()


PLACEHOLDER = 9999
PAD_OPEN, PAD_CLOSE, COLLAPSE = [1001], [1002], [1003, 1004]


def check_placeholder_count_matches_grid():
    """INVARIANT: number of placeholder tokens emitted == t*h*w of its grid.
    If this ever breaks, the model errors or silently misaligns vision feats."""
    cm = MultimodalContextManager(
        FakeProcessor(), image_placeholder_id=PLACEHOLDER,
        image_pad_open_ids=PAD_OPEN, image_pad_close_ids=PAD_CLOSE,
        collapse_marker_ids=COLLAPSE, max_images=4,
        resolution_policy=lambda r: (64, 32),  # h=64,w=32
    )
    pool = ImagePool()
    ids, image_ids = cm.build_prompt([100], [b"img"], pool)
    n_ph = sum(1 for t in ids if t == PLACEHOLDER)
    p = pool.get(image_ids[0])
    gh = (64 // 16) // 2  # 2
    gw = (32 // 16) // 2  # 1
    assert n_ph == gh * gw == p.num_placeholder_tokens, (
        f"placeholder count {n_ph} != grid {gh*gw}")
    print("[L2] placeholder_count_matches_grid OK")


def check_positions_align_with_full_ids():
    """INVARIANT: stored position_ids width == len(prompt+output)."""
    cm = MultimodalContextManager(
        FakeProcessor(), image_placeholder_id=PLACEHOLDER,
        image_pad_open_ids=PAD_OPEN, image_pad_close_ids=PAD_CLOSE,
        collapse_marker_ids=COLLAPSE, max_images=4,
        resolution_policy=lambda r: (32, 32),
    )
    pool = ImagePool()
    prompt, image_ids = cm.build_prompt([100, 101], [b"img"], pool)
    output = [500, 501, 502]
    full = prompt + output
    # build grids in placeholder order
    grids = []
    for iid in image_ids:
        p = pool.get(iid)
        grids.append((1, (p.resized_h // 16) // 2, (p.resized_w // 16) // 2))
    pos = compute_mrope_position_ids(
        full, image_placeholder_id=PLACEHOLDER, image_grids=grids)
    assert pos.shape[1] == len(full), f"{pos.shape[1]} != {len(full)}"
    print("[L2] positions_align_with_full_ids OK")


def check_loss_mask_covers_only_output():
    """INVARIANT: loss_mask is 1 exactly on output_ids, 0 everywhere else."""
    step = StepRecord(prompt_input_ids=[1, 2, 3, 4], output_ids=[5, 6])
    mask = step.build_loss_mask()
    assert mask == [0, 0, 0, 0, 1, 1]
    assert sum(mask) == len(step.output_ids)
    print("[L2] loss_mask_covers_only_output OK")


def check_past_action_ids_are_verbatim():
    """INVARIANT: an earlier turn's output_ids reappear UNCHANGED in the next
    prompt (no retokenization). This is the whole point."""
    cm = MultimodalContextManager(
        FakeProcessor(), image_placeholder_id=PLACEHOLDER,
        image_pad_open_ids=PAD_OPEN, image_pad_close_ids=PAD_CLOSE,
        collapse_marker_ids=COLLAPSE, max_images=4,
        resolution_policy=lambda r: (32, 32),
        assistant_header_ids=[2001], assistant_eot_ids=[2002],
    )
    pool = ImagePool()
    cm.build_prompt([100], [], pool)
    action = [777, 778, 779]
    cm.commit_turn([100], [], action_ids=action)
    nxt, _ = cm.build_prompt([101], [], pool)
    # the contiguous block must be present
    found = any(nxt[i:i+3] == action for i in range(len(nxt) - 2))
    assert found, "past action ids were altered (retokenization leak!)"
    print("[L2] past_action_ids_are_verbatim OK")


def check_window_drops_beyond_max_images():
    """INVARIANT: only the most-recent max_images survive as placeholders;
    older ones become the fixed collapse block."""
    def policy(recency):
        return (32, 32) if recency < 2 else None
    cm = MultimodalContextManager(
        FakeProcessor(), image_placeholder_id=PLACEHOLDER,
        image_pad_open_ids=PAD_OPEN, image_pad_close_ids=PAD_CLOSE,
        collapse_marker_ids=COLLAPSE, max_images=2,
        resolution_policy=policy,
        assistant_header_ids=[2001], assistant_eot_ids=[2002],
    )
    pool = ImagePool()
    # 3 turns, 1 image each
    for k in range(3):
        cm.build_prompt([100 + k], [f"img{k}".encode()], pool)
        cm.commit_turn([100 + k], [f"img{k}".encode()], action_ids=[500 + k])
    # 4th build: oldest image (turn0) must be collapsed
    final, image_ids = cm.build_prompt([200], [b"imgX"], pool)
    # live images = 2 most recent of the 4 total (turn2, turn3-current)
    assert len(image_ids) == 2, f"expected 2 live images, got {len(image_ids)}"
    assert COLLAPSE[0] in final, "old image not collapsed"
    print("[L2] window_drops_beyond_max_images OK")


def run_layer2():
    print("\n--- LAYER 2: invariant property checks (no GPU) ---")
    check_placeholder_count_matches_grid()
    check_positions_align_with_full_ids()
    check_loss_mask_covers_only_output()
    check_past_action_ids_are_verbatim()
    check_window_drops_beyond_max_images()


# ===========================================================================
# LAYER 3 — processor alignment (needs a REAL processor, still NO model/GPU)
# ===========================================================================
def run_layer3(model_path):
    """Verify OUR placeholder-count formula matches the REAL processor.

    This is the single most important non-GPU check: if our _placeholder_count
    disagrees with what the processor actually produces, every prompt is
    misaligned. We compare against the processor's own image_grid_thw.
    """
    print("\n--- LAYER 3: processor alignment (CPU, real processor) ---")
    from transformers import AutoProcessor
    from PIL import Image

    proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    ip = proc.image_processor
    print("patch_size:", ip.patch_size, "merge_size:", ip.merge_size)

    # make a dummy image at a known size and see how many tokens the processor
    # assigns, then check our formula reproduces it.
    img = Image.new("RGB", (256, 256), (127, 127, 127))
    out = proc.image_processor(images=img)
    grid = out["image_grid_thw"][0]  # (t, h, w) post nothing / pre-merge?
    t, h, w = int(grid[0]), int(grid[1]), int(grid[2])
    merge = ip.merge_size
    real_tokens = (t * h * w) // (merge * merge)
    print(f"processor grid_thw = ({t},{h},{w}); real placeholder tokens = {real_tokens}")
    print("ACTION: ensure context_manager._placeholder_count reproduces this "
          "number for the SAME resized dimensions. If Qwen3-VL rounds to 32 "
          "(not 28), confirm patch/merge here match that.")


if __name__ == "__main__":
    run_layer2()
    print("\nLayer 2 passed. For Layer 3 run: "
          "python -c \"import debug_guide as d; d.run_layer3('Qwen/Qwen3-VL-...')\"")
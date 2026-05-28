"""Validation harness with a fake processor. No torch / no model needed."""

import numpy as np
from agentrl.trainer.components.mrope_mm import compute_mrope_position_ids
from agentrl.trainer.components.context_manager_mm import MultimodalContextManager
from agentrl.trainer.components.buffer_mm import ImagePool


# ---- fake processor exposing patch/merge like Qwen-VL ----
class _FakeImgProc:
    patch_size = 16
    merge_size = 2


class FakeProcessor:
    image_processor = _FakeImgProc()


PLACEHOLDER = 9999
PAD_OPEN = [1001]
PAD_CLOSE = [1002]
COLLAPSE = [1003, 1004]  # fixed collapse marker block


def test_mrope_text_only():
    ids = [5, 6, 7, 8]
    pos = compute_mrope_position_ids(ids, image_placeholder_id=PLACEHOLDER, image_grids=[])
    # text-only: all three dims equal 0..3
    assert pos.shape == (3, 4)
    assert (pos[0] == [0, 1, 2, 3]).all()
    assert (pos == pos[0]).all(), "text dims must be equal"
    print("test_mrope_text_only OK")


def test_mrope_with_image():
    # 2 text, then an image with grid (t=1,h=2,w=2) -> 4 placeholders, then 2 text
    ids = [5, 6] + [PLACEHOLDER] * 4 + [7, 8]
    grids = [(1, 2, 2)]
    pos = compute_mrope_position_ids(ids, image_placeholder_id=PLACEHOLDER, image_grids=grids)
    assert pos.shape == (3, 8)
    # text tokens 0,1 -> positions 0,1
    assert (pos[:, 0] == 0).all() and (pos[:, 1] == 1).all()
    # image starts at scalar 2; t all =2; h in {2,3}; w in {2,3}
    img = pos[:, 2:6]
    assert (img[0] == 2).all(), "temporal dim constant for single-frame image"
    assert set(img[1].tolist()) == {2, 3}, "height indices"
    assert set(img[2].tolist()) == {2, 3}, "width indices"
    # after image: next_start = 2 + max(1,2,2) = 4
    assert (pos[:, 6] == 4).all(), f"got {pos[:,6]}"
    assert (pos[:, 7] == 5).all()
    print("test_mrope_with_image OK")


def test_windowing_drops_old_images():
    """max_images=1: with 2 turns each having 1 image, the older image must
    collapse to the marker block; the newer stays as placeholders."""

    def res_policy(recency):
        # recency 0 -> high res 32x32; older -> drop
        if recency == 0:
            return (32, 32)
        return None

    cm = MultimodalContextManager(
        FakeProcessor(),
        image_placeholder_id=PLACEHOLDER,
        image_pad_open_ids=PAD_OPEN,
        image_pad_close_ids=PAD_CLOSE,
        collapse_marker_ids=COLLAPSE,
        max_images=1,
        resolution_policy=res_policy,
        assistant_header_ids=[2001],
        assistant_eot_ids=[2002],
    )
    pool = ImagePool()

    # turn 0: obs with one image (raw bytes distinct), action
    img_a = b"AAAA_png"
    p0, ids0 = cm.build_prompt([100, 101], [img_a], pool)
    # only one image so far -> it is recency 0 -> rendered as placeholders
    n_tok = (32 // 16 // 2) * (32 // 16 // 2)  # =1*1=1
    assert PLACEHOLDER in p0
    assert len(ids0) == 1
    cm.commit_turn([100, 101], [img_a], action_ids=[500, 501])

    # turn 1: new obs with a new image -> now img_a is older (recency 1 -> drop)
    img_b = b"BBBB_png"
    p1, ids1 = cm.build_prompt([102, 103], [img_b], pool)
    # the older image (img_a) must appear as the collapse marker block, not placeholders
    assert COLLAPSE[0] in p1 and COLLAPSE[1] in p1, "old image not collapsed"
    # exactly one live image (img_b) referenced
    assert len(ids1) == 1, f"expected 1 live image, got {len(ids1)}"
    # the live image id must be the down/again rendered img_b
    assert ids1[0] in pool._by_key
    print("test_windowing_drops_old_images OK")


def test_history_reuses_action_ids():
    """Past action ids must be reused verbatim (not retokenized)."""

    def res_policy(recency):
        return (32, 32)

    cm = MultimodalContextManager(
        FakeProcessor(),
        image_placeholder_id=PLACEHOLDER,
        image_pad_open_ids=PAD_OPEN,
        image_pad_close_ids=PAD_CLOSE,
        collapse_marker_ids=COLLAPSE,
        max_images=4,
        resolution_policy=res_policy,
        assistant_header_ids=[2001],
        assistant_eot_ids=[2002],
    )
    pool = ImagePool()
    cm.build_prompt([100], [], pool)
    cm.commit_turn([100], [], action_ids=[777, 778, 779])
    p1, _ = cm.build_prompt([101], [], pool)
    # the exact action id block 777,778,779 must be present contiguously
    s = ",".join(map(str, p1))
    assert "777,778,779" in s, "past action ids not reused verbatim"
    print("test_history_reuses_action_ids OK")


if __name__ == "__main__":
    test_mrope_text_only()
    test_mrope_with_image()
    test_windowing_drops_old_images()
    test_history_reuses_action_ids()
    print("\nALL TESTS PASSED")
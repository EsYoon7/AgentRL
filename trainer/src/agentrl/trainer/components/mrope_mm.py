"""
mrope (3D RoPE) position id computation for Qwen-VL style models.

Why this exists: text tokens use a scalar position; image tokens use a 3D
(temporal, height, width) position derived from the image's patch grid. When we
window/down-sample history images, the placeholder count of each image changes,
which changes the grid AND shifts every subsequent text token's position. We
compute positions at rollout time on the EXACT windowed layout and store them,
so training never has to re-derive (and never diverges).

This is a faithful re-implementation of the Qwen2/2.5-VL get_rope_index logic
for a single sequence, kept small and dependency-light. For Qwen3-VL confirm
patch/merge constants against the actual processor; the algorithm is the same.

Reference behavior:
  - text token at sequence step advances all 3 dims by 1 (they stay equal).
  - an image block occupies (t, h, w) grid positions; the three coordinate
    planes are filled with the grid indices, offset by the running max+1.
  - after an image, the scalar continues from (max position in image + 1).
"""

from __future__ import annotations

import numpy as np


def compute_mrope_position_ids(
    input_ids: list[int],
    *,
    image_placeholder_id: int,
    image_grids: list[tuple[int, int, int]],  # (t, h, w) per image, post-merge
) -> np.ndarray:
    """Return position_ids of shape [3, len(input_ids)].

    image_grids must be in the SAME order the image placeholder blocks appear
    in input_ids. Each grid is the POST-merge grid (i.e. matches the number of
    placeholder tokens = t*h*w).
    """
    seq_len = len(input_ids)
    pos = np.zeros((3, seq_len), dtype=np.int64)

    img_idx = 0
    i = 0
    next_start = 0  # the scalar position value the next text token takes

    while i < seq_len:
        if input_ids[i] == image_placeholder_id:
            t, h, w = image_grids[img_idx]
            n = t * h * w
            assert i + n <= seq_len, "image block overruns sequence"

            st = next_start
            # temporal index: repeats across h*w for each t
            t_index = np.repeat(np.arange(t), h * w)
            # height index: for each t, each row repeated across w
            h_index = np.tile(np.repeat(np.arange(h), w), t)
            # width index: for each t and row, 0..w-1
            w_index = np.tile(np.arange(w), t * h)

            pos[0, i : i + n] = st + t_index
            pos[1, i : i + n] = st + h_index
            pos[2, i : i + n] = st + w_index

            # next text position continues from the max coordinate used + 1
            next_start = st + max(t, h, w)
            i += n
            img_idx += 1
        else:
            pos[:, i] = next_start
            next_start += 1
            i += 1

    return pos


def grids_from_placeholder_counts(
    image_grids_hw: list[tuple[int, int]],
) -> list[tuple[int, int, int]]:
    """Convenience: turn (h, w) post-merge grids into (t=1, h, w) for images."""
    return [(1, h, w) for (h, w) in image_grids_hw]
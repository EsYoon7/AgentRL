"""
position_fill_mm.py  -> trainer/src/agentrl/trainer/components/

Fill mrope position_ids on the TRAINER side. The rollout worker only has sglang
(no HF model) so it cannot call get_rope_index; we stored input_ids and
image_grid_thw per item, and compute positions here, once, before the forward.

FSDPWorker.collate asserts position_ids is present when packing, so we must fill
it for every item. For text-only items (no image_grid_thw) positions are a plain
arange broadcast to 3 rows (matching how mrope degenerates for text).

Call fill_position_ids(data, get_rope_index_fn) right after you receive `data`
from the task manager and before ref/actor forward. get_rope_index_fn is the
model's bound method; obtain it once from the actor/ref worker, or replicate the
model's logic. Simplest: run it on the actor rank via a small RPC, or compute on
CPU here if you can import the model's get_rope_index without the weights.
"""

from __future__ import annotations

import torch


def _text_only_positions(seq_len):
    ar = torch.arange(seq_len, dtype=torch.long)
    return torch.stack([ar, ar, ar], dim=0).unsqueeze(1)  # [3,1,L]


def fill_position_ids(data, get_rope_index_fn=None):
    """Mutate each item in `data`, adding item['position_ids'] shape [3,1,L].

    get_rope_index_fn(input_ids, image_grid_thw=...) -> position_ids
      (the model's method). If None, only text-only items are handled and items
      with images will raise -- so pass the real fn when images are present.
    """
    for item in data:
        if "position_ids" in item:
            continue
        input_ids = item["input_ids"]            # [1, L]
        seq_len = input_ids.shape[1]
        mm = item.get("multi_modal_inputs") or {}
        grid = mm.get("image_grid_thw")

        if grid is None:
            item["position_ids"] = _text_only_positions(seq_len)
            continue

        if get_rope_index_fn is None:
            raise RuntimeError(
                "image item needs get_rope_index_fn to compute mrope positions")
        pos, _ = get_rope_index_fn(input_ids, image_grid_thw=grid)
        # normalize to [3,1,L]
        if pos.dim() == 2:        # [3, L]
            pos = pos.unsqueeze(1)
        elif pos.dim() == 3 and pos.shape[1] != 1:  # [3, B, L] with B>1
            pos = pos[:, :1, :]
        item["position_ids"] = pos.cpu()


# ---------------------------------------------------------------------------
# Helper to obtain get_rope_index from an HF model instance (e.g. on the actor).
# Add a small method to FSDPWorker that exposes this, or call locally if you can
# load the model config-only. Example FSDPWorker method:
#
#   def compute_positions(self, list_of_input_ids, list_of_grids):
#       fn = getattr(self.model, "get_rope_index", None) or \
#            getattr(self.model.model, "get_rope_index", None)
#       out = []
#       for ids, grid in zip(list_of_input_ids, list_of_grids):
#           pos, _ = fn(ids.cuda(), image_grid_thw=(grid.cuda() if grid is not None else None))
#           out.append(pos.cpu())
#       return out
#
# Then in the trainer, gather input_ids/grids, call actor.compute_positions, and
# assign back. This keeps the (frozen-structure) get_rope_index on the GPU model.
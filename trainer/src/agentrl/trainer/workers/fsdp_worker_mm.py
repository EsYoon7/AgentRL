"""
fsdp_worker_mm.py  -> trainer/src/agentrl/trainer/workers/

Subclass of FSDPWorker that exposes get_rope_index so the trainer can compute
mrope position_ids for image items (the rollout worker can't -- it has no HF
model). Everything else is inherited unchanged.

Use spawn(FSDPWorkerMM, ...) for the actor in async_trainer_mm.py if you want
compute_positions; otherwise you can keep FSDPWorker for ref.
"""

from __future__ import annotations

import torch

from agentrl.trainer.workers.fsdp_worker import FSDPWorker


class FSDPWorkerMM(FSDPWorker):
    def compute_positions(self, list_of_input_ids, list_of_grids):
        """Return mrope position_ids [3,1,L] per item via model.get_rope_index.

        list_of_input_ids : list of LongTensor [1, L]
        list_of_grids     : list of image_grid_thw tensors (or None)
        """
        fn = getattr(self.model, "get_rope_index", None) or getattr(
            getattr(self.model, "model", self.model), "get_rope_index", None)
        assert fn is not None, "model has no get_rope_index"

        out = []
        for ids, grid in zip(list_of_input_ids, list_of_grids):
            ids_c = ids.cuda()
            grid_c = grid.cuda() if grid is not None else None
            with torch.no_grad():
                pos, _ = fn(ids_c, image_grid_thw=grid_c)
            if pos.dim() == 2:          # [3, L]
                pos = pos.unsqueeze(1)
            elif pos.dim() == 3 and pos.shape[1] != 1:
                pos = pos[:, :1, :]
            out.append(pos.cpu())
        return out
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
    # Qwen3.5-VL image placeholder token id (<|image_pad|>). mm_token_type_ids
    # marks these positions with 1, all other tokens with 0. Confirmed against
    # transformers qwen3_vl processing: mm_token_type_ids[input_ids==image_token]=1.
    IMAGE_TOKEN_ID = 248056

    def compute_positions(self, list_of_input_ids, list_of_grids):
        print(f"[CP r{self.rank}] ENTER n_items={len(list_of_input_ids)}", flush=True)
        fn = getattr(self.model, "get_rope_index", None) or getattr(
            getattr(self.model, "model", self.model), "get_rope_index", None)
        assert fn is not None, "model has no get_rope_index"

        out = []
        for i, (ids, grid) in enumerate(zip(list_of_input_ids, list_of_grids)):
            ids_c = ids.cuda()
            grid_c = grid.cuda() if grid is not None else None
            mm_token_type_ids = (ids_c == self.IMAGE_TOKEN_ID).to(torch.int)

            # --- 진단: placeholder 토큰 수 vs grid가 나타내는 토큰 수 ---
            n_img_tokens = int(mm_token_type_ids.sum().item())
            if grid_c is not None:
                # image_grid_thw [N,3] = (t, h, w) per image. merge_size^2로 나눈 게
                # 실제 placeholder 수. Qwen-VL merge_size는 보통 2 → /4.
                grid_list = grid.tolist()
                expected = sum(int(t*h*w) for t, h, w in grid_list)  # merge 전
            else:
                grid_list = None
                expected = 0
            print(f"[CP r{self.rank}] item {i}/{len(list_of_input_ids)}: "
                f"ids_len={ids.shape[-1]} n_img_placeholders={n_img_tokens} "
                f"grid={grid_list} grid_raw_tokens={expected}", flush=True)
            # ----------------------------------------------------------

            with torch.no_grad():
                try:
                    pos, _ = fn(ids_c, mm_token_type_ids, image_grid_thw=grid_c)
                except TypeError:
                    pos, _ = fn(ids_c, image_grid_thw=grid_c)
            print(f"[CP r{self.rank}] item {i} get_rope_index OK", flush=True)  # ← 추가
            if pos.dim() == 2:
                pos = pos.unsqueeze(1)
            elif pos.dim() == 3 and pos.shape[1] != 1:
                pos = pos[:, :1, :]
            out.append(pos.cpu())
        print(f"[CP r{self.rank}] DONE", flush=True)
        return out
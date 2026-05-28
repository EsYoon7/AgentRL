"""
Per-trajectory context manager for multi-turn multimodal rollout.

ONE instance per trajectory. It owns the evolving conversation and is
responsible for producing, at each turn, the EXACT token-id prompt that the
rollout engine will consume -- with image windowing/down-sampling already
applied. The ids it produces are what get persisted for training.

Why token-id level and not text level:
    If we kept history as text and re-applied the chat template every turn,
    the assistant turns we generated earlier would be retokenized as part of
    the new prompt. For training that prompt region is masked, so it does not
    directly corrupt loss -- BUT it can shift positions of the (unmasked)
    output tokens and change what they attend to, perturbing their logprob.
    Keeping history as preserved id-blocks removes that risk entirely.

The image windowing mirrors the model's NATIVE inference behavior: keep the
most-recent `max_images` screenshots at high resolution; older ones are either
down-sampled (lower pixel budget) or dropped to a fixed collapse-marker id
block. We reproduce inference exactly, so the training distribution matches.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from PIL import Image

from ..components.buffer_mm import ImagePool


@dataclass
class HistoryTurn:
    """One past turn, stored as id blocks (never as raw text)."""

    # the user/observation block for this turn, as token ids (text portion)
    obs_text_ids: list[int]
    # screenshots attached to this turn's observation. Each entry is the RAW
    # PNG bytes at native/full resolution -- the window policy decides what
    # resolution to actually render it at, per current distance from the head.
    obs_raw_images: list[bytes] = field(default_factory=list)
    # the assistant action generated this turn, ids preserved from rollout
    action_ids: list[int] = field(default_factory=list)


class MultimodalContextManager:
    """Builds windowed token-id prompts; mirrors native inference windowing.

    Parameters
    ----------
    processor :
        HF processor (e.g. Qwen3-VL AutoProcessor). Used to (a) turn raw PNG +
        target resolution into pixel info and (b) tell us the placeholder token
        count for an image at a given resolution.
    image_placeholder_id :
        token id used as the per-patch image placeholder (model specific).
    image_pad_open_ids / image_pad_close_ids :
        fixed id blocks wrapping an image (e.g. <|vision_start|>/<|vision_end|>).
    collapse_marker_ids :
        FIXED id block substituted for a dropped (fully removed) image. Must be
        the exact ids used during the model's training, so we hardcode the block
        rather than re-tokenizing a "[image removed]" string.
    max_images :
        number of most-recent screenshots kept "alive".
    resolution_policy :
        callable(distance_from_head:int) -> (resized_h, resized_w) | None
        Returns target resolution for an image that is `distance` turns behind
        the current head. None means "drop and use collapse marker".
    """

    def __init__(
        self,
        processor,
        *,
        image_placeholder_id: int,
        image_pad_open_ids: list[int],
        image_pad_close_ids: list[int],
        collapse_marker_ids: list[int],
        max_images: int,
        resolution_policy: Callable[[int], Optional[tuple[int, int]]],
        system_prompt_ids: Optional[list[int]] = None,
        assistant_header_ids: Optional[list[int]] = None,
        assistant_eot_ids: Optional[list[int]] = None,
    ) -> None:
        self.processor = processor
        self.image_placeholder_id = image_placeholder_id
        self.image_pad_open_ids = image_pad_open_ids
        self.image_pad_close_ids = image_pad_close_ids
        self.collapse_marker_ids = collapse_marker_ids
        self.max_images = max_images
        self.resolution_policy = resolution_policy

        self.system_prompt_ids = system_prompt_ids or []
        self.assistant_header_ids = assistant_header_ids or []
        self.assistant_eot_ids = assistant_eot_ids or []

        self._history: list[HistoryTurn] = []

    # ------------------------------------------------------------------
    # image helpers
    # ------------------------------------------------------------------
    def _placeholder_count(self, resized_h: int, resized_w: int) -> int:
        """How many placeholder tokens an image expands to at this resolution.

        Qwen-VL: tokens = (H/patch * W/patch) / (merge_size**2).
        We read patch/merge from the processor so it stays model-correct.
        """
        ip = self.processor.image_processor
        patch = ip.patch_size
        merge = ip.merge_size
        grid_h = resized_h // patch
        grid_w = resized_w // patch
        return (grid_h * grid_w) // (merge * merge)

    def _render_image_block(
        self,
        raw_png: bytes,
        resized_h: int,
        resized_w: int,
        pool: ImagePool,
    ) -> tuple[list[int], str]:
        """Return (id_block_for_this_image, image_id_in_pool)."""
        n_tok = self._placeholder_count(resized_h, resized_w)
        image_id = pool.add(raw_png, resized_h, resized_w, n_tok)
        block = (
            list(self.image_pad_open_ids)
            + [self.image_placeholder_id] * n_tok
            + list(self.image_pad_close_ids)
        )
        return block, image_id

    # ------------------------------------------------------------------
    # the main entry point
    # ------------------------------------------------------------------
    def build_prompt(
        self,
        new_obs_text_ids: list[int],
        new_obs_raw_images: list[bytes],
        pool: ImagePool,
    ) -> tuple[list[int], list[str]]:
        """Assemble the windowed prompt for the CURRENT turn.

        Returns
        -------
        prompt_input_ids : list[int]
        image_ids : list[str]   # pool references, in placeholder order
        """
        ids: list[int] = list(self.system_prompt_ids)
        image_ids: list[str] = []

        # The current turn's images count as distance 0; older history turns
        # are distance 1, 2, ... from the head. We figure out, for the whole
        # conversation (history + current obs), which screenshots are within
        # the most-recent `max_images` and at what resolution each renders.

        # Flatten all images with their owning turn distance. distance 0 is the
        # turn we are about to send (new_obs). History turn k (0=oldest) is at
        # distance = (len(history) - k) for its images... but distance for the
        # WINDOW is measured from the most recent, so compute per global index.
        # Simplest correct approach: walk turns from newest to oldest, count
        # images, decide resolution by how many newer images precede them.

        # Build the *sequence order* first (oldest -> newest), then assign
        # resolution based on recency rank.

        # 1) collect images in chronological order with a back-reference
        chrono: list[tuple[int, int, bytes]] = []  # (turn_pos, img_pos, raw)
        for t_pos, turn in enumerate(self._history):
            for i_pos, raw in enumerate(turn.obs_raw_images):
                chrono.append((t_pos, i_pos, raw))
        cur_turn_pos = len(self._history)
        for i_pos, raw in enumerate(new_obs_raw_images):
            chrono.append((cur_turn_pos, i_pos, raw))

        # 2) recency rank: newest image has rank 0
        total = len(chrono)
        # rank from the end
        rank_of = {idx: (total - 1 - idx) for idx in range(total)}

        # 3) decide per-image resolution (or drop) and remember by (turn,img)
        decision: dict[tuple[int, int], Optional[tuple[int, int]]] = {}
        for idx, (t_pos, i_pos, raw) in enumerate(chrono):
            recency = rank_of[idx]
            if recency < self.max_images:
                res = self.resolution_policy(recency)  # high-res band
            else:
                res = self.resolution_policy(recency)  # may be low-res or None
            decision[(t_pos, i_pos)] = res

        # ------------------------------------------------------------------
        # 4) now emit ids turn by turn (oldest -> newest), then current obs
        # ------------------------------------------------------------------
        def emit_obs(turn_pos: int, obs_text_ids: list[int], raws: list[bytes]):
            for i_pos, raw in enumerate(raws):
                res = decision[(turn_pos, i_pos)]
                if res is None:
                    # dropped: fixed collapse-marker id block, no pool entry
                    ids.extend(self.collapse_marker_ids)
                else:
                    rh, rw = res
                    block, image_id = self._render_image_block(raw, rh, rw, pool)
                    ids.extend(block)
                    image_ids.append(image_id)
            ids.extend(obs_text_ids)

        for t_pos, turn in enumerate(self._history):
            emit_obs(t_pos, turn.obs_text_ids, turn.obs_raw_images)
            # past assistant action: reuse preserved ids (NOT retokenized)
            ids.extend(self.assistant_header_ids)
            ids.extend(turn.action_ids)
            ids.extend(self.assistant_eot_ids)

        # current observation + the open assistant header for generation
        emit_obs(cur_turn_pos, new_obs_text_ids, new_obs_raw_images)
        ids.extend(self.assistant_header_ids)

        return ids, image_ids

    def commit_turn(
        self,
        new_obs_text_ids: list[int],
        new_obs_raw_images: list[bytes],
        action_ids: list[int],
    ) -> None:
        """Append the just-completed turn to history (as id blocks)."""
        self._history.append(
            HistoryTurn(
                obs_text_ids=list(new_obs_text_ids),
                obs_raw_images=list(new_obs_raw_images),
                action_ids=list(action_ids),
            )
        )

    @property
    def num_turns(self) -> int:
        return len(self._history)
"""
Data structures for multi-turn multimodal RL rollout buffering.

Design invariant (the whole point of this module):
    For every step we persist the EXACT `prompt_input_ids` the rollout engine
    saw and the EXACT `output_ids` it generated. Training forwards those same
    ids. Tokenization happens once, at rollout time. We never detokenize an
    assistant turn and retokenize it -> no retokenization drift.

Images are stored ONCE per (image_id, resolution) in a per-trajectory pool and
referenced by index from each step, so the sliding-window overlap between
consecutive turns does not duplicate raw pixels.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Image pool
# ---------------------------------------------------------------------------
# Key insight from the design discussion: because history images are
# downsampled as they age (Qwen3-VL min/max_pixels), the SAME screenshot can
# appear at DIFFERENT resolutions across turns. So the pool key must be
# (content_hash, resized_h, resized_w), not just the content. The high-res and
# low-res versions of one screenshot are distinct pool entries.


@dataclass
class PooledImage:
    """A single raw image at one specific target resolution."""

    image_id: str  # content_hash + resolution, stable within a trajectory
    raw_png: bytes  # raw PNG bytes (simplest, debuggable, RAM-friendly)
    resized_h: int  # the EXACT height the processor must resize to
    resized_w: int  # the EXACT width the processor must resize to
    # number of vision placeholder tokens this image expands to at this
    # resolution; needed to validate prompt_input_ids alignment at train time.
    num_placeholder_tokens: int


class ImagePool:
    """Per-trajectory pool. Deduplicates (content, resolution) pairs.

    We dedup on (content_hash, resized_h, resized_w). The same screenshot at
    full-res and at down-sampled-res are two different entries because they
    produce different pixel_values, different placeholder counts, and different
    positions.
    """

    def __init__(self) -> None:
        self._by_key: dict[str, PooledImage] = {}

    @staticmethod
    def _content_hash(raw_png: bytes) -> str:
        return hashlib.sha1(raw_png).hexdigest()[:16]

    def add(
        self,
        raw_png: bytes,
        resized_h: int,
        resized_w: int,
        num_placeholder_tokens: int,
    ) -> str:
        """Insert (or find) an image at a specific resolution. Returns image_id."""
        chash = self._content_hash(raw_png)
        image_id = f"{chash}_{resized_h}x{resized_w}"
        if image_id not in self._by_key:
            self._by_key[image_id] = PooledImage(
                image_id=image_id,
                raw_png=raw_png,
                resized_h=resized_h,
                resized_w=resized_w,
                num_placeholder_tokens=num_placeholder_tokens,
            )
        return image_id

    def get(self, image_id: str) -> PooledImage:
        return self._by_key[image_id]

    def __len__(self) -> int:
        return len(self._by_key)

    def total_bytes(self) -> int:
        return sum(len(p.raw_png) for p in self._by_key.values())


# ---------------------------------------------------------------------------
# Step record  (one training sample)
# ---------------------------------------------------------------------------


@dataclass
class StepRecord:
    """One turn = one independent single-turn training sample.

    The training forward sequence is:  prompt_input_ids + output_ids
    loss_mask covers ONLY the output_ids region.
    """

    # --- the invariant: ids exactly as the rollout engine saw / produced them
    prompt_input_ids: list[int]          # windowed prompt, tokenized ONCE at rollout
    output_ids: list[int]                # generated tokens, original ids preserved

    # --- alignment metadata, computed at rollout time, reused verbatim at train
    # mrope 3D positions for the FULL sequence (prompt+output). Shape [3, T].
    # Stored because recomputing requires re-deriving image grid layout; storing
    # avoids any divergence.
    position_ids: Optional[np.ndarray] = None

    # --- images live in the pool; the step only references them, in order
    image_ids: list[str] = field(default_factory=list)

    # --- learning signals
    reward: float = 0.0
    advantage: float = 0.0

    # --- for backend-mismatch (rollout vs train logprob) importance sampling.
    # rollout_logprobs[i] is logprob of output_ids[i] under the rollout policy.
    rollout_logprobs: Optional[list[float]] = None

    # --- bookkeeping
    turn_index: int = 0
    trajectory_id: str = ""

    def __post_init__(self) -> None:
        if self.rollout_logprobs is not None:
            assert len(self.rollout_logprobs) == len(self.output_ids), (
                f"rollout_logprobs ({len(self.rollout_logprobs)}) must align "
                f"with output_ids ({len(self.output_ids)})"
            )

    @property
    def full_ids(self) -> list[int]:
        return self.prompt_input_ids + self.output_ids

    def build_loss_mask(self) -> list[int]:
        """1 on output tokens, 0 on prompt (incl. image placeholders)."""
        return [0] * len(self.prompt_input_ids) + [1] * len(self.output_ids)


@dataclass
class TrajectoryRecord:
    """All steps of one trajectory + its shared image pool."""

    trajectory_id: str
    steps: list[StepRecord] = field(default_factory=list)
    image_pool: ImagePool = field(default_factory=ImagePool)
    task_success: bool = False  # episode-level signal, e.g. for GRPO grouping
    meta: dict = field(default_factory=dict)

    def add_step(self, step: StepRecord) -> None:
        step.trajectory_id = self.trajectory_id
        step.turn_index = len(self.steps)
        self.steps.append(step)
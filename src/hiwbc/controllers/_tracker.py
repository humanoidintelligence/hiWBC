"""Shared base for the reference-motion trackers (Twist / BeyondMimic / ProtoMotions / KungfuBot).

These share ASAP's shape — proprioception plus reference-motion terms — but the
reference is *per-tick future-motion data*, not a phase scalar. In RoboJuDo that
data is produced by a motion Controller and read from ``ctrl_data``; in hiWBC it
arrives in ``reference.frames``, so motion playback and any body-frame transforms
stay upstream, consistent with "reference in, joint targets out".

Each tracker replicates the exact observation field order of its RoboJuDo
policy. Field order is safety-critical. Most apply no action EMA
(``action_beta=1.0``): ``last_action`` is the raw model output.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence

import numpy as np

from ..controller import Controller
from ..types import ReferenceKind

GRAVITY_VEC = np.array([0.0, 0.0, -1.0], dtype=np.float32)


def require_frames(frames: dict | None, *keys: str) -> tuple[np.ndarray, ...]:
    if frames is None:
        raise ValueError("MOTION_FRAMES reference requires frames data")
    missing = [k for k in keys if k not in frames]
    if missing:
        raise ValueError(f"reference.frames missing keys {missing}")
    return tuple(np.asarray(frames[k], dtype=np.float32).reshape(-1) for k in keys)


class TrackerBase(Controller):
    """Common history buffer + no-EMA setup for trackers."""

    accepts = (ReferenceKind.MOTION_FRAMES,)

    def __init__(
        self,
        *,
        num_actions: int,
        default_pos: Sequence[float],
        forward_fn: Callable,
        history_length: int = 0,
        action_scale: float | Sequence[float] = 1.0,
        action_clip: float | None = None,
        action_beta: float = 1.0,
        **base,
    ) -> None:
        # Most trackers apply no EMA (beta=1.0); H2H student uses 0.8.
        super().__init__(
            num_actions=num_actions,
            default_pos=default_pos,
            action_scale=action_scale,
            action_clip=action_clip,
            action_beta=action_beta,
            **base,
        )
        self._forward_fn = forward_fn
        self.history_length = history_length
        self._history: deque[np.ndarray] = deque(maxlen=history_length)

    def _reset_history(self, block_size: int) -> None:
        self._history = deque(
            [np.zeros(block_size, dtype=np.float32)] * self.history_length,
            maxlen=self.history_length,
        )

    def _history_flat(self, block_size: int) -> np.ndarray:
        if not self.history_length:
            return np.zeros(0, dtype=np.float32)
        if not self._history:
            return np.zeros(block_size * self.history_length, dtype=np.float32)
        return np.concatenate(list(self._history)).astype(np.float32)

    def forward(self, obs):
        return self._forward_fn(obs)


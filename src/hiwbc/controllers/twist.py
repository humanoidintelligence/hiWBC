"""Twist motion tracker (RoboJuDo ``TwistPolicy``). See :mod:`._tracker` for the shared setup."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..math_utils import quat_rotate_inverse, quat_to_euler
from ..references.clip import require_clip
from ..registry import register_controller
from ..types import ControllerState, Reference, ReferenceKind
from ._tracker import TrackerBase, require_frames


class TwistTrackerController(TrackerBase):
    """RoboJuDo ``TwistPolicy``: obs = [mimic_target, proprio], + flattened history.

    ``proprio`` = ang_vel, roll_pitch, dof_pos - default, dof_vel(ankles zeroed
    upstream), last_action. The upper-body mimic target comes in
    ``reference.frames['mimic_target']``.
    """

    def __init__(self, *, ang_vel_scale=1.0, dof_pos_scale=1.0, dof_vel_scale=1.0,
                 zero_dof_vel_mask: Sequence[float] | None = None, **kw):
        super().__init__(**kw)
        self.ang_vel_scale = ang_vel_scale
        self.dof_pos_scale = dof_pos_scale
        self.dof_vel_scale = dof_vel_scale
        # RoboJuDo zeroes the ankle joint velocities (``ankle_idx``) in the obs.
        self._vel_keep = (
            None if zero_dof_vel_mask is None
            else 1.0 - np.asarray(zero_dof_vel_mask, dtype=np.float32)
        )

    # ----- motion clips: mimic_target from the current frame (RoboJuDo MotionTwistCtrl) ---
    @property
    def motion_spaces(self) -> tuple[str, ...]:
        return ("joint",) if self.joint_names is not None else ()

    def check_clip(self, player) -> None:
        super().check_clip(player)
        require_clip(player, "TWIST", joint_names=self.joint_names, root_pos=True, root_quat=True)

    def clip_reference(self, player, state: ControllerState) -> Reference:
        win = player.window(1, state.base_quat)
        q = win["root_quat"][0]
        dof = player.joints(win["joint_pos"], self.joint_names, self.default_pos)[0]
        mimic = np.concatenate([
            win["root_z"][:1], quat_to_euler(q),
            quat_rotate_inverse(q, win["root_lin_vel"][0]),
            quat_rotate_inverse(q, win["root_ang_vel"][0])[2:3], dof,
        ]).astype(np.float32)
        return Reference(ReferenceKind.MOTION_FRAMES, frames={"mimic_target": mimic})

    def observe(self, state: ControllerState, reference: Reference) -> np.ndarray:
        (mimic_target,) = require_frames(reference.frames, "mimic_target")
        rpy = quat_to_euler(state.base_quat)
        dof_vel = state.dof_vel if self._vel_keep is None else state.dof_vel * self._vel_keep
        proprio = np.concatenate(
            [
                state.base_ang_vel * self.ang_vel_scale,
                rpy[:2],
                (state.dof_pos - self.default_pos) * self.dof_pos_scale,
                dof_vel * self.dof_vel_scale,
                self._last_action,
            ]
        ).astype(np.float32)
        full = np.concatenate([mimic_target, proprio]).astype(np.float32)
        if self.history_length and (not self._history or self._history[0].size != full.size):
            # RoboJuDo seeds the buffer with zeros; history is oldest-first, frame-major.
            self._reset_history(full.size)
        obs = np.concatenate([full, self._history_flat(full.size)]).astype(np.float32)
        if self.history_length:
            self._history.append(full)
        return obs


@register_controller("twist_tracker")
def _build_twist(**kw) -> TwistTrackerController:
    return TwistTrackerController(**kw)

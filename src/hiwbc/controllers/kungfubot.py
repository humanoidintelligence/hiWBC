"""KungfuBot motion tracker (RoboJuDo ``KungfuBotGeneralPolicy``). See :mod:`._tracker`."""

from __future__ import annotations

from collections import deque
from typing import ClassVar

import numpy as np

from ..math_utils import matrix_from_quat, quat_to_euler, subtract_frame_transforms_rot
from ..references.clip import require_clip
from ..registry import register_controller
from ..types import ControllerState, Reference, ReferenceKind
from ._bodyfk import KeyPoints, conj, rotate
from ._tracker import TrackerBase, require_frames

_KEY = "kungfubot_key_points"


class KungfuBotTrackerController(TrackerBase):
    """RoboJuDo ``KungfuBotGeneralPolicy``: a multi-input model.

    ``observe`` returns a tuple ``(actor_obs, future_motion_targets, history)``.
    ``actor_obs`` = action, anchor_ref_rot, ang_vel, dof_pos - default, dof_vel,
    history, next_step_ref_motion, roll_pitch (all sorted by name, per the
    reference). The anchor rotation and future-motion terms come from
    ``reference.frames``.
    """

    # ----- motion clips (RoboJuDo ``MotionKungfuBotCtrl``) ---------------------------
    motion_spaces = ("joint",)
    #: PHC's motion library differentiates root velocities at the clip's rate and
    #: smooths them (sigma = 2 frames); the checkpoint was trained on that.
    clip_player_options: ClassVar[dict] = {"source_velocity_sigma": 2.0}

    def __init__(self, *, ang_vel_scale=1.0, dof_pos_scale=1.0, dof_vel_scale=1.0,
                 roll_pitch_scale=1.0, actions_scale=1.0, history_scale=1.0,
                 future_steps=None, kinematics=None, key_bodies=(),
                 pbhc_anchor_quat_order=False, **kw):
        """Clip options (all needed for ``clip_reference``):

        Args:
            future_steps: Frame offsets of the future-motion window
                (RoboJuDo: ``linspace(1, 95, 20)``, truncated to int).
            kinematics: A ``hirobot.geometry.kinematics.Kinematics`` of the robot (or a
                zero-argument callable returning one, loaded on first use).
            key_bodies: ``(body, offset)`` pairs: key points at ``offset`` (body
                frame, metres) from each body, for ``next_step_ref_motion``.
            pbhc_anchor_quat_order: Reorder the relative anchor quaternion as PBHC
                did before its fix (RoboJuDo ``compatibility_old_version``,
                https://github.com/TeleHuman/PBHC/issues/68). Checkpoints trained
                with the bug need it.
        """
        super().__init__(**kw)
        self.future_steps = None if future_steps is None else np.asarray(future_steps, dtype=int)
        self.pbhc_anchor_quat_order = bool(pbhc_anchor_quat_order)
        self.key_points = (KeyPoints(kinematics, key_bodies, self.joint_names)
                           if kinematics is not None and key_bodies else None)
        self.ang_vel_scale = ang_vel_scale
        self.dof_pos_scale = dof_pos_scale
        self.dof_vel_scale = dof_vel_scale
        self.roll_pitch_scale = roll_pitch_scale
        self.actions_scale = actions_scale
        self.history_scale = history_scale

    def check_clip(self, player) -> None:
        super().check_clip(player)
        if self.future_steps is None or self.key_points is None:
            raise ValueError("KungfuBot needs future_steps, kinematics and key_bodies to "
                             "track clips (the unitree_g1 profile sets them)")
        require_clip(player, "KungfuBot", joint_names=self.joint_names, root_pos=True,
                     root_quat=True)
        if not player.has_derived(_KEY):
            player.add_derived(_KEY, self.key_points.clip_term(
                player.joint_names, self.default_pos, with_velocity=False))

    def clip_reference(self, player, state: ControllerState) -> Reference:
        """Reference terms of RoboJuDo ``MotionKungfuBotCtrl`` from the player's window.

        The window starts at the current frame (RoboJuDo lags one frame: it computes
        tick t's window before advancing). Roll / pitch, root height, body-frame
        root velocities and joint angles for each future step; ``next_step_ref_motion``
        is the first step's terms plus the key points in the anchor (root) frame,
        from forward kinematics at the clip's own rate (as PHC; see ``check_clip``);
        ``anchor_ref_rot`` is the first two columns of the anchor rotation relative
        to the robot's base.
        """
        steps = self.future_steps
        win = player.window(int(steps.max()) + 1, state.base_quat)
        quat, pos = win["root_quat"][steps], win["root_pos"][steps]
        dof = player.joints(win["joint_pos"], self.joint_names, self.default_pos)[steps]
        rp = np.stack([quat_to_euler(q)[:2] for q in quat])
        lin_vel = rotate(conj(quat), win["root_lin_vel"][steps])
        yaw_vel = rotate(conj(quat), win["root_ang_vel"][steps])[:, 2]
        z = win["root_z"][steps]

        pts = win[_KEY][steps[0]].reshape(-1, 3)
        key = rotate(conj(quat[0]), pts - pos[0]).reshape(-1)
        next_step = np.concatenate([z[:1], rp[0], lin_vel[0], yaw_vel[:1], dof[0], key])

        rel = subtract_frame_transforms_rot(np.asarray(state.base_quat, np.float32),
                                            quat[0].astype(np.float32))
        if self.pbhc_anchor_quat_order:
            rel = rel[[1, 2, 3, 0]]
        anchor = matrix_from_quat(rel)[:, :2].reshape(-1)
        f32 = np.float32
        return Reference(ReferenceKind.MOTION_FRAMES, frames={
            "anchor_ref_rot": anchor.astype(f32),
            "next_step_ref_motion": next_step.astype(f32),
            "future_motion_base_lin_vel": lin_vel.reshape(-1).astype(f32),
            "future_motion_base_yaw_vel": yaw_vel.astype(f32),
            "future_motion_dof_pos": dof.reshape(-1).astype(f32),
            "future_motion_roll_pitch": rp.reshape(-1).astype(f32),
            "future_motion_root_height": z.astype(f32),
        })

    def observe(self, state: ControllerState, reference: Reference):
        anchor_ref_rot, next_step_ref_motion = require_frames(
            reference.frames, "anchor_ref_rot", "next_step_ref_motion"
        )
        fut_lin_vel, fut_yaw_vel, fut_dof_pos, fut_roll_pitch, fut_root_height = require_frames(
            reference.frames,
            "future_motion_base_lin_vel",
            "future_motion_base_yaw_vel",
            "future_motion_dof_pos",
            "future_motion_roll_pitch",
            "future_motion_root_height",
        )
        rpy = quat_to_euler(state.base_quat)
        roll_pitch = rpy[:2] * self.roll_pitch_scale
        action = self._last_action * self.actions_scale
        ang_vel = state.base_ang_vel * self.ang_vel_scale
        dof_pos_rel = (state.dof_pos - self.default_pos) * self.dof_pos_scale
        dof_vel = state.dof_vel * self.dof_vel_scale

        # history entry: action, ang_vel, dof_pos (abs), dof_vel, roll_pitch.
        # Flattened term-major, newest first (RoboJuDo ``_get_obs_history``).
        block = [np.asarray(t, dtype=np.float32) for t in
                 (action, ang_vel, state.dof_pos * self.dof_pos_scale, dof_vel, roll_pitch)]
        if self.history_length and not self._history:
            self._history = deque([[np.zeros_like(t) for t in block]] * self.history_length,
                                  maxlen=self.history_length)
        history = (
            np.concatenate([np.concatenate(items) for items in zip(*self._history, strict=True)])
            if self.history_length else np.zeros(0, dtype=np.float32)
        ).astype(np.float32) * self.history_scale

        actor_obs = np.concatenate(
            [action, anchor_ref_rot, ang_vel, dof_pos_rel, dof_vel, history,
             next_step_ref_motion, roll_pitch]
        ).astype(np.float32)
        future_motion_targets = np.concatenate(
            [fut_lin_vel, fut_yaw_vel, fut_dof_pos, fut_roll_pitch, fut_root_height]
        ).astype(np.float32)

        if self.history_length:
            self._history.appendleft(block)
        return (actor_obs, future_motion_targets, history)

    def reset(self) -> None:
        super().reset()
        self._history = deque(maxlen=self.history_length)


@register_controller("kungfubot_tracker")
def _build_kungfubot(**kw) -> KungfuBotTrackerController:
    return KungfuBotTrackerController(**kw)

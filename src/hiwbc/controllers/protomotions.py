"""ProtoMotions motion tracker (RoboJuDo ``ProtoMotionsTrackerPolicy``). See :mod:`._tracker`."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..math_utils import quat_rotate_inverse
from ..references.clip import require_clip
from ..registry import register_controller
from ..types import ControllerState, Reference, ReferenceKind
from ._bodyfk import KeyPoints, rotate
from ._tracker import GRAVITY_VEC, TrackerBase, require_frames

_TRACK = "h2h_track_points"


def _heading_inv(q: np.ndarray) -> np.ndarray:
    """Inverse of the yaw of ``q`` taken from its x axis (RoboJuDo ``calc_heading_quat_inv_np``)."""
    x = rotate(q, np.array([1.0, 0.0, 0.0]))
    yaw = np.arctan2(x[1], x[0])
    return np.array([0.0, 0.0, -np.sin(yaw / 2), np.cos(yaw / 2)])


class ProtoMotionsTrackerController(TrackerBase):
    """RoboJuDo ``ProtoMotionsTrackerPolicy`` obs order.

    obs = dof_pos (or offset), dof_vel, base_ang_vel, projected_gravity,
    task_obs, last_action, history. ``task_obs`` is the imitation observation
    computed upstream and arrives in ``reference.frames['task_obs']``;
    :meth:`clip_reference` computes it from a clip for H2H (RoboJuDo
    ``MotionH2HCtrl`` + ``H2HStudentPolicy``).
    """

    motion_spaces: tuple[str, ...] = ()

    def __init__(self, *, dof_vel_scale=1.0, ang_vel_scale=1.0, use_dof_pos_offset=False,
                 kinematics=None, track_points=(), kinematic_joint_names: Sequence[str] = (),
                 max_head_distance: float | None = 1.0, **kw):
        """Clip options (``clip_reference``, H2H):

        Args:
            kinematics: A ``hirobot.geometry.kinematics.Kinematics`` (or a zero-argument
                callable returning one). Without it the controller takes no clips.
            track_points: ``(body, offset)`` pairs tracked in ``task_obs``; the last
                is the head.
            kinematic_joint_names: The model's joints, for FK of clip poses (which
                may include joints the policy does not drive).
            max_head_distance: The reference head is pulled to within this
                horizontal distance (m) of the robot (RoboJuDo); ``None`` disables.
        """
        super().__init__(**kw)
        if kinematics is not None and track_points:
            self.motion_spaces = ("joint",)
            names = tuple(kinematic_joint_names) or self.joint_names
            self.track_points = KeyPoints(kinematics, track_points, names)
        else:
            self.track_points = None
        self.max_head_distance = max_head_distance
        self.dof_vel_scale = dof_vel_scale
        self.ang_vel_scale = ang_vel_scale
        self.use_dof_pos_offset = use_dof_pos_offset

    # ----- motion clips ------------------------------------------------------------
    def check_clip(self, player) -> None:
        super().check_clip(player)
        require_clip(player, "H2H", joint_names=self.joint_names, root_pos=True, root_quat=True)
        if not player.has_derived(_TRACK):
            kp = self.track_points
            fill = np.zeros(len(kp.joint_names))
            own = {n: i for i, n in enumerate(self.joint_names)}
            for j, n in enumerate(kp.joint_names):
                if n in own:
                    fill[j] = self.default_pos[own[n]]
            player.add_derived(_TRACK, kp.clip_term(player.joint_names, fill, with_velocity=True))

    @staticmethod
    def _base_pos(state: ControllerState) -> np.ndarray:
        base_pos = state.extra.get("base_pos")
        if base_pos is None:
            raise ValueError("H2H needs the robot's world position: state.extra['base_pos']")
        return np.asarray(base_pos, np.float64)

    def robot_points(self, state: ControllerState) -> np.ndarray:
        """The robot's tracked points (world), FK of ``state.dof_pos`` at its base pose.

        Needs ``state.extra["base_pos"]`` (world position). Joints the policy does
        not drive (e.g. wrists) are at 0, as in RoboJuDo's FK.
        """
        base_pos = self._base_pos(state)
        kp = self.track_points
        q = np.zeros(len(kp.joint_names))
        col = {n: i for i, n in enumerate(kp.joint_names)}
        for i, n in enumerate(self.joint_names):
            q[col[n]] = state.dof_pos[i]
        return kp.positions(q, base_pos, np.asarray(state.base_quat, np.float64))

    def clip_reference(self, player, state: ControllerState) -> Reference:
        """H2H ``task_obs`` (``compute_imitation_observations_teleop_max``) from a clip.

        Per tracked point, in the robot's heading frame: reference minus robot
        position, reference minus robot base, reference velocity. Reference points
        come from FK at the clip's own rate, velocities Gaussian-smoothed as PHC.
        The clip starts under the robot (``state.extra["base_pos"]``).
        """
        base_pos = self._base_pos(state)  # before window(): the first call anchors the clip
        win = player.window(1, state.base_quat, base_pos)
        p = len(self.track_points.bodies)
        ref = win[_TRACK][0]
        ref_pos, ref_vel = ref[:3 * p].reshape(p, 3).copy(), ref[3 * p:].reshape(p, 3)
        body = self.robot_points(state)
        if self.max_head_distance is not None:
            d = base_pos[:2] - ref_pos[-1, :2]
            dist = float(np.linalg.norm(d))
            if dist > self.max_head_distance:
                ref_pos[-1, :2] = base_pos[:2] - d / dist * self.max_head_distance
        h = _heading_inv(np.asarray(state.base_quat, np.float64))
        task = np.concatenate([rotate(h, ref_pos - body).reshape(-1),
                               rotate(h, ref_pos - base_pos).reshape(-1),
                               rotate(h, ref_vel).reshape(-1)]).astype(np.float32)
        return Reference(ReferenceKind.MOTION_FRAMES, frames={"task_obs": task})

    def observe(self, state: ControllerState, reference: Reference) -> np.ndarray:
        (task_obs,) = require_frames(reference.frames, "task_obs")
        base_gravity = quat_rotate_inverse(state.base_quat, GRAVITY_VEC)
        dof_pos = state.dof_pos - self.default_pos if self.use_dof_pos_offset else state.dof_pos
        # history block: dof_pos, dof_vel, ang_vel, gravity, last_action (unscaled)
        block = np.concatenate(
            [state.dof_pos, state.dof_vel, state.base_ang_vel, base_gravity, self._last_action]
        ).astype(np.float32)
        obs = np.concatenate(
            [
                dof_pos,
                state.dof_vel * self.dof_vel_scale,
                state.base_ang_vel * self.ang_vel_scale,
                base_gravity,
                task_obs,
                self._last_action,
                self._history_flat(block.size),
            ]
        ).astype(np.float32)
        if self.history_length:
            if not self._history or self._history[0].size != block.size:
                self._reset_history(block.size)
            self._history.appendleft(block)
        return obs


@register_controller("protomotions_tracker")
def _build_protomotions(**kw) -> ProtoMotionsTrackerController:
    return ProtoMotionsTrackerController(**kw)

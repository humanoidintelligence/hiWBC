"""BeyondMimic motion tracker (RoboJuDo ``BeyondMimicPolicy``). See :mod:`._tracker`.

Each released checkpoint is trained on one motion, and its ONNX export carries
that motion: given ``time_step`` it also outputs the reference ``joint_pos``,
``joint_vel`` and body poses for that frame. With a ``motion_fn`` the controller
exposes the motion as a behavior and plays it the way RoboJuDo's
``use_motion_from_model`` mode does:

* starting runs one forward on a zero observation at frame 0, which gives the
  first reference frame and the initial ``last_action``;
* each tick the observation uses the reference from the previous forward, and
  the forward runs at the current frame (so tick 0 and tick 1 both read frame 0);
* the motion's first anchor (``torso_link``) heading is aligned to the robot's
  base heading at the first tick, yaw only;
* the robot anchor position is taken equal to the reference's
  (``override_robot_anchor_pos``), so ``motion_anchor_pos_b`` is zero: hiWBC has
  no torso position estimate.

The robot anchor orientation is the torso's: ``state.extra["torso_quat"]`` if
given, else the base orientation turned by the waist yaw, roll and pitch joints
(the G1 kinematic chain pelvis -> waist_yaw -> waist_roll -> torso).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..math_utils import matrix_from_quat, quat_mul, subtract_frame_transforms_rot
from ..registry import register_controller
from ..types import BehaviorSpec, ControllerState, Reference, ReferenceKind, ReferenceSource
from ._tracker import TrackerBase, require_frames

_WAIST = (("waist_yaw", 2), ("waist_roll", 0), ("waist_pitch", 1))  # joint, axis


def _axis_quat(axis: int, angle: float) -> np.ndarray:
    q = np.zeros(4)
    q[axis], q[3] = np.sin(angle / 2), np.cos(angle / 2)
    return q


def _yaw(q: np.ndarray) -> float:
    """Heading of a [x, y, z, w] quaternion (z of extrinsic xyz Euler angles)."""
    x, y, z, w = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


class BeyondMimicTrackerController(TrackerBase):
    """RoboJuDo ``BeyondMimicPolicy`` obs order.

    obs = command, [motion_anchor_pos_b], motion_anchor_ori_b, [base_lin_vel],
    base_ang_vel, dof_pos - default, dof_vel, last_action.
    Bracketed terms are dropped when ``without_state_estimator`` is set. The
    anchor pos/ori and command come from ``reference.frames``.

    Args:
        motion_fn: ``motion_fn(obs, time_step) -> dict`` with the model's
            ``actions``, ``joint_pos``, ``joint_vel`` and ``body_quat_w``
            ([w, x, y, z]) outputs. Enables the behavior; ``forward_fn`` is still
            used for references that do not come from it.
        motion_name: Behavior name of the baked-in motion.
        motion_frames: Frames in the motion (the model holds the last one after).
        anchor_body_index: Index of the anchor body in ``body_quat_w``.
    """

    def __init__(self, *, without_state_estimator: bool = False,
                 motion_fn: Callable[[np.ndarray, int], dict] | None = None,
                 motion_name: str | None = None, motion_frames: int | None = None,
                 anchor_body_index: int | None = None, **kw):
        super().__init__(**kw)
        self.without_state_estimator = without_state_estimator
        self._motion_fn = motion_fn
        self.motion_name = motion_name
        self.motion_frames = motion_frames
        self.anchor_body_index = anchor_body_index
        if motion_fn is not None and (not motion_name or not motion_frames or anchor_body_index is None):
            raise ValueError("motion_fn needs motion_name, motion_frames and anchor_body_index")
        names = self.joint_names or ()
        self._waist = [(names.index(j), a) for j, a in _WAIST if j in names]
        self._time_step: int | None = None   # frame of the forward running this tick
        self._motion_out: dict | None = None  # outputs of the last motion forward
        self.start_pos = None
        if motion_fn is not None:
            first = motion_fn(np.zeros(self.obs_dim, np.float32), 0)
            self.start_pos = np.asarray(first["joint_pos"], np.float32).reshape(-1)

    @property
    def obs_dim(self) -> int:
        n = self.num_actions
        return 2 * n + 6 + 3 + 3 * n + (0 if self.without_state_estimator else 6)

    def observe(self, state: ControllerState, reference: Reference) -> np.ndarray:
        keys = ["command", "motion_anchor_ori_b"]
        if not self.without_state_estimator:
            keys = ["command", "motion_anchor_pos_b", "motion_anchor_ori_b"]
        parts = list(require_frames(reference.frames, *keys))
        # insert in the policy's field order
        if self.without_state_estimator:
            command, anchor_ori = parts
            head = [command, anchor_ori]
        else:
            command, anchor_pos, anchor_ori = parts
            head = [command, anchor_pos, anchor_ori]

        if self.without_state_estimator:
            lin_vel_term: list = []
        else:
            (base_lin_vel,) = require_frames(reference.frames, "base_lin_vel")
            lin_vel_term = [base_lin_vel]

        obs = np.concatenate(
            [
                *head,
                *lin_vel_term,
                state.base_ang_vel,
                state.dof_pos - self.default_pos,
                state.dof_vel,
                self._last_action,
            ]
        ).astype(np.float32)
        return obs

    def forward(self, obs):
        if self._time_step is None:
            return self._forward_fn(obs)
        self._motion_out = self._motion_fn(obs, self._time_step)
        return self._motion_out["actions"]

    def step(self, state, reference):
        self._time_step = reference.extra.get("time_step") if self._motion_fn else None
        try:
            return super().step(state, reference)
        finally:
            self._time_step = None

    def reset(self) -> None:
        super().reset()
        self._motion_out = None

    # ----- behavior: the motion baked into the checkpoint ---------------------
    @property
    def behaviors(self) -> list[BehaviorSpec]:
        if self._motion_fn is None:
            return []
        needs = "" if self.without_state_estimator else " Needs base_lin_vel in the state."
        return [BehaviorSpec(
            name=self.motion_name,
            description=(f"Play the motion trained into this checkpoint "
                         f"({(self.motion_frames - 1) * self.control_dt:g} s).{needs}"),
            start_pos=tuple(float(x) for x in self.start_pos),
        )]

    def start_behavior(self, name: str, **args) -> ReferenceSource:
        if self._motion_fn is None or name != self.motion_name:
            return super().start_behavior(name, **args)
        if args:
            raise TypeError(f"behavior {name!r} takes no arguments, got {sorted(args)}")
        self.reset()
        # RoboJuDo's warm-up forward (_prepare_policy): zero obs at frame 0.
        self._motion_out = self._motion_fn(np.zeros(self.obs_dim, np.float32), 0)
        self._last_action = np.asarray(self._motion_out["actions"], np.float32).reshape(-1).copy()
        return _BakedMotion(self)

    def torso_quat(self, state: ControllerState) -> np.ndarray:
        """Robot anchor orientation [x, y, z, w]: ``extra["torso_quat"]`` or base + waist joints."""
        q = state.extra.get("torso_quat")
        if q is not None:
            return np.asarray(q, np.float64)
        if len(self._waist) != 3:
            raise ValueError("torso orientation needs state.extra['torso_quat'] or the three waist joints")
        q = np.asarray(state.base_quat, np.float64)
        for idx, axis in self._waist:
            q = quat_mul(q, _axis_quat(axis, float(state.dof_pos[idx])))
        return np.asarray(q, np.float64)


class _BakedMotion:
    """Feeds the checkpoint's own reference back to it, one frame per tick."""

    def __init__(self, ctrl: BeyondMimicTrackerController) -> None:
        self._ctrl = ctrl
        self._t = 0
        self._align: np.ndarray | None = None

    @property
    def done(self) -> bool:
        return self._t >= self._ctrl.motion_frames

    def next(self, state: ControllerState) -> Reference:
        c = self._ctrl
        out = c._motion_out
        anchor = np.asarray(out["body_quat_w"], np.float64).reshape(-1, 4)[c.anchor_body_index][[1, 2, 3, 0]]
        if self._align is None:
            # Turn the motion's first heading onto the robot's base heading, yaw only.
            self._align = _axis_quat(2, _yaw(np.asarray(state.base_quat, np.float64)) - _yaw(anchor))
        anchor = quat_mul(self._align, anchor)
        ori = subtract_frame_transforms_rot(c.torso_quat(state), anchor)
        frames = {
            "command": np.concatenate([np.asarray(out["joint_pos"]).reshape(-1),
                                       np.asarray(out["joint_vel"]).reshape(-1)]).astype(np.float32),
            "motion_anchor_ori_b": matrix_from_quat(ori)[:, :2].reshape(-1),
        }
        if not c.without_state_estimator:
            if state.base_lin_vel is None:
                raise ValueError(f"behavior {c.motion_name!r} needs base_lin_vel in the state")
            frames["motion_anchor_pos_b"] = np.zeros(3, np.float32)
            frames["base_lin_vel"] = np.asarray(state.base_lin_vel, np.float32)
        t = min(self._t, c.motion_frames - 1)
        self._t += 1
        return Reference(ReferenceKind.MOTION_FRAMES, frames=frames, extra={"time_step": t})


@register_controller("beyondmimic_tracker")
def _build_beyondmimic(**kw) -> BeyondMimicTrackerController:
    return BeyondMimicTrackerController(**kw)

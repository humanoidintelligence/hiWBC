"""SuperMimic motion tracker (Unitree G1, 29 DoF), unofficial reproduction.

SuperMimic (arXiv 2610.28480) has no official code or weights. This controller
replicates the inference interface of `KyrieYoung926/supermimic_deploy
<https://github.com/KyrieYoung926/supermimic_deploy>`_ (MIT), whose ONNX policy
was trained with `SuperMimic-Reproduced
<https://github.com/CuboiLeo/SuperMimic-Reproduced>`_ (MIT). The source of truth
is ``controller.py`` (``tick`` / ``flatten_history``) and
``scripts/prepare_motions.py`` in that repo. Other reproductions use different,
incompatible observation layouts and are not covered.

The model takes two inputs:

- ``obs_history`` (930,): 10 ticks of a 93-d proprio entry
  ``[q - q0 (29), dq (29), base ang vel (3), projected gravity (3), prev action (29)]``,
  laid out **term-major**: all 10 ``q - q0`` blocks, then all 10 ``dq`` blocks,
  and so on, oldest first within each term. On the first tick after
  :meth:`reset` the whole history is filled with the current entry.
- ``goal`` (680,): 10 reference frames ``t .. t+9`` of a 68-d goal feature
  (see :func:`goal_features`), frame-major. A short window holds its last frame.

The action maps to ``q0 + scale * action`` with per-joint ``scale``; no EMA or
clip. Joint order is the Unitree SDK / MuJoCo order of
:data:`SUPERMIMIC_JOINT_NAMES`. Per-joint ``q0`` / ``scale`` / ``kp`` / ``kd`` /
``effort`` come from the bundle's ``config.json`` (:meth:`from_bundle`).

Reference input: ``ReferenceKind.MOTION_FRAMES`` with ``frames['goal']`` of
shape ``(T, 68)``, frame 0 being the current tick at 50 Hz. Motion playback and
goal-feature computation stay upstream.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from ..controller import Controller
from ..math_utils import gravity_orientation, quat_rotate_inverse
from ..references.clip import require_clip
from ..registry import register_controller
from ..runtime import load_onnx_runner as _load_onnx
from ..types import ControllerState, JointTarget, Reference, ReferenceKind

NUM_JOINTS = 29
HISTORY_LEN = 10
FUTURE_FRAMES = 10
GOAL_DIM = 68
OBS_DIM = 3 * NUM_JOINTS + 6  # 93
# Term boundaries inside one 93-d proprio entry (flatten_history in controller.py).
_OBS_TERMS = ((0, 29), (29, 58), (58, 61), (61, 64), (64, 93))

#: Joint order of the policy, hiRobot's names (the bundle's ``config.json`` keeps
#: the ``_joint`` suffix; :meth:`SuperMimicController.from_bundle` drops it to compare).
SUPERMIMIC_JOINT_NAMES = (
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw",
    "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw",
    "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
)


def goal_features(
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    root_quat: np.ndarray,
    root_lin_vel_w: np.ndarray,
    root_ang_vel_w: np.ndarray,
    root_z: np.ndarray,
    default_pos: Sequence[float],
) -> np.ndarray:
    """Per-frame 68-d goal features from a 50 Hz reference motion.

    Mirrors ``prepare_motions.py``: ``[q - q0 (29), dq (29), projected gravity (3),
    root ang vel in the root frame (3), root lin vel in the heading frame (3),
    root height (1)]``. ``root_quat`` is ``(T, 4)`` [x, y, z, w]; velocities are
    world frame. Returns ``(T, 68)`` float32.
    """
    q = np.asarray(root_quat, dtype=np.float64).reshape(-1, 4)
    jp = np.asarray(joint_pos, dtype=np.float32).reshape(len(q), NUM_JOINTS)
    jv = np.asarray(joint_vel, dtype=np.float32).reshape(len(q), NUM_JOINTS)
    lin = np.asarray(root_lin_vel_w, dtype=np.float64).reshape(len(q), 3)
    ang = np.asarray(root_ang_vel_w, dtype=np.float64).reshape(len(q), 3)
    z = np.asarray(root_z, dtype=np.float32).reshape(len(q), 1)
    grav = np.stack([quat_rotate_inverse(qi, np.array([0.0, 0.0, -1.0])) for qi in q])
    ang_b = np.stack([quat_rotate_inverse(qi, wi) for qi, wi in zip(q, ang)])
    x, y, zq, w = q.T
    yaw = np.arctan2(2 * (w * zq + x * y), 1 - 2 * (y * y + zq * zq))
    c, s = np.cos(yaw), np.sin(yaw)
    lin_h = np.stack([c * lin[:, 0] + s * lin[:, 1], -s * lin[:, 0] + c * lin[:, 1], lin[:, 2]], 1)
    q0 = np.asarray(default_pos, dtype=np.float32)
    return np.concatenate([jp - q0, jv, grav, ang_b, lin_h, z], axis=1).astype(np.float32)


def load_onnx_runner(path: str, *, providers: Sequence[str] | None = None) -> Callable:
    """Build a ``forward_fn((obs_history, goal)) -> action`` from the bundle's ONNX.

    A thin wrapper over :func:`hiwbc.runtime.load_onnx_runner` (one thread, as in
    supermimic_deploy); needs the ``onnx`` extra.
    """
    return _load_onnx(path, input_names=("obs_history", "goal"), providers=providers,
                      num_threads=1)


class SuperMimicController(Controller):
    """SuperMimic tracker: term-major proprio history + 10 future goal frames."""

    accepts = (ReferenceKind.MOTION_FRAMES,)

    def __init__(
        self,
        *,
        forward_fn: Callable,
        default_pos: Sequence[float],
        action_scale: Sequence[float],
        kp: Sequence[float] | None = None,
        kd: Sequence[float] | None = None,
        effort: Sequence[float] | None = None,
        bound_torque: bool = True,
        control_dt: float = 0.02,
    ) -> None:
        super().__init__(
            num_actions=NUM_JOINTS,
            default_pos=default_pos,
            action_scale=action_scale,
            action_beta=1.0,
            joint_names=SUPERMIMIC_JOINT_NAMES,
            kp=kp,
            kd=kd,
            control_dt=control_dt,
        )
        self._forward_fn = forward_fn
        self.effort = None if effort is None else np.asarray(effort, dtype=np.float32)
        if self.effort is not None and self.effort.shape != (NUM_JOINTS,):
            raise ValueError(f"effort has shape {self.effort.shape}, expected ({NUM_JOINTS},)")
        if bound_torque and (self.kp is None or self.kd is None or self.effort is None):
            raise ValueError("bound_torque needs kp, kd and effort")
        self.bound_torque = bound_torque
        self._history: deque[np.ndarray] = deque(maxlen=HISTORY_LEN)

    @classmethod
    def from_bundle(
        cls,
        assets_dir: str | Path,
        *,
        check_hash: bool = True,
        providers: Sequence[str] | None = None,
        **kwargs,
    ) -> SuperMimicController:
        """Build from a supermimic_deploy ``assets/`` dir (``config.json`` + ``policy.onnx``)."""
        assets = Path(assets_dir).expanduser()
        cfg = json.loads((assets / "config.json").read_text())
        if cfg.get("anchor_obs") or cfg.get("motor_ids") != list(range(NUM_JOINTS)):
            raise ValueError("unsupported SuperMimic bundle (anchor obs or remapped motors)")
        if tuple(n.removesuffix("_joint") for n in cfg["joint_names"]) != SUPERMIMIC_JOINT_NAMES:
            raise ValueError("bundle joint order differs from SUPERMIMIC_JOINT_NAMES")
        if (cfg.get("history"), cfg.get("future_frames")) != (HISTORY_LEN, FUTURE_FRAMES):
            raise ValueError("bundle history / future_frames differ from 10 / 10")
        onnx_path = assets / "policy.onnx"
        if check_hash:
            digest = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
            if digest != cfg["onnx_sha256"]:
                raise ValueError(f"policy.onnx sha256 {digest} != config {cfg['onnx_sha256']}")
        return cls(
            forward_fn=load_onnx_runner(onnx_path, providers=providers),
            default_pos=cfg["q0"],
            action_scale=cfg["scale"],
            kp=cfg["kp"],
            kd=cfg["kd"],
            effort=cfg["effort"],
            control_dt=cfg.get("control_dt", 0.02),
            **kwargs,
        )

    def reset(self) -> None:
        super().reset()
        self._history.clear()

    def _entry(self, state: ControllerState) -> np.ndarray:
        quat = np.asarray(state.base_quat, dtype=np.float64)
        norm = np.linalg.norm(quat)
        if not np.isfinite(norm) or not 0.5 < norm < 1.5:
            raise ValueError("invalid base quaternion")
        grav = gravity_orientation(quat / norm)
        return np.concatenate([
            state.dof_pos - self.default_pos,
            state.dof_vel,
            state.base_ang_vel,
            grav,
            self._last_action,
        ]).astype(np.float32)

    def _goal_window(self, reference: Reference) -> np.ndarray:
        frames = reference.frames
        if frames is None or "goal" not in frames:
            raise ValueError("SuperMimic needs reference.frames['goal'] of shape (T, 68)")
        goal = np.asarray(frames["goal"], dtype=np.float32)
        if goal.ndim != 2 or goal.shape[1] != GOAL_DIM or len(goal) == 0:
            raise ValueError(f"goal has shape {goal.shape}, expected (T, {GOAL_DIM})")
        idx = np.minimum(np.arange(FUTURE_FRAMES), len(goal) - 1)
        return goal[idx].reshape(-1)

    # ----- motion clips: 10 goal frames from the current one --------------------
    motion_spaces = ("joint",)

    def check_clip(self, player) -> None:
        super().check_clip(player)
        require_clip(player, "SuperMimic", joint_names=self.joint_names, root_pos=True,
                     root_quat=True)

    def clip_reference(self, player, state: ControllerState) -> Reference:
        win = player.window(FUTURE_FRAMES, state.base_quat)
        names = self.joint_names
        goal = goal_features(
            player.joints(win["joint_pos"], names, self.default_pos),
            player.joints(win["joint_vel"], names, np.zeros(NUM_JOINTS)),
            win["root_quat"], win["root_lin_vel"], win["root_ang_vel"], win["root_z"],
            self.default_pos)
        return Reference(ReferenceKind.MOTION_FRAMES, frames={"goal": goal})

    def observe(self, state: ControllerState, reference: Reference):
        goal = self._goal_window(reference)
        entry = self._entry(state)
        if not self._history:
            self._history.extend([entry] * HISTORY_LEN)
        else:
            self._history.append(entry)
        hist = np.stack(self._history)  # (10, 93), oldest first
        obs_history = np.concatenate([hist[:, lo:hi].reshape(-1) for lo, hi in _OBS_TERMS])
        return obs_history.astype(np.float32), goal

    def forward(self, obs) -> np.ndarray:
        action = np.asarray(self._forward_fn(obs), dtype=np.float32).reshape(-1)
        if not np.isfinite(action).all():
            raise ValueError("SuperMimic policy returned a non-finite action")
        return action

    def step(self, state: ControllerState, reference: Reference) -> JointTarget:
        target = super().step(state, reference)
        if self.bound_torque:
            # Low-gain setpoints act as torque proxies: re-express the target so the
            # robot's PD torque is clipped at the effort limit (bounded_target).
            q = np.asarray(state.dof_pos, dtype=np.float32)
            dq = np.asarray(state.dof_vel, dtype=np.float32)
            tau = np.clip(self.kp * (target.position - q) - self.kd * dq, -self.effort, self.effort)
            target.position = (q + (tau + self.kd * dq) / self.kp).astype(np.float32)
        return target


@register_controller("supermimic")
def _build_supermimic(**kwargs) -> SuperMimicController:
    if "assets_dir" in kwargs:
        return SuperMimicController.from_bundle(kwargs.pop("assets_dir"), **kwargs)
    return SuperMimicController(**kwargs)

"""SONIC universal-token controller (NVIDIA GEAR-SONIC).

SONIC is a two-stage policy. An encoder maps a reference (robot joint-space
motion frames, a VR 3-point teleop target, or SMPL human joints) to a 64-d
motion token; a decoder maps the token plus 10 ticks of proprioception history
to a joint action. A VLA can also emit the token directly, skipping the encoder.

The controller holds no robot constants: joint order, standing pose, gains and
the lower-body / wrist index sets come from a :class:`SonicRobotConfig`, and the
observation layout of a checkpoint from a :class:`SonicObsConfig`. The released
G1 values are in :mod:`hiwbc.configs.sonic.unitree_g1`.

The observation layout replicates the C++ deployment in NVIDIA's
GR00T-WholeBodyControl (``gear_sonic_deploy/src/g1/g1_deploy_onnx_ref``): the
observation registry in ``g1_deploy_onnx_ref.cpp``, the history ring buffer in
``state_logger.cpp``, and the action mapping in ``CreatePolicyCommand``. Each
released checkpoint fixes its term order with an ``observation_config.yaml``;
:class:`SonicObsConfig` mirrors those files. Field order is safety-critical.

Reference input
---------------
``ReferenceKind.LATENT``: ``value`` is the 64-d token (e.g. a VLA's
``motion_token``). The encoder is not run.

``ReferenceKind.MOTION_FRAMES``: ``frames`` holds a window of future reference
frames at the 50 Hz control rate, frame 0 being the current tick. The
controller samples it at each term's stride (e.g. every 5th frame) and holds the
last frame when the window is short. Motion playback, heading alignment of the
clip to the robot, and FK all stay upstream. Keys per encoder mode:

- ``joint``: ``joint_pos`` (T, n) and ``joint_vel`` (T, n) in the controller's
  joint order (:attr:`SonicRobotConfig.joint_names`);
  ``root_quat`` (T, 4) world-frame root orientation, [x, y, z, w], already
  aligned to the robot's heading; ``root_z`` (T,) for configs that use it.
- ``teleop``: ``joint_pos`` / ``joint_vel`` (lower body is selected here),
  ``root_quat``, ``vr_3point_pos`` (3, 3) and ``vr_3point_quat`` (3, 4,
  [x, y, z, w]) for left wrist, right wrist, head, expressed in the reference
  root frame.
- ``smpl``: ``smpl_joints`` (T, 24, 3) root-local SMPL joints, ``root_quat``,
  and ``joint_pos`` (wrist joints are selected here).

``reference.extra`` may set ``encoder_mode`` (``"joint"`` / ``"teleop"`` /
``"smpl"``) per tick, and ``play=False`` to hold frame 0 with zero velocities
(the deployment's paused state).

``ReferenceKind.VELOCITY`` (with a ``planner``): ``value`` is ``[vx, vy, wz]`` in
the body frame. The locomotion planner (:mod:`.sonic_planner`) turns it into
reference frames for the ``joint`` encoder mode. With ``extra["ee_targets"]``
(``{"left_wrist" | "right_wrist" | "head": [x, y, z, qx, qy, qz, qw]}`` in the
base frame) the encoder runs in ``teleop`` mode instead: the planner drives the
legs and the three VR points the wrists and head, as the deployment does in VR
3-point teleoperation. A wrist point is 0.18 m along the wrist-yaw link's x axis
(the palm), the head point 0.35 m above the torso link; the orientation is the
link's. Points not given keep the deployment's defaults.
"""

from __future__ import annotations

import re
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from ..controller import Controller
from ..math_utils import matrix_from_quat, quat_conjugate, quat_mul, quat_rotate_inverse
from ..references.clip import require_clip
from ..registry import register_controller
from ..runtime import load_onnx_runner as _load_onnx
from ..types import ControllerState, JointTarget, Reference, ReferenceKind

_GRAVITY_VEC = np.array([0.0, 0.0, -1.0], dtype=np.float32)
_TERM_RE = re.compile(r"^(?P<base>[a-z0-9_]+?)(?:_(?P<n>\d+)frame_step(?P<s>\d+))?$")
_NUM_SMPL_JOINTS = 24

#: Encoder mode names and their ``encoder_mode_4`` ids, as in the deployment.
#: ``joint`` is the deployment's ``g1`` mode: robot joint-space motion frames.
ENCODER_MODE_IDS = {"joint": 0, "teleop": 1, "smpl": 2}

#: The deployment's VR 3-point targets when none are given (``GetVR3PointPosition`` /
#: ``GetVR3PointOrientation``): left wrist, right wrist, head in the root frame.
#: Quaternions converted to [x, y, z, w].
VR_3POINT_DEFAULT_POS = np.array(
    [[0.0903, 0.1615, -0.2411], [0.1280, -0.1522, -0.2461], [0.0241, -0.0081, 0.4028]],
    dtype=np.float32)
VR_3POINT_DEFAULT_QUAT = np.array(
    [[0.3145, 0.5533, -0.2506, 0.7295], [-0.2639, 0.5395, 0.3217, 0.7320],
     [0.011, 0.0402, -0.0002, 0.9991]], dtype=np.float32)
VR_3POINT_NAMES = ("left_wrist", "right_wrist", "head")
#: The three points: bodies and offsets in the body frame (``vr_3point_body`` /
#: ``vr_3point_body_offset`` of the training config).
VR_3POINT_BODIES = ("left_wrist_yaw_link", "right_wrist_yaw_link", "torso_link")
VR_3POINT_OFFSETS = np.array([[0.18, -0.025, 0.0], [0.18, 0.025, 0.0], [0.0, 0.0, 0.35]])


@dataclass(frozen=True)
class SonicRobotConfig:
    """The robot side of a SONIC checkpoint, in the controller's joint order.

    ``joint_names`` uses hiRobot's names (G1: no ``_joint`` suffix) and fixes the
    order of every joint vector the controller reads or writes. ``lower_body_index``
    selects the teleop encoder's lower-body joints and ``wrist_index`` the SMPL
    encoder's wrist joints, both as indices into ``joint_names``; leave
    ``wrist_index`` empty for a robot without wrists (a checkpoint that uses wrist
    terms is then refused at construction).
    """

    joint_names: tuple[str, ...]
    default_pos: np.ndarray
    action_scale: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    lower_body_index: np.ndarray
    wrist_index: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))

    def __post_init__(self) -> None:
        n = len(self.joint_names)
        if len(set(self.joint_names)) != n:
            raise ValueError("SonicRobotConfig.joint_names repeats a joint")
        for name in ("default_pos", "action_scale", "kp", "kd"):
            v = np.asarray(getattr(self, name), dtype=np.float32)
            if v.shape != (n,):
                raise ValueError(f"SonicRobotConfig.{name} has shape {v.shape}, expected ({n},)")
            object.__setattr__(self, name, v)
        for name in ("lower_body_index", "wrist_index"):
            v = np.asarray(getattr(self, name), dtype=int).reshape(-1)
            if v.size and (v.min() < 0 or v.max() >= n):
                raise ValueError(f"SonicRobotConfig.{name} indexes outside the {n} joints")
            object.__setattr__(self, name, v)

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)


@dataclass(frozen=True)
class SonicObsConfig:
    """One checkpoint's observation layout (its ``observation_config.yaml``).

    ``encoder_terms`` is the encoder input, in order. ``encoder_modes`` maps a
    mode name to ``(mode_id, required_terms)``; terms a mode does not require are
    zero-filled. ``decoder_terms`` is the decoder input and must contain
    ``token_state`` once. Term names are the deployment's observation names.
    """

    encoder_terms: tuple[str, ...]
    encoder_modes: dict[str, tuple[int, tuple[str, ...]]]
    decoder_terms: tuple[str, ...] = field(
        default=(
            "token_state",
            "his_base_angular_velocity_10frame_step1",
            "his_body_joint_positions_10frame_step1",
            "his_body_joint_velocities_10frame_step1",
            "his_last_actions_10frame_step1",
            "his_gravity_dir_10frame_step1",
        )
    )
    token_dim: int = 64


# Per-frame width of every observation term SONIC's released configs use. Joint
# terms are sized by the robot (None here, resolved in _term_widths).
_TERM_WIDTH = {
    "token_state": None,  # token_dim
    "encoder_mode_4": 4,
    "motion_joint_positions": None,
    "motion_joint_velocities": None,
    "motion_joint_positions_lowerbody": None,
    "motion_joint_velocities_lowerbody": None,
    "motion_joint_positions_wrists": None,
    "motion_root_z_position": 1,
    "motion_anchor_orientation": 6,
    "motion_anchor_orientation_heading": 6,
    "smpl_anchor_orientation": 6,
    "smpl_anchor_orientation_heading": 6,
    "smpl_joints": _NUM_SMPL_JOINTS * 3,
    "vr_3point_local_target": 9,
    "vr_3point_local_orn_target": 12,
    "his_base_angular_velocity": 3,
    "his_body_joint_positions": None,
    "his_body_joint_velocities": None,
    "his_last_actions": None,
    "his_gravity_dir": 3,
}


def _term_widths(robot: SonicRobotConfig, token_dim: int) -> dict[str, int]:
    """:data:`_TERM_WIDTH` with the robot-sized terms filled in."""
    n, lower, wrist = robot.num_joints, robot.lower_body_index.size, robot.wrist_index.size
    sized = {
        "token_state": token_dim,
        "motion_joint_positions": n,
        "motion_joint_velocities": n,
        "motion_joint_positions_lowerbody": lower,
        "motion_joint_velocities_lowerbody": lower,
        "motion_joint_positions_wrists": wrist,
        "his_body_joint_positions": n,
        "his_body_joint_velocities": n,
        "his_last_actions": n,
    }
    return {base: sized.get(base, w) for base, w in _TERM_WIDTH.items()}


def _parse_term(name: str) -> tuple[str, int, int]:
    """Split ``motion_joint_positions_10frame_step5`` into ``(base, frames, step)``."""
    m = _TERM_RE.match(name)
    if m is None or m.group("base") not in _TERM_WIDTH:
        raise ValueError(f"unsupported SONIC observation term {name!r}")
    n, s = m.group("n"), m.group("s")
    return m.group("base"), int(n) if n else 1, int(s) if s else 1


def _term_dim(name: str, widths: dict[str, int]) -> int:
    base, n, _ = _parse_term(name)
    return widths[base] * n


def vr_3point(targets: dict) -> tuple[np.ndarray, np.ndarray]:
    """``{name: [x, y, z, qx, qy, qz, qw]}`` -> ``vr_3point_pos`` (3, 3), ``vr_3point_quat`` (3, 4).

    Names are :data:`VR_3POINT_NAMES`; points not given take the defaults.
    """
    unknown = sorted(set(targets) - set(VR_3POINT_NAMES))
    if unknown:
        raise ValueError(f"SONIC tracks {list(VR_3POINT_NAMES)}, not {unknown}")
    pos, quat = VR_3POINT_DEFAULT_POS.copy(), VR_3POINT_DEFAULT_QUAT.copy()
    for i, name in enumerate(VR_3POINT_NAMES):
        if name in targets:
            t = np.asarray(targets[name], np.float64).reshape(-1)
            if t.shape != (7,) or not np.isfinite(t).all():
                raise ValueError(f"ee target {name!r} must be 7 finite numbers [x, y, z, qx, qy, qz, qw]")
            norm = np.linalg.norm(t[3:])
            if norm < 1e-6:
                raise ValueError(f"ee target {name!r} has a zero quaternion")
            pos[i], quat[i] = t[:3], t[3:] / norm
    return pos, quat


def _heading_quat(quat_xyzw: np.ndarray) -> np.ndarray:
    """Yaw-only part of a [x, y, z, w] quaternion (zero x, y, renormalize)."""
    q = np.array([0.0, 0.0, quat_xyzw[2], quat_xyzw[3]], dtype=np.float64)
    return (q / max(np.linalg.norm(q), 1e-9)).astype(np.float32)


def _rot6d(quat_xyzw: np.ndarray) -> np.ndarray:
    """First two columns of the rotation matrix, flattened row-wise (6,)."""
    return matrix_from_quat(quat_xyzw)[:, :2].reshape(-1)


def _xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_xyzw, dtype=np.float32).reshape(-1, 4)
    return q[:, [3, 0, 1, 2]].reshape(-1)


def load_onnx_runner(path: str, *, providers: Sequence[str] | None = None) -> Callable:
    """Build a ``forward_fn`` from a SONIC ONNX file (encoder or decoder).

    Both models take one float32 tensor ``obs_dict`` of shape ``(1, dim)``. A thin
    wrapper over :func:`hiwbc.runtime.load_onnx_runner`; needs the ``onnx`` extra.
    """
    return _load_onnx(path, providers=providers)


class SonicController(Controller):
    """SONIC encoder → token → decoder, with the G1 deployment's action mapping.

    ``decoder_fn(obs) -> action`` is required. ``encoder_fn(obs) -> token`` is
    needed only for ``MOTION_FRAMES`` references; a ``LATENT`` reference supplies
    the token directly. Use :func:`load_onnx_runner` for the released ONNX files,
    or :meth:`from_onnx`.

    Joint vectors are in ``robot.joint_names`` order (for the G1 checkpoints,
    IsaacLab order). The returned target carries ``robot.kp`` / ``robot.kd``.
    """

    accepts = (ReferenceKind.LATENT, ReferenceKind.MOTION_FRAMES)

    def __init__(
        self,
        *,
        decoder_fn: Callable[[np.ndarray], np.ndarray],
        encoder_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        robot: SonicRobotConfig,
        obs: SonicObsConfig,
        encoder_mode: str = "joint",
        stand_root_z: float = 0.793,
        planner=None,
        **base,
    ) -> None:
        # The deployment applies no EMA or clip: last_action is the raw decoder output.
        super().__init__(
            num_actions=robot.num_joints,
            default_pos=robot.default_pos,
            action_scale=robot.action_scale,
            action_beta=1.0,
            joint_names=robot.joint_names,
            kp=robot.kp,
            kd=robot.kd,
            **base,
        )
        self.robot = robot
        self.config = obs
        self.stand_root_z = float(stand_root_z)
        if encoder_mode not in self.config.encoder_modes:
            raise ValueError(
                f"unknown encoder_mode {encoder_mode!r}; available: {sorted(self.config.encoder_modes)}"
            )
        if self.config.decoder_terms.count("token_state") != 1:
            raise ValueError("decoder_terms must contain token_state exactly once")
        self.encoder_mode = encoder_mode
        self._encoder_fn = encoder_fn
        self._decoder_fn = decoder_fn

        td = self.config.token_dim
        self._widths = _term_widths(robot, td)
        terms = self.config.encoder_terms + self.config.decoder_terms
        empty = sorted({t for t in terms if self._widths[_parse_term(t)[0]] == 0})
        if empty:
            raise ValueError(f"checkpoint uses {empty}, but the robot config has no joints for them")
        self.encoder_dim = sum(_term_dim(t, self._widths) for t in self.config.encoder_terms)
        self.decoder_dim = sum(_term_dim(t, self._widths) for t in self.config.decoder_terms)
        # Ring buffer of per-tick proprio entries, newest last; sized for the longest history term.
        self._history_len = max(
            (n - 1) * s + 1
            for b, n, s in map(_parse_term, self.config.decoder_terms)
            if b.startswith("his_")
        )
        self._history: deque[dict[str, np.ndarray]] = deque(maxlen=self._history_len)
        self.token = np.zeros(td, dtype=np.float32)
        # Locomotion planner (sonic_planner.SonicPlanner): VELOCITY -> reference frames.
        self.planner = planner
        self._planner_active = False
        if planner is not None:
            if encoder_fn is None or "joint" not in self.config.encoder_modes:
                raise ValueError("the SONIC planner needs the encoder with a 'joint' mode")
            self.accepts = (*self.accepts, ReferenceKind.VELOCITY)
            self.stand_command = np.zeros(3, dtype=np.float32)

    @classmethod
    def from_onnx(
        cls,
        decoder_path: str,
        encoder_path: str | None = None,
        *,
        providers: Sequence[str] | None = None,
        **kwargs,
    ) -> SonicController:
        """Build from the released ``model_decoder.onnx`` / ``model_encoder.onnx``."""
        return cls(
            decoder_fn=load_onnx_runner(decoder_path, providers=providers),
            encoder_fn=load_onnx_runner(encoder_path, providers=providers) if encoder_path else None,
            **kwargs,
        )

    def reset(self) -> None:
        super().reset()
        self._history.clear()
        self.token = np.zeros(self.config.token_dim, dtype=np.float32)
        if self.planner is not None:
            self.planner.reset()
        self._planner_active = False

    def stand_reference(self, state: ControllerState) -> Reference:
        """Hold the default pose at ``stand_root_z``, facing the robot's current heading.

        A paused one-frame joint-space reference (``play=False``: frame 0 repeated,
        zero velocities), for the ``joint`` encoder mode. With a planner, standing is
        the planner's IDLE: a zero ``VELOCITY`` reference.
        """
        if self.planner is not None:
            return Reference(ReferenceKind.VELOCITY, value=self.stand_command.copy())
        if self.encoder_mode != "joint" or self._encoder_fn is None:
            raise NotImplementedError("SONIC stand_reference needs the encoder in 'joint' mode")
        n = self.num_actions
        frames = {
            "joint_pos": self.default_pos[None].copy(),
            "joint_vel": np.zeros((1, n), dtype=np.float32),
            "root_quat": _heading_quat(np.asarray(state.base_quat, dtype=np.float32))[None],
            "root_z": np.array([self.stand_root_z], dtype=np.float32),
        }
        return Reference(ReferenceKind.MOTION_FRAMES, frames=frames, extra={"play": False})

    # ----- proprio history (StateLogger) ------------------------------------
    def _push_history(self, state: ControllerState) -> None:
        # One entry per tick, logged before inference, so last_action is the
        # previous tick's raw decoder output (as in GatherRobotStateToLogger).
        self._history.append(
            {
                "his_base_angular_velocity": np.asarray(state.base_ang_vel, dtype=np.float32),
                "his_body_joint_positions": (state.dof_pos - self.default_pos).astype(np.float32),
                "his_body_joint_velocities": np.asarray(state.dof_vel, dtype=np.float32),
                "his_last_actions": self._last_action.copy(),
                "his_gravity_dir": quat_rotate_inverse(state.base_quat, _GRAVITY_VEC),
            }
        )

    def _zero_entry(self, base: str) -> np.ndarray:
        # The deployment pads missing history with all-zero entries. A zero
        # quaternion rotates [0, 0, -1] to [0, 0, 1], so padded gravity is +z.
        if base == "his_gravity_dir":
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return np.zeros(self._widths[base], dtype=np.float32)

    def _history_term(self, base: str, n: int, s: int) -> np.ndarray:
        """``n`` entries ``s`` ticks apart, oldest first, zero-padded at the old end."""
        entries = list(self._history)
        picked = []
        for k in range(n):  # k = 0 is the newest
            i = len(entries) - 1 - k * s
            picked.append(entries[i][base] if i >= 0 else self._zero_entry(base))
        return np.concatenate(picked[::-1])

    # ----- encoder input ----------------------------------------------------
    @staticmethod
    def _frames(frames: dict | None, key: str) -> np.ndarray:
        if not frames or key not in frames:
            raise KeyError(f"SONIC reference.frames is missing {key!r}")
        return np.asarray(frames[key], dtype=np.float32)

    def _sample(self, frames: dict, key: str, n: int, s: int, play: bool, frame_ndim: int):
        """Frames ``0, s, 2s, ...`` of a window, clamped to its last frame."""
        arr = self._frames(frames, key)
        if arr.ndim == frame_ndim:  # a single frame
            arr = arr[None]
        idx = np.arange(n) * s if play else np.zeros(n, dtype=int)
        return arr[np.minimum(idx, arr.shape[0] - 1)]

    def _anchor_term(self, frames, state, n, s, play, heading: bool) -> np.ndarray:
        # 6-D rotation of the reference root in the robot's base (or heading) frame.
        roots = self._sample(frames, "root_quat", n, s, play, frame_ndim=1)
        left = _heading_quat(state.base_quat) if heading else np.asarray(state.base_quat)
        left_inv = quat_conjugate(left)
        return np.concatenate([_rot6d(quat_mul(left_inv, q)) for q in roots])

    def _encoder_term(self, name: str, mode_id: int, state, frames, play) -> np.ndarray:
        base, n, s = _parse_term(name)
        if base == "encoder_mode_4":
            return np.array([mode_id, 0.0, 0.0, 0.0], dtype=np.float32)
        if base.startswith("motion_joint_"):
            key = "joint_pos" if "_positions" in base else "joint_vel"
            values = self._sample(frames, key, n, s, play, frame_ndim=1)
            if key == "joint_vel" and not play:
                values = np.zeros_like(values)  # a paused reference has zero velocity
            if base.endswith("_lowerbody"):
                values = values[:, self.robot.lower_body_index]
            elif base.endswith("_wrists"):
                values = values[:, self.robot.wrist_index]
            return values.reshape(-1)
        if base == "motion_root_z_position":
            return self._sample(frames, "root_z", n, s, play, frame_ndim=0).reshape(-1)
        if base in ("motion_anchor_orientation", "smpl_anchor_orientation"):
            return self._anchor_term(frames, state, n, s, play, heading=False)
        if base in ("motion_anchor_orientation_heading", "smpl_anchor_orientation_heading"):
            return self._anchor_term(frames, state, n, s, play, heading=True)
        if base == "smpl_joints":
            return self._sample(frames, "smpl_joints", n, s, play, frame_ndim=2).reshape(-1)
        if base == "vr_3point_local_target":
            return self._frames(frames, "vr_3point_pos").reshape(9)
        if base == "vr_3point_local_orn_target":
            # The encoder was trained on [w, x, y, z]; hiWBC passes [x, y, z, w].
            return _xyzw_to_wxyz(self._frames(frames, "vr_3point_quat")).reshape(12)
        raise ValueError(f"{name!r} is not an encoder term")

    # ----- motion clips ---------------------------------------------------------
    @property
    def motion_spaces(self) -> tuple[str, ...]:
        if self._encoder_fn is None:
            return ()
        return tuple(m for m in ("joint", "smpl") if m in self.config.encoder_modes)

    def _clip_window(self, mode: str) -> int:
        """Frames the encoder reads from the current one (the longest term window)."""
        return max((n - 1) * s + 1 for b, n, s in map(_parse_term, self.config.encoder_modes[mode][1]))

    def check_clip(self, player) -> None:
        super().check_clip(player)
        if player.space == "joint":
            reads_z = "motion_root_z_position" in {
                _parse_term(t)[0] for t in self.config.encoder_modes["joint"][1]}
            require_clip(player, "SONIC", joint_names=self.robot.joint_names, root_pos=reads_z,
                         root_quat=True)
        else:
            require_clip(player, "SONIC", root_quat=True)

    def clip_reference(self, player, state: ControllerState) -> Reference:
        mode = player.space
        win = player.window(self._clip_window(mode), state.base_quat)
        names = self.robot.joint_names
        rows = len(win["root_quat"])
        frames = {
            "joint_pos": player.joints(win["joint_pos"], names, self.default_pos, rows),
            "joint_vel": player.joints(win["joint_vel"], names, np.zeros(len(names)), rows),
            "root_quat": win["root_quat"],
        }
        if win["root_z"] is not None:
            frames["root_z"] = win["root_z"]
        if mode == "smpl":
            frames["smpl_joints"] = win["smpl_joints"]
        frames = {k: np.asarray(v, dtype=np.float32) for k, v in frames.items()}
        return Reference(ReferenceKind.MOTION_FRAMES, frames=frames, extra={"encoder_mode": mode})

    def encoder_obs(self, state: ControllerState, reference: Reference) -> np.ndarray:
        """Encoder input for the active mode; terms the mode does not use stay zero.

        Unlike the deployment, which silently falls back to another mode when a
        required term is unavailable, a missing input raises.
        """
        mode = reference.extra.get("encoder_mode", self.encoder_mode)
        if mode not in self.config.encoder_modes:
            raise ValueError(f"unknown encoder_mode {mode!r}")
        mode_id, required = self.config.encoder_modes[mode]
        play = bool(reference.extra.get("play", True))
        parts = []
        for name in self.config.encoder_terms:
            if name in required:
                parts.append(self._encoder_term(name, mode_id, state, reference.frames, play))
            else:
                parts.append(np.zeros(_term_dim(name, self._widths), dtype=np.float32))
        obs = np.concatenate(parts).astype(np.float32)
        if obs.size != self.encoder_dim:
            raise ValueError(f"encoder obs has {obs.size} values, expected {self.encoder_dim}")
        return obs

    def cartesian_rest(self) -> dict[str, np.ndarray]:
        """The deployment's default VR points (what it tracks with no VR input)."""
        return {name: np.concatenate([VR_3POINT_DEFAULT_POS[i], VR_3POINT_DEFAULT_QUAT[i]])
                for i, name in enumerate(VR_3POINT_NAMES)
                if name in self.cartesian_targets.values()}

    # ----- locomotion planner -----------------------------------------------
    def _planner_reference(self, state: ControllerState, reference: Reference) -> Reference:
        """This tick's planner frames for a VELOCITY reference (``joint`` or ``teleop`` mode)."""
        from .sonic_planner import velocity_command

        if reference.value is None:
            raise ValueError("VELOCITY reference requires [vx, vy, wz] in value")
        planner = self.planner
        quat = np.asarray(state.base_quat, np.float64)[[3, 0, 1, 2]]
        if not self._planner_active:
            planner.start(state.dof_pos)   # Initialize: from the measured joints, IDLE
            self._planner_active = True
        velocity_command(planner, reference.value, quat, dt=self.control_dt)
        targets = reference.extra.get("ee_targets") or {}
        if targets and "teleop" not in self.config.encoder_modes:
            raise ValueError("this SONIC checkpoint has no teleop encoder mode")
        mode = "teleop" if targets else "joint"
        frames = planner.window(self._clip_window(mode), quat)
        if targets:
            frames["vr_3point_pos"], frames["vr_3point_quat"] = vr_3point(targets)
        return Reference(ReferenceKind.MOTION_FRAMES, frames=frames, extra={"encoder_mode": mode})

    # ----- contract ---------------------------------------------------------
    def observe(self, state: ControllerState, reference: Reference) -> np.ndarray:
        if reference.kind == ReferenceKind.VELOCITY:
            if self.planner is None:
                raise ValueError("VELOCITY reference requires the SONIC planner")
            reference = self._planner_reference(state, reference)
        else:
            self._planner_active = False   # back to the planner later: re-initialize
        if reference.kind == ReferenceKind.LATENT:
            if reference.value is None:
                raise ValueError("LATENT reference requires the motion token in value")
            token = np.asarray(reference.value, dtype=np.float32).reshape(-1)
            if token.shape != (self.config.token_dim,):
                raise ValueError(
                    f"motion token has shape {token.shape}, expected ({self.config.token_dim},)"
                )
        else:
            if self._encoder_fn is None:
                raise ValueError("MOTION_FRAMES reference requires an encoder_fn")
            token = np.asarray(
                self._encoder_fn(self.encoder_obs(state, reference)), dtype=np.float32
            ).reshape(-1)
            if token.shape != (self.config.token_dim,):
                raise ValueError(
                    f"encoder returned shape {token.shape}, expected ({self.config.token_dim},)"
                )
        self.token = token

        self._push_history(state)
        parts = []
        for name in self.config.decoder_terms:
            base, n, s = _parse_term(name)
            parts.append(token if base == "token_state" else self._history_term(base, n, s))
        obs = np.concatenate(parts).astype(np.float32)
        if obs.size != self.decoder_dim:
            raise ValueError(f"decoder obs has {obs.size} values, expected {self.decoder_dim}")
        return obs

    def forward(self, obs: np.ndarray) -> np.ndarray:
        return self._decoder_fn(obs)

    def step(self, state: ControllerState, reference: Reference) -> JointTarget:
        target = super().step(state, reference)
        target.extra["motion_token"] = self.token
        if self._planner_active:
            self.planner.advance(state.dof_pos)   # CurrentFrameAdvancement, after the policy
        return target


@register_controller("sonic")
def _build_sonic(**kwargs) -> SonicController:
    return SonicController(**kwargs)

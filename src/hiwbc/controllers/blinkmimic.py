"""BlinkMimic / XMimic motion tracker (Unitree G1, 29 DoF).

BlinkMimic (`fuyichen0725/blinkmimic`, PyTorch + mjlab) and XMimic
(`nico-bohlinger/xmimic`, its JAX port) train the same policy: a motion encoder
over future reference frames and a decoder over a proprio history. Neither
releases weights; checkpoints trained on LAFAN1 are CC BY-NC-ND (non-commercial).
The source of truth is ``mdp.py`` (``MDP.obs``), ``motion.py`` (resampling,
``goal``) and ``networks.py`` (``Policy``) in both repos.

The model takes two inputs:

- ``obs_history`` (10, 99): 10 ticks of a 99-d entry, **frame-major, oldest
  first**, ``[q - q0 (29), qd (29), base ang vel, body frame (3), projected
  gravity (3), previous action (29), mat_to_6d(R^T R_ref) (6)]``. ``R_ref`` is
  the world orientation of the reference root at the current frame, so the clip
  must be heading-aligned to the robot (:class:`~hiwbc.references.ClipPlayer`
  does that). On the first tick after :meth:`reset` all 10 entries are the
  current one. The joint-offset term of training (``dq0``) is zero here.
- ``goal`` (730,): future frames ``f+1 .. f+10`` (held at the clip's last frame)
  of a 73-d feature (:func:`goal_features`), frame-major.

The network (:func:`load_npz_runner`, :func:`load_pt_runner`, numpy at inference)
computes ``z = tanh(enc(goal_norm(goal)))``, ``a = dec([z, obs_norm(hist)])``
plus the reference skip ``(q_ref(f+1) - q0) / scale``. The target is
``q0 + scale * a`` with per-joint ``scale``; no EMA, no clip. The previous action
in the observation is the raw ``a``.

Reference input: ``ReferenceKind.MOTION_FRAMES`` with ``frames['goal']`` of shape
``(10, 73)`` or ``(730,)`` and ``frames['ref_root_quat']`` ``(4,)`` [x, y, z, w],
world frame. :meth:`BlinkMimicController.clip_reference` builds both.

A single-motion ONNX export (BlinkMimic ``tools/export_onnx.py``,
:func:`load_onnx_motion`) holds the goal of one motion and takes ``(obs_history,
time_step)``. With its ``motion_fn`` the controller exposes that motion as a
behavior, played as upstream ``tools/play_onnx.py`` does:

* ``time_step`` is ``0, 1, ..., frames - 2`` (one tick per 50 Hz frame; like a
  clip, the last frame is never the current one), then the behavior ends;
* the 6D term at tick 0 uses ``anchor_quat_w_init`` (the root at frame 0), at
  tick ``t > 0`` the ``next_anchor_quat_w`` output of tick ``t - 1``;
* the previous action starts at zero and the history at 10 copies of the first
  entry, as for any reference (:meth:`reset`);
* the motion's world orientation is turned, yaw only, by the heading of the
  robot base at the first tick minus that of ``anchor_quat_w_init``. play_onnx
  starts the robot at the reference's first state, so there the turn is zero.

A tick whose forward fails (e.g. a remote timeout) does not advance the motion.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from ..controller import Controller
from ..math_utils import gravity_orientation, quat_mul
from ..references.clip import require_clip
from ..registry import register_controller
from ..types import (
    BehaviorSpec,
    ControllerState,
    JointTarget,
    Reference,
    ReferenceKind,
    ReferenceSource,
)
from .supermimic import SUPERMIMIC_JOINT_NAMES

NUM_JOINTS = 29
HISTORY_LEN = 10
FUTURE_FRAMES = 10
OBS_DIM = 3 * NUM_JOINTS + 12  # 99
FRAME_DIM = 15 + 2 * NUM_JOINTS  # 73
GOAL_DIM = FUTURE_FRAMES * FRAME_DIM  # 730
#: Index of the raw joint angles of frame f+1 in the goal (``GOAL_Q_OFFSET`` upstream).
GOAL_Q_OFFSET = 15
#: The normalizers' ``eps``: ``(x - mean) / sqrt(var + eps**2)``.
NORM_EPS = 1e-2

#: Joint order of the policy (Unitree SDK / MuJoCo order, the LAFAN1 retarget CSVs).
BLINKMIMIC_JOINT_NAMES = SUPERMIMIC_JOINT_NAMES


# ----- rotations, [x, y, z, w] quaternions, float64 ---------------------------------
def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], axis=-1).reshape(q.shape[:-1] + (3, 3))


def _so3_log(R: np.ndarray) -> np.ndarray:
    """Rotation vector of rotation matrices: largest-diagonal quaternion branch, then the
    shortest-arc rotation vector (upstream ``rotations.so3_log``)."""
    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    qw = np.sqrt(np.maximum(1 + m00 + m11 + m22, 1e-10)) / 2
    qx = np.sqrt(np.maximum(1 + m00 - m11 - m22, 1e-10)) / 2
    qy = np.sqrt(np.maximum(1 - m00 + m11 - m22, 1e-10)) / 2
    qz = np.sqrt(np.maximum(1 - m00 - m11 + m22, 1e-10)) / 2
    cands = np.stack([
        np.stack([qw, (m21 - m12) / (4 * qw), (m02 - m20) / (4 * qw), (m10 - m01) / (4 * qw)], -1),
        np.stack([(m21 - m12) / (4 * qx), qx, (m01 + m10) / (4 * qx), (m02 + m20) / (4 * qx)], -1),
        np.stack([(m02 - m20) / (4 * qy), (m01 + m10) / (4 * qy), qy, (m12 + m21) / (4 * qy)], -1),
        np.stack([(m10 - m01) / (4 * qz), (m02 + m20) / (4 * qz), (m12 + m21) / (4 * qz), qz], -1),
    ], -2)
    idx = np.argmax(np.stack([qw, qx, qy, qz], -1), -1)
    q = np.take_along_axis(cands, idx[..., None, None], axis=-2)[..., 0, :]  # [w, x, y, z]
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    q = q * np.where(q[..., :1] < 0, -1.0, 1.0)
    v = q[..., 1:]
    sin_half = np.linalg.norm(v, axis=-1, keepdims=True)
    half = np.arctan2(sin_half, q[..., :1])
    scale = np.where(sin_half < 1e-6, 2.0 / np.maximum(q[..., :1], 1e-6),
                     2 * half / (sin_half + 1e-12))
    return v * scale


def mat_to_6d(R: np.ndarray) -> np.ndarray:
    """First two columns, flattened row-major: ``[R00, R01, R10, R11, R20, R21]``."""
    return np.asarray(R)[..., :, :2].reshape(np.shape(R)[:-2] + (6,))


def yaw_mat(R: np.ndarray) -> np.ndarray:
    """Heading (yaw-only) rotation of ``R``, from its projected x axis."""
    fx, fy = R[..., 0, 0], R[..., 1, 0]
    n = np.sqrt(fx * fx + fy * fy + 1e-9)
    c, s = fx / n, fy / n
    o, one = np.zeros_like(c), np.ones_like(c)
    return np.stack([c, -s, o, s, c, o, o, o, one], -1).reshape(R.shape[:-2] + (3, 3))


# ----- G1 constants (BlinkMimic g1_constants.py, BeyondMimic's actuator model) ---------
_ARM_5020 = 0.003609725
_ARM_7520_14 = 0.010177520
_ARM_7520_22 = 0.025101925
_ARM_4010 = 0.00425
#: 10 Hz natural frequency, with upstream's truncated pi (the gains in its MJCF use it).
NATURAL_FREQ = 10 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0
_MOTORS = {  # joint (no side) -> (effort limit N·m, armature kg·m²)
    "hip_pitch": (88.0, _ARM_7520_14), "hip_roll": (139.0, _ARM_7520_22),
    "hip_yaw": (88.0, _ARM_7520_14), "knee": (139.0, _ARM_7520_22),
    "ankle_pitch": (50.0, 2 * _ARM_5020), "ankle_roll": (50.0, 2 * _ARM_5020),
    "waist_yaw": (88.0, _ARM_7520_14), "waist_roll": (50.0, 2 * _ARM_5020),
    "waist_pitch": (50.0, 2 * _ARM_5020),
    "shoulder_pitch": (25.0, _ARM_5020), "shoulder_roll": (25.0, _ARM_5020),
    "shoulder_yaw": (25.0, _ARM_5020), "elbow": (25.0, _ARM_5020),
    "wrist_roll": (25.0, _ARM_5020), "wrist_pitch": (5.0, _ARM_4010),
    "wrist_yaw": (5.0, _ARM_4010),
}
_DEFAULT_POS = {"hip_pitch": -0.312, "knee": 0.669, "ankle_pitch": -0.363, "elbow": 0.6,
                "left_shoulder_roll": 0.2, "left_shoulder_pitch": 0.2,
                "right_shoulder_roll": -0.2, "right_shoulder_pitch": 0.2}


def _motor(name: str) -> tuple[float, float]:
    return _MOTORS[name.removeprefix("left_").removeprefix("right_")]


#: Per joint, in :data:`BLINKMIMIC_JOINT_NAMES` order (float64).
G1_EFFORT = np.array([_motor(n)[0] for n in BLINKMIMIC_JOINT_NAMES])
G1_ARMATURE = np.array([_motor(n)[1] for n in BLINKMIMIC_JOINT_NAMES])
G1_KP = G1_ARMATURE * NATURAL_FREQ ** 2
G1_KD = 2.0 * DAMPING_RATIO * G1_ARMATURE * NATURAL_FREQ
G1_ACTION_SCALE = 0.25 * G1_EFFORT / G1_KP
G1_DEFAULT_POS = np.array([
    _DEFAULT_POS.get(n, _DEFAULT_POS.get(n.removeprefix("left_").removeprefix("right_"), 0.0))
    for n in BLINKMIMIC_JOINT_NAMES])


# ----- goal features ------------------------------------------------------------------
def goal_features(root_pos: np.ndarray, root_quat: np.ndarray, joint_pos: np.ndarray, *,
                  frames: int = FUTURE_FRAMES, dt: float = 0.02) -> np.ndarray:
    """The ``(frames, 73)`` goal at reference frame ``f``, as upstream ``Motion.goal``.

    The input rows are consecutive 50 Hz reference frames ``f, f+1, ...,
    f+frames+1`` with no repeats. Near the clip's end pass only the rows up to
    its last frame: fewer than ``frames + 2`` rows means the last row is the
    clip's last frame. At least 2 rows (upstream never plays the last frame as
    ``f``). ``root_quat`` is [x, y, z, w]; root pose in the world frame,
    heading-aligned; ``joint_pos`` in the policy's joint order.

    Per future frame ``k`` (``f+1 .. f+frames``, held at the clip's last frame):
    ``[H^T (p(k) - p(f)) (3), mat_to_6d(H^T R(k)) (6), R(k)^T v(k) (3),
    R(k)^T w(k) (3), q(k) (29), 0.1 * qd(k) (29)]`` with ``H = yaw_mat(R(f))``.
    ``v``, ``qd``: central differences, one-sided at the clip's last frame;
    ``w = so3_log(R(k+1) R(k-1)^T) / (2 dt)``, likewise.
    """
    p = np.asarray(root_pos, dtype=np.float64).reshape(-1, 3)
    n = len(p)
    if n < 2:
        raise ValueError("goal_features needs the current frame and at least one after it")
    if n > frames + 2:
        raise ValueError(f"goal_features takes at most {frames + 2} rows, got {n}")
    R = _quat_to_mat(np.asarray(root_quat, dtype=np.float64).reshape(n, 4))
    q = np.asarray(joint_pos, dtype=np.float64).reshape(n, -1)
    k = np.minimum(np.arange(1, frames + 1), n - 1)  # future rows, held at the last
    at_end = (n < frames + 2) & (k == n - 1)          # the clip's last frame: one-sided
    lo, hi = k - 1, np.where(at_end, k, k + 1)
    span = ((hi - lo) * dt)[:, None]
    v = (p[hi] - p[lo]) / span
    qd = (q[hi] - q[lo]) / span
    w = _so3_log(R[hi] @ np.swapaxes(R[lo], -1, -2)) / span
    Rk = R[k]
    Rkt = np.swapaxes(Rk, -1, -2)
    Ht = yaw_mat(R[0]).T
    dp = (p[k] - p[0]) @ Ht.T
    rot = mat_to_6d(Ht @ Rk)
    v_local = np.einsum("kij,kj->ki", Rkt, v)
    w_local = np.einsum("kij,kj->ki", Rkt, w)
    return np.concatenate([dp, rot, v_local, w_local, q[k], 0.1 * qd], 1).astype(np.float32)


# ----- the network: one numpy MLP for the .npz and .pt runners -------------------------
def bf16_round(x: np.ndarray) -> np.ndarray:
    """Round float32 values to bfloat16 (round to nearest even), returned as float32."""
    b = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
    return b.astype(np.uint32).view(np.float32)


def _elu(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0.0))).astype(np.float32)


def _mlp(layers, x: np.ndarray, bf16: bool) -> np.ndarray:
    """Linear layers (weights ``(in, out)``) with ELU between them, none after the last.

    ``bf16``: both matmul inputs rounded to bfloat16, float32 accumulation, float32
    bias (XMimic's evaluator, ``nets.mlp_apply`` with ``MM_OUT = float32``).
    """
    h = np.asarray(x, dtype=np.float32)
    for i, (w, b) in enumerate(layers):
        h = (bf16_round(h) @ w if bf16 else h @ w) + b  # w is pre-rounded when bf16
        if i < len(layers) - 1:
            h = _elu(h)
    return h


class PolicyRunner:
    """``forward_fn((obs_history (10, 99), goal (730,))) -> action (29,)``; stateless.

    ``z = tanh(enc(goal_norm(goal)))``, ``a = dec([z, obs_norm(hist).flat])``, plus
    ``(goal[15:44] - ref_q0) / ref_alpha`` when ``ref_q0`` is given (ref skip).
    """

    def __init__(self, enc, dec, obs_norm, goal_norm, *, ref_q0=None, ref_alpha=None,
                 precision: str = "fp32") -> None:
        if precision not in ("fp32", "bf16"):
            raise ValueError(f"precision must be 'fp32' or 'bf16', got {precision!r}")
        _check_layers(enc, dec)
        self.bf16 = precision == "bf16"
        cast = bf16_round if self.bf16 else (lambda w: w)
        self.enc = [(cast(w), b) for w, b in enc]
        self.dec = [(cast(w), b) for w, b in dec]
        self.obs_mean, self.obs_std = _norm(*obs_norm, OBS_DIM, "obs_norm")
        self.goal_mean, self.goal_std = _norm(*goal_norm, GOAL_DIM, "goal_norm")
        if (ref_q0 is None) != (ref_alpha is None):
            raise ValueError("ref_q0 and ref_alpha go together")
        self.ref_q0 = None if ref_q0 is None else _vec(ref_q0, "ref_q0")
        self.ref_alpha = None if ref_alpha is None else _vec(ref_alpha, "ref_alpha")
        if self.ref_alpha is not None and np.any(self.ref_alpha <= 0):
            raise ValueError("ref_alpha must be positive")
        self.precision = precision
        #: XMimic test vectors (obs_history, goal, action) for :func:`self_check`, if any.
        self.test_vectors = None

    def __call__(self, obs) -> np.ndarray:
        hist, goal = obs
        hist = np.asarray(hist, dtype=np.float32).reshape(-1, OBS_DIM)
        goal = np.asarray(goal, dtype=np.float32).reshape(-1)
        if hist.shape != (HISTORY_LEN, OBS_DIM) or goal.shape != (GOAL_DIM,):
            raise ValueError(f"expected obs_history ({HISTORY_LEN}, {OBS_DIM}) and goal "
                             f"({GOAL_DIM},), got {hist.shape} and {goal.shape}")
        z = np.tanh(_mlp(self.enc, (goal - self.goal_mean) / self.goal_std, self.bf16))
        o = ((hist - self.obs_mean) / self.obs_std).reshape(-1)
        a = _mlp(self.dec, np.concatenate([z, o]), self.bf16)
        if self.ref_q0 is not None:
            a = a + (goal[GOAL_Q_OFFSET:GOAL_Q_OFFSET + NUM_JOINTS] - self.ref_q0) / self.ref_alpha
        return a.astype(np.float32)


def _vec(x, name: str) -> np.ndarray:
    v = np.asarray(x, dtype=np.float32).reshape(-1)
    if v.shape != (NUM_JOINTS,):
        raise ValueError(f"{name} has shape {v.shape}, expected ({NUM_JOINTS},)")
    return v


def _norm(mean, var, dim: int, name: str) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(mean, dtype=np.float32).reshape(-1)
    var = np.asarray(var, dtype=np.float32).reshape(-1)
    if mean.shape != (dim,) or var.shape != (dim,):
        raise ValueError(f"{name} statistics have shapes {mean.shape} / {var.shape}, "
                         f"expected ({dim},)")
    return mean, np.sqrt(var + np.float32(NORM_EPS) ** 2).astype(np.float32)


def _check_layers(enc, dec) -> None:
    for name, layers in (("enc", enc), ("dec", dec)):
        if not layers:
            raise ValueError(f"the checkpoint has no {name} layers")
        for i, (w, b) in enumerate(layers):
            if w.ndim != 2 or b.shape != (w.shape[1],):
                raise ValueError(f"{name}.{i}: weight {w.shape} and bias {b.shape} do not match")
            if i and w.shape[0] != layers[i - 1][0].shape[1]:
                raise ValueError(f"{name}.{i} takes {w.shape[0]} inputs, "
                                 f"{name}.{i - 1} gives {layers[i - 1][0].shape[1]}")
    if enc[0][0].shape[0] != GOAL_DIM:
        raise ValueError(f"the encoder takes {enc[0][0].shape[0]} inputs, expected {GOAL_DIM}")
    want = enc[-1][0].shape[1] + HISTORY_LEN * OBS_DIM
    if dec[0][0].shape[0] != want:
        raise ValueError(f"the decoder takes {dec[0][0].shape[0]} inputs, expected {want}")
    if dec[-1][0].shape[1] != NUM_JOINTS:
        raise ValueError(f"the decoder gives {dec[-1][0].shape[1]} actions, expected {NUM_JOINTS}")


def _check_args(args: dict, source: str) -> None:
    got = (args.get("hist"), args.get("goal_frames"))
    if got != (HISTORY_LEN, FUTURE_FRAMES):
        raise ValueError(f"{source}: hist / goal_frames are {got}, this controller needs "
                         f"({HISTORY_LEN}, {FUTURE_FRAMES})")


def _layers(get, prefix: str, step: int, transpose: bool) -> list:
    out, i = [], 0
    while (w := get(f"{prefix}.{i * step}", "w")) is not None:
        w = np.asarray(w, dtype=np.float32)
        out.append((np.ascontiguousarray(w.T if transpose else w),
                    np.asarray(get(f"{prefix}.{i * step}", "b"), dtype=np.float32)))
        i += 1
    return out


def _check_ref(name: str, given, stored: np.ndarray, source: str) -> None:
    if given is not None and not np.allclose(_vec(given, name), stored, rtol=1e-5, atol=1e-6):
        raise ValueError(f"{source}: the checkpoint's {name} differs from the one given")


def load_npz_runner(path: str | Path, *, default_pos: Sequence[float],
                    action_scale: Sequence[float], precision: str = "bf16") -> PolicyRunner:
    """``forward_fn`` from an XMimic ``ckpt_*.npz`` (``runtime.export_policy``).

    XMimic always trains with the reference skip, whose ``q0`` / ``scale`` are not in
    the file: pass the robot's ``default_pos`` / ``action_scale``. ``precision="bf16"``
    rounds both matmul inputs to bfloat16 as XMimic's evaluator does; ``"fp32"``
    does not. Numpy only. When the file holds XMimic's test vectors they are kept
    in ``runner.test_vectors`` for :func:`self_check`.
    """
    path = Path(path).expanduser()
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    source = path.name
    meta = json.loads(str(data["meta"])) if "meta" in data else {}
    _check_args(meta.get("args", {}), source)

    def get(prefix, kind):
        return data.get(f"{prefix}.{kind}")

    runner = PolicyRunner(
        _layers(get, "enc", 1, False), _layers(get, "dec", 1, False),
        (data["obs_norm.mean"], data["obs_norm.var"]),
        (data["goal_norm.mean"], data["goal_norm.var"]),
        ref_q0=default_pos, ref_alpha=action_scale, precision=precision)
    if all(k in data for k in ("test_oh", "test_g", "test_a")):
        runner.test_vectors = (data["test_oh"], data["test_g"], data["test_a"])
    return runner


def self_check(runner: PolicyRunner, test_vectors=None) -> float:
    """Largest ``|a - a_ref| / max(1, |a_ref|max)`` over a checkpoint's test vectors.

    ``test_vectors`` = ``(obs_history (B, 10, 99) raw, goal (B, 730) raw, action
    (B, 29))``, the history raw (the runner normalizes it); default: the ones the
    npz loader found. XMimic computed those actions with bfloat16 matmul inputs, so
    a ``precision="bf16"`` runner matches them to ~1e-7 (use 1e-4 as the bound);
    an ``"fp32"`` runner differs by up to ~1e-2.
    """
    test_vectors = runner.test_vectors if test_vectors is None else test_vectors
    if test_vectors is None:
        raise ValueError("no test vectors: the checkpoint has no test_oh / test_g / test_a")
    oh, g, a = test_vectors
    oh = np.asarray(oh, np.float32).reshape(-1, HISTORY_LEN, OBS_DIM)
    g = np.asarray(g, np.float32).reshape(len(oh), GOAL_DIM)
    a = np.asarray(a, np.float32).reshape(len(oh), NUM_JOINTS)
    got = np.stack([runner((h, gi)) for h, gi in zip(oh, g, strict=True)])
    return float(np.abs(got - a).max() / max(1.0, float(np.abs(a).max())))


def load_pt_runner(path: str | Path, *, default_pos: Sequence[float] | None = None,
                   action_scale: Sequence[float] | None = None) -> PolicyRunner:
    """``forward_fn`` from a BlinkMimic ``.pt`` checkpoint (``ck["pi"]`` + ``ck["args"]``).

    Read with ``torch.load(weights_only=True)`` (the ``torch`` extra, imported here);
    inference is numpy, float32. The reference skip uses the checkpoint's
    ``ref_q0`` / ``ref_alpha`` when ``args["pi_ref_skip"]``; ``default_pos`` /
    ``action_scale``, if given, must agree with them.
    """
    import torch

    path = Path(path).expanduser()
    ck = torch.load(str(path), map_location="cpu", weights_only=True)
    source = path.name
    if not isinstance(ck, dict) or "pi" not in ck:
        raise ValueError(f"{source}: not a BlinkMimic checkpoint (no 'pi' state dict)")
    args = ck.get("args") or {}
    _check_args(args, source)
    sd = {k: v.detach().cpu().float().numpy() for k, v in ck["pi"].items()}

    def get(prefix, kind):
        return sd.get(f"{prefix}.{'weight' if kind == 'w' else 'bias'}")

    enc, dec = _layers(get, "enc", 2, True), _layers(get, "dec", 2, True)
    for name, layers in (("enc", enc), ("dec", dec)):
        key = f"{name}_hidden"
        if key in args:
            want = tuple(int(x) for x in str(args[key]).split(","))
            if tuple(w.shape[1] for w, _ in layers[:-1]) != want:
                raise ValueError(f"{source}: {name} layer widths differ from args {key}={want}")
    ref_q0 = ref_alpha = None
    if args.get("pi_ref_skip", False):
        if "ref_q0" not in sd or "ref_alpha" not in sd:
            raise ValueError(f"{source}: pi_ref_skip is set but ref_q0 / ref_alpha are missing")
        ref_q0, ref_alpha = _vec(sd["ref_q0"], "ref_q0"), _vec(sd["ref_alpha"], "ref_alpha")
        _check_ref("default_pos", default_pos, ref_q0, source)
        _check_ref("action_scale", action_scale, ref_alpha, source)
    return PolicyRunner(enc, dec, (sd["obs_norm.mean"], sd["obs_norm.var"]),
                        (sd["goal_norm.mean"], sd["goal_norm.var"]),
                        ref_q0=ref_q0, ref_alpha=ref_alpha, precision="fp32")


# ----- the controller ---------------------------------------------------------------
class BlinkMimicController(Controller):
    """BlinkMimic / XMimic tracker: frame-major proprio history + 10 future goal frames.

    Args:
        forward_fn: ``forward_fn((obs_history, goal)) -> action`` for clip
            references. ``None`` for a single-motion export, which then only
            plays its behavior.
        motion_fn: ``motion_fn(obs_history (10, 99), time_step) -> dict`` with
            ``actions`` and ``next_anchor_quat_w`` ([w, x, y, z]), as
            :func:`load_onnx_motion`. Enables the behavior ``motion_name``.
        motion_name: Behavior name of the baked motion.
        motion_frames: Frames in the baked motion (50 Hz).
        anchor_quat_init: The motion's root orientation at frame 0, [w, x, y, z].
    """

    accepts = (ReferenceKind.MOTION_FRAMES,)
    motion_spaces = ("joint",)

    def __init__(
        self,
        *,
        forward_fn: Callable | None = None,
        default_pos: Sequence[float],
        action_scale: Sequence[float],
        kp: Sequence[float] | None = None,
        kd: Sequence[float] | None = None,
        effort: Sequence[float] | None = None,
        bound_torque: bool = True,
        control_dt: float = 0.02,
        motion_fn: Callable | None = None,
        motion_name: str | None = None,
        motion_frames: int | None = None,
        anchor_quat_init: Sequence[float] | None = None,
    ) -> None:
        super().__init__(
            num_actions=NUM_JOINTS,
            default_pos=default_pos,
            action_scale=action_scale,
            action_beta=1.0,
            joint_names=BLINKMIMIC_JOINT_NAMES,
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
        if bound_torque and (np.any(self.kp <= 0) or np.any(self.effort <= 0)):
            raise ValueError("bound_torque needs kp > 0 and effort > 0 on every joint")
        self.bound_torque = bound_torque
        self._history: deque[np.ndarray] = deque(maxlen=HISTORY_LEN)
        if forward_fn is None and motion_fn is None:
            raise ValueError("BlinkMimic needs forward_fn, motion_fn or both")
        self._motion_fn = motion_fn
        self.motion_name = motion_name
        self.motion_frames = motion_frames
        self.anchor_quat_init = None
        self.start_pos = None
        if motion_fn is not None:
            if not motion_name or motion_frames is None or anchor_quat_init is None:
                raise ValueError("motion_fn needs motion_name, motion_frames and anchor_quat_init")
            if int(motion_frames) < 2:
                raise ValueError(f"motion_frames is {motion_frames}, need at least 2")
            self.motion_frames = int(motion_frames)
            q = np.asarray(anchor_quat_init, dtype=np.float64).reshape(-1)
            if q.shape != (4,) or not np.isfinite(q).all() or not 0.5 < np.linalg.norm(q) < 1.5:
                raise ValueError("anchor_quat_init must be a [w, x, y, z] unit quaternion")
            self.anchor_quat_init = q / np.linalg.norm(q)
            # The reference joint angles at frame 0: the behavior's start pose.
            first = motion_fn(np.zeros((HISTORY_LEN, OBS_DIM), np.float32), 0)
            self.start_pos = np.asarray(first["joint_pos"], np.float32).reshape(-1)
            if self.start_pos.shape != (NUM_JOINTS,):
                raise ValueError(f"motion_fn joint_pos has shape {self.start_pos.shape}")
        self._time_step: int | None = None    # baked-motion frame of the tick being run
        self._motion_out: dict | None = None  # outputs of the last baked-motion forward
        self._motion_t: int | None = None     # its time_step
        self._clip_player = None              # the player clip_reference last read

    def reset(self) -> None:
        super().reset()
        self._history.clear()
        self._motion_out = self._motion_t = None

    # ----- observation ------------------------------------------------------------
    def _entry(self, state: ControllerState, ref_quat: np.ndarray) -> np.ndarray:
        quat = np.asarray(state.base_quat, dtype=np.float64)
        norm = np.linalg.norm(quat)
        if not np.isfinite(norm) or not 0.5 < norm < 1.5:
            raise ValueError("invalid base quaternion")
        quat = quat / norm
        R, R_ref = _quat_to_mat(quat), _quat_to_mat(ref_quat)
        return np.concatenate([
            state.dof_pos - self.default_pos,
            state.dof_vel,
            state.base_ang_vel,
            gravity_orientation(quat),
            self._last_action,
            mat_to_6d(R.T @ R_ref),
        ]).astype(np.float32)

    @staticmethod
    def _reference(reference: Reference) -> tuple[np.ndarray, np.ndarray]:
        frames = reference.frames
        if frames is None or "goal" not in frames or "ref_root_quat" not in frames:
            raise ValueError("BlinkMimic needs reference.frames['goal'] (10, 73) and "
                             "['ref_root_quat'] (4,)")
        goal = np.asarray(frames["goal"], dtype=np.float32)
        if goal.shape not in ((FUTURE_FRAMES, FRAME_DIM), (GOAL_DIM,)):
            raise ValueError(f"goal has shape {goal.shape}, expected ({FUTURE_FRAMES}, "
                             f"{FRAME_DIM}) or ({GOAL_DIM},)")
        if not np.isfinite(goal).all():
            raise ValueError("non-finite goal")
        return goal.reshape(-1), BlinkMimicController._ref_quat(frames)

    @staticmethod
    def _ref_quat(frames) -> np.ndarray:
        quat = np.asarray(frames["ref_root_quat"], dtype=np.float64)
        if quat.shape != (4,):
            raise ValueError(f"ref_root_quat has shape {quat.shape}, expected (4,)")
        norm = np.linalg.norm(quat)
        if not np.isfinite(norm) or not 0.5 < norm < 1.5:
            raise ValueError("invalid ref_root_quat")
        return quat / norm

    def _push(self, entry: np.ndarray) -> np.ndarray:
        if not self._history:
            self._history.extend([entry] * HISTORY_LEN)
        else:
            self._history.append(entry)
        return np.stack(self._history)  # (10, 99) oldest first

    def observe(self, state: ControllerState, reference: Reference):
        if self._time_step is not None:
            # The baked motion: the ONNX export holds the goal, only the history goes in.
            frames = reference.frames
            if frames is None or "ref_root_quat" not in frames:
                raise ValueError("the baked motion needs reference.frames['ref_root_quat']")
            return self._push(self._entry(state, self._ref_quat(frames)))
        if self._forward_fn is None:
            raise ValueError(f"this BlinkMimic export only plays its behavior "
                             f"{self.motion_name!r}; it has no policy for other references")
        goal, ref_quat = self._reference(reference)
        return self._push(self._entry(state, ref_quat)), goal  # (10, 99), (730,)

    def forward(self, obs) -> np.ndarray:
        if self._time_step is None:
            action = np.asarray(self._forward_fn(obs), dtype=np.float32).reshape(-1)
        else:
            out = self._motion_fn(obs, self._time_step)
            action = np.asarray(out["actions"], dtype=np.float32).reshape(-1)
            nxt = np.asarray(out["next_anchor_quat_w"], dtype=np.float64).reshape(-1)
            if nxt.shape != (4,) or not np.isfinite(nxt).all():
                raise ValueError("BlinkMimic motion_fn returned an invalid next_anchor_quat_w")
            if action.shape == (NUM_JOINTS,) and np.isfinite(action).all():
                self._motion_out = {"actions": action, "next_anchor_quat_w": nxt}
                self._motion_t = self._time_step
        if not np.isfinite(action).all():
            raise ValueError("BlinkMimic policy returned a non-finite action")
        return action

    def step(self, state: ControllerState, reference: Reference) -> JointTarget:
        t = reference.extra.get("time_step") if self._motion_fn is not None else None
        self._time_step = None if t is None else int(t)
        try:
            target = super().step(state, reference)
        finally:
            self._time_step = None
        if self.bound_torque:
            # Re-express the target so the robot's PD torque is clipped at the effort
            # limit, as the training actuators (forcerange = effort) do.
            q = np.asarray(state.dof_pos, dtype=np.float32)
            dq = np.asarray(state.dof_vel, dtype=np.float32)
            tau = np.clip(self.kp * (target.position - q) - self.kd * dq, -self.effort, self.effort)
            target.position = (q + (tau + self.kd * dq) / self.kp).astype(np.float32)
        return target

    # ----- motion clips -------------------------------------------------------------
    def check_clip(self, player) -> None:
        if self._forward_fn is None:
            raise ValueError(f"this BlinkMimic export only plays its behavior "
                             f"{self.motion_name!r}; it cannot track clips")
        super().check_clip(player)
        require_clip(player, "BlinkMimic", joint_names=self.joint_names, root_pos=True,
                     root_quat=True)

    def clip_reference(self, player, state: ControllerState) -> Reference:
        """Goal and current root orientation at ``player``'s current frame ``f``.

        As upstream, ``f`` stops one before the clip's last frame (the goal starts
        at ``f+1``); a one-frame clip is held. A new ``player`` resets the history
        and the previous action, as each upstream episode starts.
        """
        if player is not self._clip_player:
            # A new clip starts like an upstream episode: fresh history, a_prev = 0.
            self.reset()
            self._clip_player = player
        last = player.num_frames - 1
        f = min(player.frame, max(last - 1, 0))
        rows = min(FUTURE_FRAMES + 2, last - f + 1)  # f .. f+11, no repeated frames
        saved, player.frame = player.frame, f
        try:
            win = player.window(max(rows, 2), state.base_quat)
        finally:
            player.frame = saved
        pos, quat = win["root_pos"], win["root_quat"]
        joints = player.joints(win["joint_pos"], self.joint_names, self.default_pos)
        goal = goal_features(pos, quat, joints, frames=FUTURE_FRAMES, dt=self.control_dt)
        return Reference(ReferenceKind.MOTION_FRAMES,
                         frames={"goal": goal, "ref_root_quat": quat[0].astype(np.float32)})

    # ----- behavior: the motion baked into a single-motion export ---------------------
    @property
    def behaviors(self) -> list[BehaviorSpec]:
        if self._motion_fn is None:
            return []
        return [BehaviorSpec(
            name=self.motion_name,
            description=(f"Play the motion baked into this export "
                         f"({(self.motion_frames - 1) * self.control_dt:g} s)."),
            start_pos=tuple(float(x) for x in self.start_pos),
        )]

    def start_behavior(self, name: str, **args) -> ReferenceSource:
        if self._motion_fn is None or name != self.motion_name:
            return super().start_behavior(name, **args)
        if args:
            raise TypeError(f"behavior {name!r} takes no arguments, got {sorted(args)}")
        self.reset()   # a_prev = 0, empty history (filled with the first entry)
        return _BakedMotion(self)


def _heading_quat(q_xyzw: np.ndarray) -> np.ndarray:
    """Yaw-only rotation [x, y, z, w] with the heading of ``q`` (its projected x axis,
    as :func:`yaw_mat`)."""
    R = _quat_to_mat(np.asarray(q_xyzw, np.float64))
    a = np.arctan2(R[1, 0], R[0, 0])
    return np.array([0.0, 0.0, np.sin(a / 2), np.cos(a / 2)])


def _wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.asarray(q, np.float64)[[1, 2, 3, 0]]


class _BakedMotion:
    """The single-motion export's reference, one 50 Hz frame per tick (play_onnx).

    Tick ``t`` runs ``time_step = t`` with the 6D term against the root
    orientation at frame ``t``: ``anchor_quat_init`` for ``t = 0``, else the
    previous forward's ``next_anchor_quat_w``. Progress is read from the
    controller's last successful baked-motion forward, so a failed tick is redone.
    Ends after ``time_step = frames - 2``.
    """

    def __init__(self, ctrl: BlinkMimicController) -> None:
        self._ctrl = ctrl
        self._align: np.ndarray | None = None   # heading turn, [x, y, z, w]

    @property
    def done(self) -> bool:
        t = self._ctrl._motion_t
        return t is not None and t >= self._ctrl.motion_frames - 2

    def next(self, state: ControllerState) -> Reference:
        c = self._ctrl
        init = _wxyz_to_xyzw(c.anchor_quat_init)
        if self._align is None:
            # Turn the motion's heading at frame 0 onto the robot base's, yaw only.
            robot = np.asarray(state.base_quat, np.float64)
            h0 = _heading_quat(init)
            self._align = quat_mul(_heading_quat(robot), h0 * np.array([-1, -1, -1, 1.0]))
        if c._motion_t is None:
            t, anchor = 0, init
        else:
            t, anchor = c._motion_t + 1, _wxyz_to_xyzw(c._motion_out["next_anchor_quat_w"])
        anchor = quat_mul(self._align, anchor)
        return Reference(ReferenceKind.MOTION_FRAMES,
                         frames={"ref_root_quat": anchor.astype(np.float32)},
                         extra={"time_step": t})


def load_onnx_motion(path: str | Path, providers: Sequence[str] | None = None) -> tuple:
    """``(motion_fn, metadata)`` from a BlinkMimic ``tools/export_onnx.py`` file.

    ``motion_fn(obs_history (10, 99), time_step) -> {actions, joint_pos,
    next_anchor_quat_w}`` (quaternion [w, x, y, z]); ``time_step`` is fed as a
    float frame index. ``metadata`` is the file's custom metadata, parsed:
    ``joint_names`` (``_joint`` suffix stripped), ``kp``, ``kd``, ``default_pos``,
    ``action_scale`` (float64, file order), ``anchor_quat_init`` ([w, x, y, z]),
    ``frames``, ``motion``, ``history``, ``goal_frames``, ``control_dt``,
    ``observation_names``, ``anchor_body_name``. Requires the ``onnx`` extra (``onnxruntime``).
    """
    import onnxruntime as ort

    path = Path(path).expanduser()
    session = ort.InferenceSession(str(path),
                                   providers=list(providers or ["CPUExecutionProvider"]))
    m = session.get_modelmeta().custom_metadata_map
    source = path.name
    need = ("joint_names", "joint_stiffness", "joint_damping", "default_joint_pos",
            "action_scale", "observation_history", "goal_frames", "control_dt",
            "anchor_quat_w_init", "motion_num_frames")
    missing = [k for k in need if k not in m]
    if missing:
        raise ValueError(f"{source}: not a BlinkMimic export (metadata lacks {missing})")

    def floats(key):
        return np.array([float(x) for x in m[key].split(",")], dtype=np.float64)

    meta = {
        "joint_names": tuple(n.removesuffix("_joint") for n in m["joint_names"].split(",")),
        "kp": floats("joint_stiffness"), "kd": floats("joint_damping"),
        "default_pos": floats("default_joint_pos"), "action_scale": floats("action_scale"),
        "anchor_quat_init": floats("anchor_quat_w_init"),
        "frames": int(m["motion_num_frames"]),
        "motion": m.get("motion") or path.stem,
        "history": int(m["observation_history"]), "goal_frames": int(m["goal_frames"]),
        "control_dt": float(m["control_dt"]),
        "observation_names": tuple(m.get("observation_names", "").split(",")),
        "anchor_body_name": m.get("anchor_body_name"),
    }
    inputs = {i.name for i in session.get_inputs()}
    outputs = {o.name for o in session.get_outputs()}
    want_out = ("actions", "joint_pos", "next_anchor_quat_w")
    if inputs != {"obs_history", "time_step"} or not set(want_out) <= outputs:
        raise ValueError(f"{source}: inputs {sorted(inputs)} / outputs {sorted(outputs)} are not "
                         "a BlinkMimic export's (obs_history, time_step -> actions, ..., "
                         "next_anchor_quat_w)")

    def run(obs_history, time_step):
        feed = {"obs_history": np.asarray(obs_history, np.float32).reshape(1, -1, OBS_DIM),
                "time_step": np.array([[float(time_step)]], np.float32)}
        return {k: v[0] for k, v in zip(want_out, session.run(list(want_out), feed), strict=True)}

    return run, meta


@register_controller("blinkmimic")
def _build_blinkmimic(**kwargs) -> BlinkMimicController:
    return BlinkMimicController(**kwargs)

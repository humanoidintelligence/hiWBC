"""Motion clip playback: resample to the control rate, align heading, window future frames.

A :class:`ClipPlayer` holds one clip (hiRobot ``MotionClip`` or anything with the
same fields) resampled to the controller's period. Each tick the caller asks the
controller for ``ctrl.clip_reference(player, state)``, steps the controller, then
calls ``player.advance()``. Windows past the end repeat the last frame, as the
SONIC deployment does.

Heading: on the first tick the clip is rotated about z so that its first root
heading matches the robot's heading at that moment (the deployment's
``apply_delta_heading``), and translated so its first root position is at the
origin in x, y. Root velocities are world-frame, in the aligned frame.

Missing data: ``joint_vel`` is differentiated from ``joint_pos``; root linear and
angular velocities are differentiated from ``root_pos`` / ``root_quat``. Without
``root_quat`` the root is upright at the robot's heading; without ``root_pos``
root velocities are zero and ``root_z`` is ``None``.

With ``source_velocity_sigma`` the root velocities are instead computed at the
clip's own rate and Gaussian-smoothed before resampling, as PHC's motion library
does; controllers trained on PHC data (KungfuBot) ask for it through
``clip_player_options``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

_EPS = 1e-9


# ----- quaternion helpers, [x, y, z, w], vectorized over the leading axis ----------
def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=-1)


def _qconj(q: np.ndarray) -> np.ndarray:
    return q * np.array([-1.0, -1.0, -1.0, 1.0])


def _qnormalize(q: np.ndarray) -> np.ndarray:
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), _EPS)


def _heading(q: np.ndarray) -> np.ndarray:
    """Yaw-only part (zero x, y, renormalize), as the deployment's ``calc_heading_quat``."""
    h = np.zeros_like(q)
    h[..., 2], h[..., 3] = q[..., 2], q[..., 3]
    return _qnormalize(h)


def _qrotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    u, w = q[..., :3], q[..., 3:4]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def _qlog(q: np.ndarray) -> np.ndarray:
    """Rotation vector of a unit quaternion (shortest arc)."""
    q = np.where(q[..., 3:4] < 0, -q, q)
    s = np.linalg.norm(q[..., :3], axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(s, q[..., 3:4])
    return np.where(s > _EPS, q[..., :3] * angle / np.maximum(s, _EPS), 2.0 * q[..., :3])


def _slerp(q0: np.ndarray, q1: np.ndarray, a: np.ndarray) -> np.ndarray:
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.abs(dot)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin = np.sin(theta)
    near = sin < 1e-6
    w0 = np.where(near, 1.0 - a, np.sin((1.0 - a) * theta) / np.where(near, 1.0, sin))
    w1 = np.where(near, a, np.sin(a * theta) / np.where(near, 1.0, sin))
    return _qnormalize(w0 * q0 + w1 * q1)


def _gaussian(arr: np.ndarray, sigma: float) -> np.ndarray:
    """1-D Gaussian filter along axis 0, edges extended (``scipy gaussian_filter1d``, ``mode="nearest"``)."""
    if sigma <= 0 or len(arr) < 2:
        return arr
    r = int(4.0 * sigma + 0.5)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.concatenate([np.repeat(arr[:1], r, 0), arr, np.repeat(arr[-1:], r, 0)])
    return sum(k[i] * pad[i:i + len(arr)] for i in range(2 * r + 1))


def _source_velocities(pos: np.ndarray | None, quat: np.ndarray | None, fps: float,
                       sigma: float) -> dict[str, np.ndarray]:
    """Root velocities at the clip's own rate, as PHC's motion library computes them.

    Linear: ``np.gradient`` of position. Angular: forward difference of orientation,
    zero on the last frame. Both then Gaussian-smoothed with ``sigma`` frames.
    """
    dt, out = 1.0 / fps, {}
    if pos is not None:
        out["root_lin_vel"] = _gaussian(_diff(pos, dt), sigma)
    if quat is not None:
        q = _qnormalize(quat)
        dq = np.zeros_like(q)
        dq[:, 3] = 1.0
        dq[:-1] = _qmul(q[1:], _qconj(q[:-1]))
        out["root_ang_vel"] = _gaussian(_qlog(dq) / dt, sigma)
    return out


# ----- resampling -------------------------------------------------------------------
def _sample_points(num_frames: int, fps: float, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Source indices (i0, i1) and blend for ticks 0, dt, 2 dt, ... up to the last frame."""
    ratio = fps * dt  # source frames per tick
    n_ticks = int(np.floor((num_frames - 1) / ratio + _EPS)) + 1
    x = np.arange(n_ticks) * ratio
    snapped = np.round(x)
    x = np.where(np.abs(x - snapped) < 1e-6, snapped, x)
    i0 = np.minimum(np.floor(x).astype(int), num_frames - 1)
    i1 = np.minimum(i0 + 1, num_frames - 1)
    return i0, i1, (x - i0)


def _lerp(arr: np.ndarray, i0, i1, a) -> np.ndarray:
    if np.all(a == 0):
        return arr[i0].copy()  # same rate: exact copies, no rounding
    a = a.reshape((-1,) + (1,) * (arr.ndim - 1))
    return arr[i0] * (1.0 - a) + arr[i1] * a


def _diff(arr: np.ndarray, dt: float) -> np.ndarray:
    if len(arr) < 2:
        return np.zeros_like(arr)
    return np.gradient(arr, dt, axis=0)


def _ang_vel(quats: np.ndarray, dt: float) -> np.ndarray:
    """World-frame angular velocity from a quaternion sequence (central differences)."""
    n = len(quats)
    if n < 2:
        return np.zeros((n, 3))
    nxt = quats[np.minimum(np.arange(n) + 1, n - 1)]
    prv = quats[np.maximum(np.arange(n) - 1, 0)]
    span = np.where((np.arange(n) == 0) | (np.arange(n) == n - 1), 1.0, 2.0)[:, None] * dt
    return _qlog(_qmul(nxt, _qconj(prv))) / span


def _strip(name: str) -> str:
    return name.removesuffix("_joint")


class ClipPlayer:
    """Plays a ``MotionClip`` at the controller's period.

    Args:
        clip: hiRobot ``MotionClip`` (or an object with the same attributes).
        control_dt: The controller's period (``ctrl.control_dt``).
        source_velocity_sigma: ``None`` (default): root velocities are differentiated
            at the control rate. A number: they are computed at the clip's own rate
            and Gaussian-smoothed with this many source frames, then resampled, as
            PHC's motion library does (KungfuBot was trained on that; see
            ``Controller.clip_player_options``).
    """

    def __init__(self, clip: Any, control_dt: float,
                 source_velocity_sigma: float | None = None) -> None:
        if not control_dt > 0:
            raise ValueError("control_dt must be positive")
        if source_velocity_sigma is not None and source_velocity_sigma < 0:
            raise ValueError("source_velocity_sigma must be >= 0")
        self.dt = float(control_dt)
        self.source_velocity_sigma = source_velocity_sigma
        self.space = clip.space
        self.joint_names: tuple[str, ...] | None = (
            None if clip.joint_names is None else tuple(_strip(n) for n in clip.joint_names))
        self._raw: dict[str, np.ndarray] = {}
        self._sources: list[dict[str, Any]] = []  # each clip at its own rate, for derived terms
        self._derived_fns: dict[str, Any] = {}
        self._derived: dict[str, np.ndarray] = {}
        self.has_root_quat = clip.root_quat is not None
        self.has_root_pos = clip.root_pos is not None
        self.has_joint_vel = clip.joint_vel is not None
        self.has_smpl_joints = clip.smpl_joints is not None
        self._add(clip)
        self.frame = 0
        self._align_quat: np.ndarray | None = None  # heading rotation applied to the clip
        self._align_pos: np.ndarray | None = None
        self._world_xy = np.zeros(3)
        self._aligned: dict[str, np.ndarray] = {}

    # ----- loading --------------------------------------------------------------
    def _resampled(self, clip: Any) -> dict[str, np.ndarray]:
        arrays = {k: np.asarray(v, dtype=np.float64) for k, v in
                  [("joint_pos", clip.joint_pos), ("joint_vel", clip.joint_vel),
                   ("smpl_joints", clip.smpl_joints), ("root_pos", clip.root_pos),
                   ("root_quat", clip.root_quat)] if v is not None}
        num_frames = len(next(iter(arrays.values())))
        i0, i1, a = _sample_points(num_frames, float(clip.fps), self.dt)
        if self.source_velocity_sigma is not None:
            arrays.update({"src_" + k: v for k, v in _source_velocities(
                arrays.get("root_pos"), arrays.get("root_quat"), float(clip.fps),
                self.source_velocity_sigma).items()})
        out = {}
        for key, arr in arrays.items():
            if key == "root_quat":
                q = _qnormalize(arr)
                out[key] = q[i0].copy() if np.all(a == 0) else _slerp(q[i0], q[i1], a[:, None])
            else:
                out[key] = _lerp(arr, i0, i1, a)
        return out

    def _add(self, clip: Any) -> None:
        self._sources.append({"fps": float(clip.fps), **{
            k: None if getattr(clip, k) is None else np.asarray(getattr(clip, k), dtype=np.float64)
            for k in ("joint_pos", "root_pos", "root_quat")}})
        new = self._resampled(clip)
        if not self._raw:
            self._raw = new
            return
        for key in self._raw:
            self._raw[key] = np.concatenate([self._raw[key], new[key]])

    def append(self, clip: Any) -> None:
        """Queue ``clip`` after the current one (streaming). Same space and fields."""
        names = None if clip.joint_names is None else tuple(_strip(n) for n in clip.joint_names)
        same = (clip.space == self.space and names == self.joint_names
                and (clip.root_quat is not None) == self.has_root_quat
                and (clip.root_pos is not None) == self.has_root_pos
                and (clip.joint_vel is not None) == self.has_joint_vel
                and (clip.smpl_joints is not None) == self.has_smpl_joints)
        if not same:
            raise ValueError("an appended clip must have the same space, joint_names and fields")
        self._add(clip)
        if self._align_quat is not None:
            self._aligned = self._align()
            self._derived = {k: self._compute_derived(fn) for k, fn in self._derived_fns.items()}

    # ----- playback ---------------------------------------------------------------
    @property
    def num_frames(self) -> int:
        return len(next(iter(self._raw.values())))

    @property
    def duration(self) -> float:
        """Seconds from the first to the last frame."""
        return (self.num_frames - 1) * self.dt

    @property
    def time(self) -> float:
        return min(self.frame, self.num_frames - 1) * self.dt

    @property
    def done(self) -> bool:
        """True once the last frame has been played."""
        return self.frame >= self.num_frames

    def advance(self, steps: int = 1) -> None:
        self.frame = min(self.frame + steps, self.num_frames)

    def _align(self) -> dict[str, np.ndarray]:
        raw, n = self._raw, self.num_frames
        if self.has_root_quat:
            quat = _qnormalize(_qmul(np.broadcast_to(self._align_quat, (n, 4)), raw["root_quat"]))
        else:
            quat = np.broadcast_to(self._align_quat, (n, 4)).copy()
        out = {"root_quat": quat}
        if self.has_root_pos:
            pos = _qrotate(np.broadcast_to(self._align_quat, (n, 4)), raw["root_pos"] - self._align_pos) + self._world_xy
            out["root_pos"] = pos
            out["root_lin_vel"] = (_qrotate(np.broadcast_to(self._align_quat, (n, 4)),
                                            raw["src_root_lin_vel"])
                                   if "src_root_lin_vel" in raw else _diff(pos, self.dt))
        else:
            out["root_lin_vel"] = np.zeros((n, 3))
        if not self.has_root_quat:
            out["root_ang_vel"] = np.zeros((n, 3))
        elif "src_root_ang_vel" in raw:
            out["root_ang_vel"] = _qrotate(np.broadcast_to(self._align_quat, (n, 4)),
                                           raw["src_root_ang_vel"])
        else:
            out["root_ang_vel"] = _ang_vel(quat, self.dt)
        if "joint_pos" in raw:
            out["joint_vel"] = raw["joint_vel"] if self.has_joint_vel else _diff(raw["joint_pos"], self.dt)
        return out

    def start(self, base_quat: Sequence[float], base_pos: Sequence[float] | None = None) -> None:
        """Align the clip to the robot's heading. Called by the first ``window`` if not yet.

        The clip's first root position goes to the horizontal origin, or under
        ``base_pos`` (the robot's world position) when given.
        """
        robot_heading = _heading(np.asarray(base_quat, dtype=np.float64))
        if self.has_root_quat:
            first = self._raw["root_quat"][0]
            self._align_quat = _qnormalize(_qmul(robot_heading, _qconj(_heading(first))))
        else:
            self._align_quat = robot_heading
        if self.has_root_pos:
            p0 = self._raw["root_pos"][0].copy()
            p0[2] = 0.0
            self._align_pos = p0
            self._world_xy = np.zeros(3)
            if base_pos is not None:
                self._world_xy[:2] = np.asarray(base_pos, dtype=np.float64)[:2]
        self._aligned = self._align()
        self._derived = {k: self._compute_derived(fn) for k, fn in self._derived_fns.items()}

    # ----- terms computed at the clip's own rate ------------------------------------
    def add_derived(self, name: str, fn) -> None:
        """Register a term computed from each clip at its own frame rate, then resampled.

        ``fn(frames)`` gets a dict with ``fps``, ``joint_pos`` (clip joint order) and
        ``root_pos`` / ``root_quat`` after heading alignment (``None`` when missing),
        and returns an array with one row per source frame. The result is
        resampled to the control rate like the clip and appears in :meth:`window`
        under ``name``. Used for body positions and velocities from forward
        kinematics, which PHC-trained policies saw computed this way.
        """
        if name in self._raw or name in self._derived_fns:
            raise ValueError(f"derived term {name!r} already exists")
        self._derived_fns[name] = fn
        if self.started:
            self._derived[name] = self._compute_derived(fn)

    def has_derived(self, name: str) -> bool:
        return name in self._derived_fns

    def _compute_derived(self, fn) -> np.ndarray:
        parts = []
        for src in self._sources:
            n = len(next(v for k, v in src.items() if k != "fps" and v is not None))
            frames = {"fps": src["fps"], "joint_pos": src["joint_pos"], "root_pos": None,
                      "root_quat": None}
            align = np.broadcast_to(self._align_quat, (n, 4))
            if src["root_pos"] is not None:
                frames["root_pos"] = _qrotate(align, src["root_pos"] - self._align_pos) + self._world_xy
            if src["root_quat"] is not None:
                frames["root_quat"] = _qnormalize(_qmul(align, _qnormalize(src["root_quat"])))
            values = np.asarray(fn(frames), dtype=np.float64)
            if len(values) != n:
                raise ValueError(f"a derived term must have one row per source frame ({n})")
            i0, i1, a = _sample_points(n, src["fps"], self.dt)
            parts.append(_lerp(values, i0, i1, a))
        return np.concatenate(parts)

    @property
    def started(self) -> bool:
        return self._align_quat is not None

    def window(self, length: int, base_quat: Sequence[float],
               base_pos: Sequence[float] | None = None) -> dict[str, np.ndarray | None]:
        """``length`` frames from the current one (clamped to the last), aligned.

        Keys: ``joint_pos`` / ``joint_vel`` (clip joint order), ``smpl_joints``,
        ``root_quat`` (always), ``root_pos`` / ``root_z`` (None without ``root_pos``),
        ``root_lin_vel`` / ``root_ang_vel`` (world frame), and each term registered
        with :meth:`add_derived`. Missing data is ``None``.
        """
        if not self.started:
            self.start(base_quat, base_pos)
        idx = np.minimum(self.frame + np.arange(max(int(length), 1)), self.num_frames - 1)
        a, raw = self._aligned, self._raw

        def pick(src, key):
            return None if key not in src else src[key][idx]

        root_pos = pick(a, "root_pos")
        derived = {k: v[idx] for k, v in self._derived.items()}
        return derived | {
            "joint_pos": pick(raw, "joint_pos"),
            "joint_vel": pick(a, "joint_vel"),
            "smpl_joints": pick(raw, "smpl_joints"),
            "root_quat": a["root_quat"][idx],
            "root_pos": root_pos,
            "root_z": None if root_pos is None else root_pos[:, 2],
            "root_lin_vel": a["root_lin_vel"][idx],
            "root_ang_vel": a["root_ang_vel"][idx],
        }

    def joints(self, values: np.ndarray | None, names: Sequence[str],
               fill: Sequence[float] | np.ndarray, rows: int | None = None) -> np.ndarray:
        """Reorder ``(W, n_clip)`` clip-order values to ``names``; joints the clip lacks get ``fill``.

        ``rows`` sets W when ``values`` is ``None`` (a clip with no joint data).
        """
        fill = np.asarray(fill, dtype=np.float64)
        rows = (rows or 1) if values is None else len(values)
        out = np.broadcast_to(fill, (rows, len(names))).copy()
        if values is None or self.joint_names is None:
            return out
        col = {n: i for i, n in enumerate(self.joint_names)}
        for j, name in enumerate(names):
            if name in col:
                out[:, j] = values[:, col[name]]
        return out

    def covers(self, names: Sequence[str]) -> list[str]:
        """Names in ``names`` the clip has no column for."""
        have = set(self.joint_names or ())
        return [n for n in names if n not in have]


def require_clip(player: ClipPlayer, owner: str, *, joint_names: Sequence[str] | None = None,
                 root_pos: bool = False, root_quat: bool = False) -> None:
    """Raise ``ValueError`` if ``player``'s clip lacks what ``owner`` reads.

    ``joint_names``: the clip must name at least one of them (the rest are filled
    with the controller's default pose).
    """
    if joint_names is not None and len(player.covers(joint_names)) == len(joint_names):
        raise ValueError(f"the clip has none of {owner}'s joints")
    if root_pos and not player.has_root_pos:
        raise ValueError(f"{owner} reads the root position / height; the clip needs root_pos")
    if root_quat and not player.has_root_quat:
        raise ValueError(f"{owner} reads the root orientation; the clip needs root_quat")

"""Body key points from forward kinematics, for trackers that observe body positions.

KungfuBot and H2H were trained on PHC's motion library, which runs FK at the clip's
own frame rate, adds "extend" points at fixed offsets from parent bodies (hands,
head), and Gaussian-smooths (sigma = 2 frames) the finite-difference velocities.
:class:`KeyPoints` reproduces that on top of ``hirobot.geometry.kinematics.Kinematics``,
which hiWBC imports only when a clip is tracked (core hiWBC depends on numpy only).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

#: PHC's velocity smoothing (``scipy gaussian_filter1d(sigma=2)`` on source frames).
PHC_SIGMA = 2.0


def rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate ``v`` (..., 3) by unit quaternions ``q`` (..., 4) ``[x, y, z, w]``."""
    u, w = q[..., :3], q[..., 3:4]
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def conj(q: np.ndarray) -> np.ndarray:
    return q * np.array([-1.0, -1.0, -1.0, 1.0])


def load_kinematics(path: str):
    try:
        from hirobot.geometry.kinematics import Kinematics
    except ImportError as e:
        raise ImportError("tracking this controller's clips needs hirobot for body FK "
                          "(pip install hirobot)") from e
    return Kinematics.from_mjcf(path)


class KeyPoints:
    """World positions of points fixed to robot bodies.

    Args:
        kinematics: A ``hirobot.geometry.kinematics.Kinematics``, or a zero-argument callable
            returning one (loaded on first use).
        points: ``(body, offset)`` pairs; ``offset`` in the body frame, metres.
        joint_names: Names of the joint vectors passed to :meth:`positions`.
    """

    def __init__(self, kinematics: Any | Callable[[], Any],
                 points: Sequence[tuple[str, Sequence[float]]], joint_names: Sequence[str]):
        self._kin = kinematics
        self.bodies = tuple(str(b) for b, _ in points)
        self.offsets = np.array([o for _, o in points], dtype=np.float64).reshape(-1, 3)
        self.joint_names = tuple(joint_names)

    @property
    def kinematics(self):
        if not hasattr(self._kin, "forward"):
            self._kin = self._kin()
        return self._kin

    def positions(self, joint_pos, base_pos, base_quat) -> np.ndarray:
        """``(..., P, 3)`` world positions for ``(..., n_joints)`` joint angles."""
        poses = self.kinematics.forward(joint_pos, base_pos=base_pos, base_quat=base_quat,
                                        joint_names=self.joint_names)
        idx = [poses.index(b) for b in self.bodies]
        pos, quat = poses.pos[..., idx, :], poses.quat[..., idx, :]
        return pos + rotate(quat, np.broadcast_to(self.offsets, pos.shape))

    def clip_term(self, clip_joint_names: Sequence[str] | None, fill: np.ndarray,
                  with_velocity: bool) -> Callable[[dict], np.ndarray]:
        """A ``ClipPlayer.add_derived`` function: key points (and PHC-style velocities).

        Returns rows of ``P * 3`` positions, followed by ``P * 3`` velocities when
        ``with_velocity``. Joints the clip lacks are at ``fill`` (the controller's
        default pose, in :attr:`joint_names` order).
        """
        from ..references.clip import _diff, _gaussian

        clip_cols = {n: i for i, n in enumerate(clip_joint_names or ())}
        fill = np.asarray(fill, dtype=np.float64)

        def fn(frames: dict) -> np.ndarray:
            src = frames["joint_pos"]
            n = len(frames["root_pos"])
            q = np.broadcast_to(fill, (n, len(self.joint_names))).copy()
            if src is not None:
                for j, name in enumerate(self.joint_names):
                    if name in clip_cols:
                        q[:, j] = src[:, clip_cols[name]]
            p = self.positions(q, frames["root_pos"], frames["root_quat"])
            out = [p.reshape(n, -1)]
            if with_velocity:
                v = _gaussian(_diff(p, 1.0 / frames["fps"]), PHC_SIGMA)
                out.append(v.reshape(n, -1))
            return np.concatenate(out, axis=1)

        return fn

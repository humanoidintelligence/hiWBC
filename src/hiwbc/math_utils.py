"""Small, dependency-free math helpers for observation construction."""

from __future__ import annotations

import numpy as np


def gravity_orientation(quat_xyzw: np.ndarray) -> np.ndarray:
    """Projected gravity in the base frame from a [x, y, z, w] quaternion.

    This is the standard projected-gravity observation used by locomotion
    policies (the body-frame direction of the world -z axis).
    """
    qx, qy, qz, qw = quat_xyzw[0], quat_xyzw[1], quat_xyzw[2], quat_xyzw[3]
    return np.array(
        [
            2.0 * (-qz * qx + qw * qy),
            -2.0 * (qz * qy + qw * qx),
            1.0 - 2.0 * (qw * qw + qz * qz),
        ],
        dtype=np.float32,
    )


def quat_rotate_inverse(quat_xyzw: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate ``vec`` by the inverse of quaternion ``quat_xyzw`` ([x, y, z, w]).

    Pure-numpy equivalent of applying ``Rotation.from_quat(q).inv()`` to a
    vector. Motion trackers (ASAP, BeyondMimic) build projected gravity as
    ``quat_rotate_inverse(base_quat, [0, 0, -1])``.
    """
    q = np.asarray(quat_xyzw, dtype=np.float64)
    v = np.asarray(vec, dtype=np.float64)
    q_vec = q[:3]
    q_w = q[3]
    a = v * (2.0 * q_w * q_w - 1.0)
    b = np.cross(q_vec, v) * (2.0 * q_w)
    c = q_vec * (2.0 * np.dot(q_vec, v))
    return (a - b + c).astype(np.float32)


def quat_to_euler(quat_xyzw: np.ndarray) -> np.ndarray:
    """Roll, pitch, yaw (radians) from a [x, y, z, w] quaternion."""
    qx, qy, qz, qw = quat_xyzw[0], quat_xyzw[1], quat_xyzw[2], quat_xyzw[3]
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = np.copysign(np.pi / 2, sinp) if abs(sinp) >= 1 else np.arcsin(sinp)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=np.float32)


def matrix_from_quat(quat_xyzw: np.ndarray) -> np.ndarray:
    """(3, 3) rotation matrix from a [x, y, z, w] quaternion."""
    qx, qy, qz, qw = quat_xyzw[0], quat_xyzw[1], quat_xyzw[2], quat_xyzw[3]
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float32,
    )


def quat_mul(a_xyzw: np.ndarray, b_xyzw: np.ndarray) -> np.ndarray:
    """Hamilton product of two [x, y, z, w] quaternions."""
    ax, ay, az, aw = a_xyzw
    bx, by, bz, bw = b_xyzw
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float32,
    )


def quat_conjugate(quat_xyzw: np.ndarray) -> np.ndarray:
    """Conjugate (inverse for a unit quaternion) of a [x, y, z, w] quaternion."""
    q = np.asarray(quat_xyzw, dtype=np.float32)
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)


def subtract_frame_transforms_rot(q01_xyzw: np.ndarray, q02_xyzw: np.ndarray) -> np.ndarray:
    """Relative rotation q12 = q01^{-1} * q02, all [x, y, z, w]."""
    return quat_mul(quat_conjugate(q01_xyzw), q02_xyzw)

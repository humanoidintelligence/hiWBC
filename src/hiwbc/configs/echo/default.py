"""``echo/default``: no checkpoint; holds the default pose. For tests and integration.

Joints default to the 29-DoF G1 (hiRobot order). Options: ``joint_names``,
``default_pos`` (zeros if omitted), ``kp`` / ``kd`` (None keeps the robot's gains).
"""

from __future__ import annotations

from ...controllers.echo import EchoController
from ...registry import register_profile

G1_29 = (
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
    "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee",
    "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
)  # fmt: skip


def build(_variant, options: dict) -> EchoController:
    names = tuple(options["joint_names"])
    default = options["default_pos"] if options["default_pos"] is not None else [0.0] * len(names)
    return EchoController(num_actions=len(names), default_pos=default, joint_names=names,
                          kp=options["kp"], kd=options["kd"], stand_command=(0.0, 0.0, 0.0))


register_profile(
    "echo/default",
    build=build,
    variants={"default": None},
    default_options={"joint_names": G1_29, "default_pos": None, "kp": None, "kd": None},
    description="Holds the default pose; no checkpoint",
)

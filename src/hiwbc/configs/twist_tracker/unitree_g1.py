"""``twist_tracker`` on the Unitree G1 (23 joints, ``twist/twist_general_motion_tracker.pt``).

Values from RoboJuDo's ``config/g1/policy/g1_twist_policy_cfg.py``. The ankle joint
velocities (``ankle_idx = [4, 5, 10, 11]``) are zeroed in the observation.
"""

from __future__ import annotations

import numpy as np

from ...controllers.twist import TwistTrackerController
from ...registry import register_profile
from ...runtime import load_torchscript_runner
from .._common import ROBOJUDO_MODELS, checkpoint

JOINT_NAMES = (
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
    "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee",
    "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
)  # fmt: skip

DEFAULT = {
    "checkpoint": "twist/twist_general_motion_tracker.pt",
    "default_pos": (-0.2, 0.0, 0.0, 0.4, -0.2, 0.0) * 2 + (0.0,) * 3 + (0.0, 0.2, 0.0, 1.2, 0.0, -0.2, 0.0, 1.2),
    "kp": (100, 100, 100, 150, 40, 40) * 2 + (150,) * 3 + (40,) * 8,
    "kd": (2, 2, 2, 4, 2, 2) * 2 + (4,) * 3 + (5,) * 8,
    "ankle_index": (4, 5, 10, 11),
}


def build(v: dict, options: dict) -> TwistTrackerController:
    n = len(JOINT_NAMES)
    mask = np.zeros(n, dtype=np.float32)
    mask[list(v["ankle_index"])] = 1.0
    ctrl = TwistTrackerController(
        num_actions=n, default_pos=v["default_pos"], joint_names=JOINT_NAMES, kp=v["kp"], kd=v["kd"],
        forward_fn=load_torchscript_runner(checkpoint(options["checkpoint_dir"], v["checkpoint"]),
                                           device=options["device"]),
        history_length=10, ang_vel_scale=0.25, dof_pos_scale=1.0, dof_vel_scale=0.05,
        zero_dof_vel_mask=mask, action_scale=0.5, action_clip=10.0,
    )
    ctrl.reset()
    return ctrl


register_profile(
    "twist_tracker/unitree_g1",
    build=build,
    variants={"default": DEFAULT},
    default_options={"checkpoint_dir": ROBOJUDO_MODELS, "device": "cpu"},
    description="TWIST general motion tracker on the G1 (23 joints)",
)

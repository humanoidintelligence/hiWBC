"""``protomotions_tracker`` on the Unitree G1: Human2Humanoid student (21 joints, ``h2h/h2h.pt``).

Values from RoboJuDo's ``config/g1/policy/g1_h2h_policy_cfg.py``. ``task_obs`` (the
teleop imitation observation) comes from upstream or, for clips, from
``clip_reference``: hands and head of PHC's ``unitree_g1_29dof.yaml`` extend_config,
FK on the G1 29-DoF MJCF (option ``mjcf``) via ``hirobot.geometry.kinematics``. Clips must
carry 29-DoF joint angles; the policy drives 21.
"""

from __future__ import annotations

from ...controllers._bodyfk import load_kinematics
from ...controllers.protomotions import ProtoMotionsTrackerController
from ...registry import register_profile
from ...runtime import load_torchscript_runner
from .._common import ROBOJUDO_MODELS, checkpoint
from ..kungfubot_tracker.unitree_g1 import G1_MJCF

JOINT_NAMES = (
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
    "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee",
    "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
)  # fmt: skip

#: G1 29-DoF joints (FK model order).
G1_29 = (
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch",
    "left_ankle_roll", "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee",
    "right_ankle_pitch", "right_ankle_roll", "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
    "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
    "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
)  # fmt: skip

#: PHC ``unitree_g1_29dof.yaml`` extend_config (left hand, right hand, head).
TRACK_POINTS = (
    ("left_wrist_yaw_link", (0.0415, 0.003, 0.0)),
    ("right_wrist_yaw_link", (0.0415, -0.003, 0.0)),
    ("torso_link", (0.0039635, 0.0, 0.4)),
)

DEFAULT = {
    "checkpoint": "h2h/h2h.pt",
    "default_pos": (-0.1, 0.0, 0.0, 0.3, -0.2, 0.0) * 2 + (0.0,) * 9,
    "kp": (100, 100, 100, 150, 40, 40) * 2 + (200,) + (40,) * 8,
    "kd": (5, 5, 5, 5, 2, 2) * 2 + (6,) + (2,) * 8,
}


def build(v: dict, options: dict) -> ProtoMotionsTrackerController:
    ctrl = ProtoMotionsTrackerController(
        num_actions=len(JOINT_NAMES), default_pos=v["default_pos"], joint_names=JOINT_NAMES,
        kp=v["kp"], kd=v["kd"],
        forward_fn=load_torchscript_runner(checkpoint(options["checkpoint_dir"], v["checkpoint"]),
                                           device=options["device"]),
        history_length=25, dof_vel_scale=1.0, ang_vel_scale=1.0, use_dof_pos_offset=False,
        action_scale=0.25, action_clip=10.0, action_beta=0.8,
        kinematics=lambda: load_kinematics(options["mjcf"]), track_points=TRACK_POINTS,
        kinematic_joint_names=G1_29,
    )
    ctrl.reset()
    return ctrl


register_profile(
    "protomotions_tracker/unitree_g1",
    build=build,
    variants={"default": DEFAULT},
    default_options={"checkpoint_dir": ROBOJUDO_MODELS, "device": "cpu", "mjcf": G1_MJCF},
    description="Human2Humanoid student tracker on the G1 (21 joints)",
)

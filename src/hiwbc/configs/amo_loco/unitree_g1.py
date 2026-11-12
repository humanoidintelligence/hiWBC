"""``amo_loco`` on the Unitree G1: drives legs + waist (15), observes those plus the arms (23).

Values from RoboJuDo's ``config/g1/policy/g1_amo_policy_cfg.py``. The 8 arm joints
are read, not driven: under ``WbcRobot`` they are ``direct_joints`` and hold the
pose the layer above gives them. The command is 3 values (velocity) or 7 (plus
height and torso yaw / pitch / roll); see docs/guides/locomotion.md.
"""

from __future__ import annotations

from ...controllers.amo import AmoLocoController, load_amo_runners
from ...registry import register_profile
from .._common import ROBOJUDO_MODELS, checkpoint

DRIVEN = (
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
    "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee",
    "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
)  # fmt: skip
ARMS = (
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
)  # fmt: skip

DEFAULT = {
    "files": ("amo/amo_jit.pt", "amo/adapter_jit.pt", "amo/adapter_norm_stats.pt"),
    "obs_default_pos": (-0.1, 0.0, 0.0, 0.3, -0.2, 0.0) * 2 + (0.0,) * 3
    + (0.5, 0.0, 0.2, 0.3, 0.5, 0.0, -0.2, 0.3),
    "kp": (150, 150, 150, 300, 80, 20) * 2 + (400,) * 3,
    "kd": (2, 2, 2, 4, 2, 1) * 2 + (15,) * 3,
    # RoboJuDo commands_map midpoints for the 3 stick axes, then height 0.75 and torso 0.
    "default_command": (0.0, 0.0, 0.0, 0.75, 0.0, 0.0, 0.0),
}


def build(v: dict, options: dict) -> AmoLocoController:
    files = [checkpoint(options["checkpoint_dir"], f) for f in v["files"]]
    ctrl = AmoLocoController(
        num_actions=len(DRIVEN), default_pos=v["obs_default_pos"][: len(DRIVEN)],
        obs_default_pos=v["obs_default_pos"], joint_names=DRIVEN, obs_joint_names=DRIVEN + ARMS,
        kp=v["kp"], kd=v["kd"], ang_vel_scale=0.25, dof_vel_scale=0.05, action_scale=0.25,
        default_command=v["default_command"],
        **load_amo_runners(*map(str, files), device=options["device"]),
    )
    ctrl.reset()
    return ctrl


register_profile(
    "amo_loco/unitree_g1",
    build=build,
    variants={"default": DEFAULT},
    default_options={"checkpoint_dir": ROBOJUDO_MODELS, "device": "cpu"},
    description="AMO on the G1: velocity + height / torso orientation (legs and waist)",
)

"""``kungfubot_tracker`` on the Unitree G1 (23 joints, ``kungfubot2/horse_test_43000.onnx``).

Values from RoboJuDo's ``g1_kungfubot_policy_cfg.py`` (``G1KungfuBotGeneralPolicyCfg``,
``G1_23KongfuGBotGeneralDoF``). RoboJuDo calls this checkpoint a test model trained
on one motion. The model has three inputs: ``actor_obs``, ``future_motion_targets``,
``prop_history``.

Clips: the key points and future window follow RoboJuDo ``G1MotionKungfuBotCtrlCfg``
and PHC's ``unitree_g1_23dof_KungfuBot.yaml`` (knees, ankles, elbows, plus hand
and head points offset from the elbows and torso). Body FK uses
``hirobot.geometry.kinematics`` on the G1 29-DoF MJCF (option ``mjcf``), whose kinematic
tree matches PHC's model; the 6 wrist joints the policy lacks stay at 0.
"""

from __future__ import annotations

from ...controllers._bodyfk import load_kinematics
from ...controllers.kungfubot import KungfuBotTrackerController
from ...registry import register_profile
from ...runtime import load_onnx_runner
from .._common import ROBOJUDO_MODELS, checkpoint

#: G1 29-DoF MJCF from unitreerobotics/unitree_ros (hiRobot's ``unitree_g1_mujoco`` model).
G1_MJCF = "~/.cache/hirobot/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.xml"

#: ``linspace(1, 95, 20)`` truncated (RoboJuDo ``future_max_steps`` / ``future_num_steps``).
FUTURE_STEPS = (1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95)

#: ``key_body_id`` 4, 6, 10, 12, 19, 23 of the KungfuBot body list, then its
#: ``extend_config`` (left / right hand, head).
KEY_BODIES = (
    ("left_knee_link", (0, 0, 0)), ("left_ankle_roll_link", (0, 0, 0)),
    ("right_knee_link", (0, 0, 0)), ("right_ankle_roll_link", (0, 0, 0)),
    ("left_elbow_link", (0, 0, 0)), ("right_elbow_link", (0, 0, 0)),
    ("left_elbow_link", (0.25, 0, 0)), ("right_elbow_link", (0.25, 0, 0)),
    ("torso_link", (0, 0, 0.42)),
)

JOINT_NAMES = (
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
    "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee",
    "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
)  # fmt: skip

# Actuator model (BeyondMimic-style): kp = armature * (2*pi*10 Hz)^2, kd = 2 * 2 * armature * 2*pi*10.
_KP_7520_14, _KP_7520_22, _KP_5020 = 40.17923847137318, 99.09842777666113, 28.50124619574858
_KD_7520_14, _KD_7520_22, _KD_5020 = 2.5578897650279457, 6.3088018534966395, 1.814445686584846
_KP_ARM, _KD_ARM = 14.25062309787429, 0.907222843292423
_SCALE_14, _SCALE_22, _SCALE_5020 = 0.5475464652142303, 0.3506614663788243, 0.43857731392336724

DEFAULT = {
    "checkpoint": "kungfubot2/horse_test_43000.onnx",
    "default_pos": (-0.1, 0.0, 0.0, 0.3, -0.2, 0.0) * 2 + (0.0,) * 3 + (0.2, 0.2, 0.0, 0.9, 0.2, -0.2, 0.0, 0.9),
    "kp": (_KP_7520_14, _KP_7520_22, _KP_7520_14, _KP_7520_22, _KP_5020, _KP_5020) * 2
    + (_KP_7520_14, _KP_5020, _KP_5020) + (_KP_ARM,) * 8,
    "kd": (_KD_7520_14, _KD_7520_22, _KD_7520_14, _KD_7520_22, _KD_5020, _KD_5020) * 2
    + (_KD_7520_14, _KD_5020, _KD_5020) + (_KD_ARM,) * 8,
    "action_scale": (_SCALE_14, _SCALE_22, _SCALE_14, _SCALE_22, _SCALE_5020, _SCALE_5020) * 2
    + (_SCALE_14,) + (_SCALE_5020,) * 10,
}


def build(v: dict, options: dict) -> KungfuBotTrackerController:
    ctrl = KungfuBotTrackerController(
        num_actions=len(JOINT_NAMES), default_pos=v["default_pos"], joint_names=JOINT_NAMES,
        kp=v["kp"], kd=v["kd"], action_scale=v["action_scale"], action_clip=100.0,
        forward_fn=load_onnx_runner(checkpoint(options["checkpoint_dir"], v["checkpoint"]),
                                    input_names=["actor_obs", "future_motion_targets", "prop_history"],
                                    output_name="action", providers=options["providers"]),
        history_length=10, ang_vel_scale=0.25, dof_pos_scale=1.0, dof_vel_scale=0.05,
        future_steps=FUTURE_STEPS, key_bodies=KEY_BODIES, pbhc_anchor_quat_order=True,
        kinematics=lambda: load_kinematics(options["mjcf"]),
    )
    ctrl.reset()
    return ctrl


register_profile(
    "kungfubot_tracker/unitree_g1",
    build=build,
    variants={"default": DEFAULT},
    default_options={"checkpoint_dir": ROBOJUDO_MODELS, "providers": None, "mjcf": G1_MJCF},
    description="KungfuBot2 (PBHC) general tracker on the G1 (23 joints; test checkpoint)",
)

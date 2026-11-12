"""``asap_tracker`` on the Unitree G1 (23 joints): one motion baked into each checkpoint.

Values from RoboJuDo's ``g1_asap_policy_cfg.py`` (``G1AsapPolicyCfg``),
``g1_kungfubot_policy_cfg.py`` (``G1KungfuBotPolicyCfg``) and the ``robomimic``
entry of ``g1_loco_mimic_cfg.py``. Each variant is one motion; the controller
exposes it as a behavior. The motion must start from its first pose
(``BehaviorSpec.start_pos``: legs at default, upper body at the listed pose);
starting CR7 from the default pose makes it fall. See docs/guides/motion_tracking.md.

The tracker cannot stand on its own (no ``stand_reference``); ``WbcRobot`` pairs
it with a standing controller.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...controllers.asap import AsapObsScales, AsapTrackerController
from ...registry import register_profile
from ...runtime import load_onnx_runner
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

_LEGS_DEFAULT = (-0.1, 0.0, 0.0, 0.3, -0.2, 0.0) * 2


@dataclass(frozen=True)
class Variant:
    motion: str
    checkpoint: str
    motion_length_s: float
    default_pos: tuple[float, ...]
    start_upper: tuple[float, ...]   # waist (3) + arms (8): the motion's first pose
    kp: tuple[float, ...]
    kd: tuple[float, ...]


# G1AsapPolicyCfg's G1_23AsapDoF gains (CR7 and robomimic).
_ASAP_KP = (100, 100, 100, 200, 20, 20) * 2 + (400,) * 3 + (90, 60, 20, 60) * 2
_ASAP_KD = (2.5, 2.5, 2.5, 5.0, 0.2, 0.1) * 2 + (5.0,) * 3 + (2.0, 1.0, 0.4, 1.0) * 2

CR7 = Variant(
    motion="cr7_level1", checkpoint="asap/mimic/CR7_level1/model_191500.onnx", motion_length_s=3.967,
    default_pos=_LEGS_DEFAULT + (0.0,) * 11,
    start_upper=(0.19964170455932617, 0.07710712403059006, -0.2882401943206787,
                 0.21672365069389343, 0.15629297494888306, -0.5167576670646667, 0.5782126784324646,
                 0.25740593671798706, -0.2504104673862457, 0.22500675916671753, 0.5127624273300171),
    kp=_ASAP_KP, kd=_ASAP_KD,
)

HORSE_SQUAT = Variant(  # KungfuBot v1 (PBHC), ASAP observation format
    motion="horse_squat", checkpoint="asap/mimic/kongfu/horse_squat.onnx", motion_length_s=6.67,
    default_pos=_LEGS_DEFAULT + (0.0,) * 3 + (0.2, 0.2, 0.0, 0.9, 0.2, -0.2, 0.0, 0.9),
    start_upper=(-5.0488499e-04, 7.1762636e-04, 2.9380890e-03,
                 2.0567834e-01, 1.9635735e-01, 4.9880091e-03, 8.9015263e-01,
                 2.0773219e-01, -1.9702259e-01, -2.3902415e-03, 8.8817328e-01),
    kp=(100, 100, 100, 150, 40, 40) * 2 + (400,) * 3 + (100, 100, 50, 50) * 2,
    kd=(2, 2, 2, 4, 2, 2) * 2 + (5,) * 3 + (2,) * 8,
)

DANCE_0605 = Variant(
    motion="dance_0605", checkpoint="asap/mimic/robomimic/dance_0605.onnx", motion_length_s=18.0,
    default_pos=_LEGS_DEFAULT + (0.0,) * 11,
    start_upper=(0.0, 0.0, 0.0, 0.35, 0.18, 0.0, 0.87, 0.35, -0.18, 0.0, 0.87),
    kp=_ASAP_KP, kd=_ASAP_KD,
)


def build(v: Variant, options: dict) -> AsapTrackerController:
    ctrl = AsapTrackerController(
        num_actions=len(JOINT_NAMES), default_pos=v.default_pos, joint_names=JOINT_NAMES,
        kp=v.kp, kd=v.kd,
        forward_fn=load_onnx_runner(checkpoint(options["checkpoint_dir"], v.checkpoint),
                                    providers=options["providers"]),
        motion_length_s=v.motion_length_s, use_history=True, history_length=4,
        obs_scales=AsapObsScales(base_ang_vel=0.25, dof_vel=0.05),
        action_scale=0.25, action_clip=100.0,
        motion_name=v.motion, start_pos=v.default_pos[:12] + v.start_upper,
    )
    ctrl.reset()
    return ctrl


register_profile(
    "asap_tracker/unitree_g1",
    build=build,
    variants={"default": CR7, "horse_squat": HORSE_SQUAT, "dance_0605": DANCE_0605},
    default_options={"checkpoint_dir": ROBOJUDO_MODELS, "providers": None},
    description="ASAP motion trackers on the G1, one baked-in motion per variant",
)

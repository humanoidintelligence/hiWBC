"""``unitree_loco`` on the Unitree G1: unitree_rl_gym (gait, 12 legs) and unitree_rl_lab (29 DoF).

Values from RoboJuDo's ``config/g1/policy/g1_unitree_policy_cfg.py``
(``G1UnitreePolicyCfg`` / ``G1UnitreeWoGaitPolicyCfg``); see docs/guides/locomotion.md.
Variants:

* ``default``: ``policy_lstm_1.pt``, 12 leg joints, gait clock. Recurrent: the LSTM
  state is reset with the controller.
* ``wo_gait``: ``policy_wo_gait.pt``, all 29 joints (IsaacLab order), 5-frame history.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...controllers.unitree import ObsScales, UnitreeLocoController
from ...registry import register_profile
from ...runtime import load_torchscript_runner
from .._common import ROBOJUDO_MODELS, checkpoint

LEGS = (
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
    "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee",
    "right_ankle_pitch", "right_ankle_roll",
)  # fmt: skip

#: IsaacLab joint order (unitree_rl_lab).
G1_29_ISAACLAB = (
    "left_hip_pitch", "right_hip_pitch", "waist_yaw",
    "left_hip_roll", "right_hip_roll", "waist_roll",
    "left_hip_yaw", "right_hip_yaw", "waist_pitch",
    "left_knee", "right_knee", "left_shoulder_pitch", "right_shoulder_pitch",
    "left_ankle_pitch", "right_ankle_pitch", "left_shoulder_roll", "right_shoulder_roll",
    "left_ankle_roll", "right_ankle_roll", "left_shoulder_yaw", "right_shoulder_yaw",
    "left_elbow", "right_elbow", "left_wrist_roll", "right_wrist_roll",
    "left_wrist_pitch", "right_wrist_pitch", "left_wrist_yaw", "right_wrist_yaw",
)  # fmt: skip


@dataclass(frozen=True)
class Variant:
    checkpoint: str
    joint_names: tuple[str, ...]
    default_pos: tuple[float, ...]
    kp: tuple[float, ...]
    kd: tuple[float, ...]
    params: dict


GAIT = Variant(
    checkpoint="unitree/policy_lstm_1.pt",
    joint_names=LEGS,
    default_pos=(-0.1, 0.0, 0.0, 0.3, -0.2, 0.0) * 2,
    kp=(100, 100, 100, 150, 40, 40) * 2,
    kd=(2, 2, 2, 4, 2, 2) * 2,
    params={"with_gait": True, "cycle_time": 0.8, "max_cmd": (0.8, 0.5, 1.57),
                "obs_scales": ObsScales(dof_pos=1.0, dof_vel=0.05, ang_vel=0.25, command=(2.0, 2.0, 0.25)),
                "action_scale": 0.25, "action_clip": None, "action_beta": 0.8},
)

WO_GAIT = Variant(
    checkpoint="unitree/policy_wo_gait.pt",
    joint_names=G1_29_ISAACLAB,
    default_pos=(-0.1, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.3, 0.3, -0.2, -0.2,
                 0.25, -0.25, 0.0, 0.0, 0.0, 0.0, 0.97, 0.97, 0.15, -0.15, 0.0, 0.0, 0.0, 0.0),
    kp=(100, 100, 200, 100, 100, 40, 100, 100, 40, 150, 150) + (40,) * 18,
    kd=(2, 2, 5, 2, 2, 5, 2, 2, 5, 4, 4, 10, 10, 2, 2, 10, 10, 2, 2) + (10,) * 10,
    params={"with_gait": False, "history_length": 5, "gravity_scale": 1.0, "max_cmd": (0.8, 0.5, 1.57),
                "obs_scales": ObsScales(dof_pos=1.0, dof_vel=0.05, ang_vel=0.2, command=(1.0, 1.0, 1.0)),
                "action_scale": 0.25, "action_clip": None, "action_beta": 1.0},
)


def build(v: Variant, options: dict) -> UnitreeLocoController:
    fn = load_torchscript_runner(checkpoint(options["checkpoint_dir"], v.checkpoint), device=options["device"])
    ctrl = UnitreeLocoController(
        num_actions=len(v.joint_names), default_pos=v.default_pos, forward_fn=fn,
        joint_names=v.joint_names, kp=v.kp, kd=v.kd, **v.params)
    ctrl.reset()  # also restores the LSTM's hidden state
    return ctrl


register_profile(
    "unitree_loco/unitree_g1",
    build=build,
    variants={"default": GAIT, "wo_gait": WO_GAIT},
    default_options={"checkpoint_dir": ROBOJUDO_MODELS, "device": "cpu"},
    description="Unitree's G1 velocity policies (unitree_rl_gym LSTM / unitree_rl_lab)",
)

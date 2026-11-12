"""``smooth_loco`` on the Unitree G1 (21 joints, ``smooth/for_test-2000-jit.pt``).

Values from RoboJuDo's ``config/g1/policy/g1_smooth_policy_cfg.py``. The standing
command is ``[-0.5, 0, -0.5]`` (the joystick neutral of ``commands_map``), not zero;
see docs/guides/locomotion.md. RoboJuDo ships this checkpoint as a test model.
"""

from __future__ import annotations

from ...controllers.smooth import SmoothLocoController, SmoothObsScales
from ...registry import register_profile
from ...runtime import load_torchscript_runner
from .._common import ROBOJUDO_MODELS, checkpoint

JOINT_NAMES = (
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
    "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee",
    "right_ankle_pitch", "right_ankle_roll",
    "waist_yaw",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw", "right_elbow",
)  # fmt: skip

DEFAULT = {
    "checkpoint": "smooth/for_test-2000-jit.pt",
    "default_pos": (-0.4, 0.0, 0.0, 0.8, -0.35, 0.0) * 2 + (0.0,) * 9,
    "kp": (200, 150, 150, 200, 20, 20) * 2 + (200,) + (40,) * 8,
    "kd": (5, 5, 5, 5, 4, 4) * 2 + (5,) + (10,) * 8,
}


def build(v: dict, options: dict) -> SmoothLocoController:
    n = len(JOINT_NAMES)
    ctrl = SmoothLocoController(
        num_actions=n, default_pos=v["default_pos"], joint_names=JOINT_NAMES, kp=v["kp"], kd=v["kd"],
        forward_fn=load_torchscript_runner(checkpoint(options["checkpoint_dir"], v["checkpoint"]),
                                           device=options["device"]),
        cycle_time=0.64,
        n_priv_latent=4 + 1 + 2 * n + 3,  # RoboJuDo SmoothPolicy: zeros of this size
        history_length=10,
        obs_scales=SmoothObsScales(ang_vel=0.25, dof_vel=0.05),
        action_scale=0.5, action_clip=10.0, action_beta=0.8,
        stand_command=(-0.5, 0.0, -0.5),
    )
    ctrl.reset()
    return ctrl


register_profile(
    "smooth_loco/unitree_g1",
    build=build,
    variants={"default": DEFAULT},
    default_options={"checkpoint_dir": ROBOJUDO_MODELS, "device": "cpu"},
    description="Smooth Humanoid Locomotion on the G1 (21 joints; test checkpoint)",
)

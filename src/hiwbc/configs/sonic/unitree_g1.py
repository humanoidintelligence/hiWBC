"""SONIC on the 29-DoF Unitree G1: robot tables and the released checkpoints' layouts.

Values are transcribed from NVIDIA GR00T-WholeBodyControl's C++ deployment
(``gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp``
and ``policy/release/observation_config*.yaml``). The deployment stores joint
arrays in MuJoCo (hardware) order; SONIC's policy runs in IsaacLab order. The
tables below keep the MuJoCo order of the source for auditability and are
permuted to IsaacLab order once, at import. Joint names are hiRobot's
(``left_hip_pitch``, not ``left_hip_pitch_joint``).

Each released checkpoint ships an ``observation_config.yaml`` that fixes the
encoder input layout; the presets mirror those files term for term. Term order
is safety-critical: it must match the ONNX model, or the robot is commanded with
garbage. The deployment calls the joint-space encoder mode ``g1``; hiWBC calls it
``joint`` (same id, 0).
"""

from __future__ import annotations

import os

import numpy as np

from ...controllers.sonic import ENCODER_MODE_IDS, SonicObsConfig, SonicRobotConfig
from ...registry import register_profile

G1_NUM_JOINTS = 29

#: G1 joint names in MuJoCo / hardware order, hiRobot's names (no ``_joint`` suffix).
G1_JOINT_NAMES_MUJOCO = (
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

#: For each IsaacLab joint index, its MuJoCo index (``mujoco_to_isaaclab`` in the source).
ISAACLAB_TO_MUJOCO_INDEX = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
     16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]
)  # fmt: skip

#: G1 joint names in IsaacLab order: the controller's own joint order.
G1_JOINT_NAMES = tuple(G1_JOINT_NAMES_MUJOCO[i] for i in ISAACLAB_TO_MUJOCO_INDEX)

#: Lower-body joints, MuJoCo order, as IsaacLab indices (teleop encoder).
G1_LOWER_BODY_INDEX = np.array([0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18])
#: Wrist joints, IsaacLab order, as IsaacLab indices (SMPL encoder).
G1_WRIST_INDEX = np.array([23, 24, 25, 26, 27, 28])

# ----- actuator model (policy_parameters.hpp) -------------------------------
_NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10 Hz
_DAMPING_RATIO = 2.0
_ARMATURE = {"5020": 0.003609725, "7520_14": 0.010177520, "7520_22": 0.025101925, "4010": 0.00425}
_EFFORT_LIMIT = {"5020": 25.0, "7520_14": 88.0, "7520_22": 139.0, "4010": 5.0}
_STIFFNESS = {m: a * _NATURAL_FREQ**2 for m, a in _ARMATURE.items()}
_DAMPING = {m: 2.0 * _DAMPING_RATIO * a * _NATURAL_FREQ for m, a in _ARMATURE.items()}

# Motor per joint, MuJoCo order.
_MOTORS = (
    ["7520_22", "7520_22", "7520_14", "7520_22", "5020", "5020"] * 2  # legs
    + ["7520_14", "5020", "5020"]  # waist
    + ["5020", "5020", "5020", "5020", "5020", "4010", "4010"] * 2  # arms
)
# Joints whose PD gains are doubled: ankles and waist roll / pitch.
_DOUBLED_GAIN = {4, 5, 10, 11, 13, 14}

_DEFAULT_POS_MUJOCO = np.array(
    [-0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
     -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
     0.0, 0.0, 0.0,
     0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
     0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0]
)  # fmt: skip
_ACTION_SCALE_MUJOCO = np.array([0.25 * _EFFORT_LIMIT[m] / _STIFFNESS[m] for m in _MOTORS])
_KP_MUJOCO = np.array(
    [_STIFFNESS[m] * (2.0 if i in _DOUBLED_GAIN else 1.0) for i, m in enumerate(_MOTORS)]
)
_KD_MUJOCO = np.array(
    [_DAMPING[m] * (2.0 if i in _DOUBLED_GAIN else 1.0) for i, m in enumerate(_MOTORS)]
)

#: Standing pose, per-joint action scale, and PD gains, all in IsaacLab order.
G1_DEFAULT_POS = _DEFAULT_POS_MUJOCO[ISAACLAB_TO_MUJOCO_INDEX].astype(np.float32)
G1_ACTION_SCALE = _ACTION_SCALE_MUJOCO[ISAACLAB_TO_MUJOCO_INDEX].astype(np.float32)
G1_KP = _KP_MUJOCO[ISAACLAB_TO_MUJOCO_INDEX].astype(np.float32)
G1_KD = _KD_MUJOCO[ISAACLAB_TO_MUJOCO_INDEX].astype(np.float32)


# ----- observation layouts ---------------------------------------------------


def _modes(joint: tuple[str, ...], teleop: tuple[str, ...], smpl: tuple[str, ...]):
    ids = ENCODER_MODE_IDS
    return {
        "joint": (ids["joint"], ("encoder_mode_4", *joint)),
        "teleop": (ids["teleop"], ("encoder_mode_4", *teleop)),
        "smpl": (ids["smpl"], ("encoder_mode_4", *smpl)),
    }


#: Default SONIC (``policy/release/observation_config.yaml``): encoder 1762, decoder 994.
SONIC_DEFAULT = SonicObsConfig(
    encoder_terms=(
        "encoder_mode_4",
        "motion_joint_positions_10frame_step5",
        "motion_joint_velocities_10frame_step5",
        "motion_root_z_position_10frame_step5",
        "motion_root_z_position",
        "motion_anchor_orientation",
        "motion_anchor_orientation_10frame_step5",
        "motion_joint_positions_lowerbody_10frame_step5",
        "motion_joint_velocities_lowerbody_10frame_step5",
        "vr_3point_local_target",
        "vr_3point_local_orn_target",
        "smpl_joints_10frame_step1",
        "smpl_anchor_orientation_10frame_step1",
        "motion_joint_positions_wrists_10frame_step1",
    ),
    encoder_modes=_modes(
        joint=(
            "motion_joint_positions_10frame_step5",
            "motion_joint_velocities_10frame_step5",
            "motion_anchor_orientation_10frame_step5",
        ),
        teleop=(
            "motion_joint_positions_lowerbody_10frame_step5",
            "motion_joint_velocities_lowerbody_10frame_step5",
            "vr_3point_local_target",
            "vr_3point_local_orn_target",
            "motion_anchor_orientation",
        ),
        smpl=(
            "smpl_joints_10frame_step1",
            "smpl_anchor_orientation_10frame_step1",
            "motion_joint_positions_wrists_10frame_step1",
        ),
    ),
)

#: Low-latency teleoperation (``low_latency/observation_config.yaml``): encoder 1247.
SONIC_LOW_LATENCY = SonicObsConfig(
    encoder_terms=(
        "encoder_mode_4",
        "motion_joint_positions_10frame_step1",
        "motion_joint_velocities_10frame_step1",
        "motion_anchor_orientation_10frame_step1",
        "motion_anchor_orientation",
        "motion_joint_positions_lowerbody_10frame_step1",
        "motion_joint_velocities_lowerbody_10frame_step1",
        "vr_3point_local_target",
        "vr_3point_local_orn_target",
        "smpl_joints_4frame_step1",
        "smpl_anchor_orientation_4frame_step1",
        "motion_joint_positions_wrists_4frame_step1",
    ),
    encoder_modes=_modes(
        joint=(
            "motion_joint_positions_10frame_step1",
            "motion_joint_velocities_10frame_step1",
            "motion_anchor_orientation_10frame_step1",
        ),
        teleop=(
            "motion_joint_positions_lowerbody_10frame_step1",
            "motion_joint_velocities_lowerbody_10frame_step1",
            "vr_3point_local_target",
            "vr_3point_local_orn_target",
            "motion_anchor_orientation",
        ),
        smpl=(
            "smpl_joints_4frame_step1",
            "smpl_anchor_orientation_4frame_step1",
            "motion_joint_positions_wrists_4frame_step1",
        ),
    ),
)

#: SONIC v1.1 (``sonic_v1_1/observation_config.yaml``): encoder 1751, heading-normalized.
SONIC_V1_1 = SonicObsConfig(
    encoder_terms=(
        "encoder_mode_4",
        "motion_joint_positions_10frame_step5",
        "motion_joint_velocities_10frame_step5",
        "motion_anchor_orientation_heading_10frame_step5",
        "motion_anchor_orientation_heading",
        "motion_joint_positions_lowerbody_10frame_step5",
        "motion_joint_velocities_lowerbody_10frame_step5",
        "vr_3point_local_target",
        "vr_3point_local_orn_target",
        "smpl_joints_10frame_step1",
        "smpl_anchor_orientation_heading_10frame_step1",
        "motion_joint_positions_wrists_10frame_step1",
    ),
    encoder_modes=_modes(
        joint=(
            "motion_joint_positions_10frame_step5",
            "motion_joint_velocities_10frame_step5",
            "motion_anchor_orientation_heading_10frame_step5",
        ),
        teleop=(
            "motion_joint_positions_lowerbody_10frame_step5",
            "motion_joint_velocities_lowerbody_10frame_step5",
            "vr_3point_local_target",
            "vr_3point_local_orn_target",
            "motion_anchor_orientation_heading",
        ),
        smpl=(
            "smpl_joints_10frame_step1",
            "smpl_anchor_orientation_heading_10frame_step1",
            "motion_joint_positions_wrists_10frame_step1",
        ),
    ),
)

#: Observation layout per released checkpoint (the profile's ``variant`` option).
SONIC_VARIANTS = {
    "default": SONIC_DEFAULT,
    "low_latency": SONIC_LOW_LATENCY,
    "sonic_v1_1": SONIC_V1_1,
}


#: The 29-DoF G1 as SONIC sees it (IsaacLab joint order).
SONIC_G1 = SonicRobotConfig(
    joint_names=G1_JOINT_NAMES,
    default_pos=G1_DEFAULT_POS,
    action_scale=G1_ACTION_SCALE,
    kp=G1_KP,
    kd=G1_KD,
    lower_body_index=G1_LOWER_BODY_INDEX,
    wrist_index=G1_WRIST_INDEX,
)


# ----- profile ----------------------------------------------------------------
#: Root of the released checkpoints (``huggingface-cli download nvidia/GEAR-SONIC``).
SONIC_CHECKPOINTS = "~/.cache/hiwbc/sonic"
#: Where each variant's ONNX pair sits under the checkpoint root.
_VARIANT_SUBDIR = {"default": "", "low_latency": "low_latency", "sonic_v1_1": "sonic_v1_1"}
#: Standing root height for ``stand_reference`` (the deployment's default base z).
G1_STAND_ROOT_Z = 0.793
#: ``[vx, vy, wz]`` per-axis (min, max) for ``BASE_VELOCITY`` through the planner (m/s, rad/s).
SONIC_VELOCITY_LIMITS = ((-0.8, 1.0), (-0.5, 0.5), (-1.0, 1.0))

#: The locomotion planner, shared by every variant (``planner_sonic.onnx``, V2).
SONIC_PLANNER = "planner_sonic.onnx"


def _files(obs: SonicObsConfig, options: dict) -> dict[str, str]:
    """The checkpoint files ``_build`` opens: decoder, then encoder and planner if used."""
    sub = _VARIANT_SUBDIR[next(k for k, v in SONIC_VARIANTS.items() if v is obs)]
    root = os.path.join(options["checkpoint_dir"], sub)
    files = {"decoder": os.path.join(root, "model_decoder.onnx")}
    if options["encoder"]:
        files["encoder"] = os.path.join(root, "model_encoder.onnx")
    if options["planner"]:
        files["planner"] = os.path.join(options["checkpoint_dir"], SONIC_PLANNER)
    return {k: os.path.expanduser(v) for k, v in files.items()}


def _info(obs: SonicObsConfig, options: dict) -> dict:
    planner = options["planner"] and options["encoder"]
    return {"joint_names": SONIC_G1.joint_names,
            "velocity_limits": SONIC_VELOCITY_LIMITS if planner else None,
            "files": list(_files(obs, options).values())}


def _build(obs: SonicObsConfig, options: dict):
    from ...controllers.sonic import SonicController
    from .._common import checkpoint

    files = _files(obs, options)
    planner = None
    if options["planner"]:
        if not options["encoder"]:
            raise ValueError("planner=True needs encoder=True")
        from ...controllers.sonic_planner import SonicPlanner, load_planner_runner

        planner = SonicPlanner(
            load_planner_runner(str(checkpoint(files["planner"])),
                                providers=options["providers"]),
            qpos_joint_index=ISAACLAB_TO_MUJOCO_INDEX,
            lower_body_index=G1_LOWER_BODY_INDEX,
            control_dt=0.02,
            background=options["planner_thread"],
        )
    ctrl = SonicController.from_onnx(
        str(checkpoint(files["decoder"])),
        str(checkpoint(files["encoder"])) if options["encoder"] else None,
        providers=options["providers"],
        robot=SONIC_G1, obs=obs, encoder_mode=options["encoder_mode"],
        stand_root_z=G1_STAND_ROOT_Z, planner=planner,
        # Each axis walked 4 s without falling on the free MuJoCo G1 at these values.
        velocity_limits=SONIC_VELOCITY_LIMITS if planner else None,
        cartesian_targets={"left_hand": "left_wrist", "right_hand": "right_wrist"} if planner else None,
    )
    ctrl.reset()
    return ctrl


register_profile(
    "sonic/unitree_g1",
    build=_build,
    variants=SONIC_VARIANTS,
    default_options={"checkpoint_dir": SONIC_CHECKPOINTS, "providers": None, "encoder": True,
                     "encoder_mode": "joint", "planner": True, "planner_thread": False},
    description="NVIDIA GEAR-SONIC on the 29-DoF G1",
    info=_info,
)

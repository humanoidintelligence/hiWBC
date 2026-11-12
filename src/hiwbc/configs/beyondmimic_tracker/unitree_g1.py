"""``beyondmimic_tracker`` on the Unitree G1 (29 joints, IsaacLab order).

Joint order, default pose, gains and per-joint action scale come from each ONNX
file's metadata, as RoboJuDo's ``use_modelmeta_config=True`` does; the profile
checks them against the expected joint set at load. ``*_wose`` checkpoints run
without a state estimator (no anchor position or base linear velocity in the
observation).

Each checkpoint's motion is exposed as a behavior (``dance``, ``jump``,
``violin``, ``waltz``) that plays the reference the ONNX file outputs for each
``time_step``. ``frames`` is the motion length, measured: the first ``time_step``
from which all reference outputs stop changing, plus one. For references from
elsewhere ``time_step`` is fed as 0; the released models' ``actions`` output does
not depend on it (checked for all four).
"""

from __future__ import annotations

import numpy as np

from ...controllers.beyondmimic import BeyondMimicTrackerController
from ...registry import register_profile
from ...runtime import load_onnx_runner
from .._common import ROBOJUDO_MODELS, checkpoint, strip_joint

VARIANTS = {
    "dance_wose": {"file": "beyondmimic/Dance_wose.onnx", "without_state_estimator": True,
                   "motion": "dance", "frames": 6574},
    "jump_wose": {"file": "beyondmimic/Jump_wose.onnx", "without_state_estimator": True,
                  "motion": "jump", "frames": 165},
    "violin": {"file": "beyondmimic/Violin.onnx", "without_state_estimator": False,
               "motion": "violin", "frames": 612},
    "waltz": {"file": "beyondmimic/Waltz.onnx", "without_state_estimator": False,
              "motion": "waltz", "frames": 945},
}
_MOTION_OUTPUTS = ("actions", "joint_pos", "joint_vel", "body_quat_w")

_G1_29 = {
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee", "left_ankle_pitch",
    "left_ankle_roll", "right_hip_pitch", "right_hip_roll", "right_hip_yaw", "right_knee",
    "right_ankle_pitch", "right_ankle_roll", "waist_yaw", "waist_roll", "waist_pitch",
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw", "left_elbow",
    "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw", "right_shoulder_pitch",
    "right_shoulder_roll", "right_shoulder_yaw", "right_elbow", "right_wrist_roll",
    "right_wrist_pitch", "right_wrist_yaw",
}  # fmt: skip


def read_metadata(path) -> dict:
    """The joint table an exported BeyondMimic ONNX carries in its custom metadata."""
    import onnxruntime as ort

    meta = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).get_modelmeta()
    m = meta.custom_metadata_map

    def floats(key):
        return np.array([float(x) for x in m[key].split(",")], dtype=np.float32)

    names = strip_joint(m["joint_names"].split(","))
    table = {"joint_names": names, "default_pos": floats("default_joint_pos"),
             "action_scale": floats("action_scale"), "kp": floats("joint_stiffness"),
             "kd": floats("joint_damping")}
    if set(names) != _G1_29 or any(len(v) != len(names) for v in table.values()):
        raise ValueError(f"{path}: metadata is not a 29-DoF G1 joint table")
    table["anchor_body_index"] = m["body_names"].split(",").index(m["anchor_body_name"])
    return table


def motion_runner(path, providers=None):
    """``motion_fn(obs, time_step) -> {actions, joint_pos, joint_vel, body_quat_w}``."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=list(providers or ["CPUExecutionProvider"]))

    def run(obs, time_step):
        feed = {"obs": np.asarray(obs, np.float32).reshape(1, -1),
                "time_step": np.array([[int(time_step)]], np.float32)}
        return {k: v[0] for k, v in zip(_MOTION_OUTPUTS, session.run(list(_MOTION_OUTPUTS), feed))}

    return run


def build(v: dict, options: dict) -> BeyondMimicTrackerController:
    path = checkpoint(options["checkpoint_dir"], v["file"])
    t = read_metadata(path)
    ctrl = BeyondMimicTrackerController(
        num_actions=len(t["joint_names"]), default_pos=t["default_pos"], joint_names=t["joint_names"],
        kp=t["kp"], kd=t["kd"], action_scale=t["action_scale"],
        forward_fn=load_onnx_runner(path, input_names=["obs"], output_name="actions",
                                    extra_inputs={"time_step": np.zeros((1, 1))},
                                    providers=options["providers"]),
        without_state_estimator=v["without_state_estimator"],
        motion_fn=motion_runner(path, options["providers"]), motion_name=v["motion"],
        motion_frames=v["frames"], anchor_body_index=t["anchor_body_index"],
    )
    ctrl.reset()
    return ctrl


register_profile(
    "beyondmimic_tracker/unitree_g1",
    build=build,
    variants=VARIANTS,
    default_options={"checkpoint_dir": ROBOJUDO_MODELS, "providers": None},
    description="BeyondMimic whole-body trackers on the G1 (29 joints)",
)

"""``blinkmimic`` on the 29-DoF Unitree G1 (BlinkMimic ``.pt`` / ``.onnx`` or XMimic ``.npz``).

Neither project releases weights; train your own. Checkpoints trained on LAFAN1
are CC BY-NC-ND (non-commercial use only). Default pose, action scale, gains and
effort limits are BlinkMimic's G1 actuator model (``G1_*`` in
:mod:`hiwbc.controllers.blinkmimic`). See docs/guides/motion_tracking.md.

Variants:

* ``default``: a BlinkMimic ``.pt`` checkpoint (:func:`load_pt_runner`, needs the
  ``torch`` extra to read it; inference is numpy, float32).
* ``xmimic``: an XMimic ``ckpt_*.npz`` (:func:`load_npz_runner`, numpy only),
  ``precision`` ``"bf16"`` (XMimic's evaluator) or ``"fp32"``.
* ``onnx``: a BlinkMimic ``tools/export_onnx.py`` export with one motion baked in
  (:func:`load_onnx_motion`, needs the ``onnx`` extra). It only plays that motion,
  as a behavior named after the metadata's ``motion`` field (else the file stem),
  e.g. ``dance1_subject2``. The metadata's joint names, gains, default pose and
  action scale must match the ``G1_*`` values (relative tolerance 1e-5), and its
  history, goal frames, control period and observation names this controller's,
  or loading fails. ``providers``: onnxruntime execution providers (default CPU).
"""

from __future__ import annotations

import numpy as np

from ...controllers.blinkmimic import (
    BLINKMIMIC_JOINT_NAMES,
    FUTURE_FRAMES,
    G1_ACTION_SCALE,
    G1_DEFAULT_POS,
    G1_EFFORT,
    G1_KD,
    G1_KP,
    HISTORY_LEN,
    BlinkMimicController,
    load_npz_runner,
    load_onnx_motion,
    load_pt_runner,
)
from ...registry import register_profile
from .._common import checkpoint

VARIANTS = {
    "default": {"format": "pt", "checkpoint": "~/.cache/hiwbc/blinkmimic/policy.pt"},
    "xmimic": {"format": "npz", "checkpoint": "~/.cache/hiwbc/xmimic/policy.npz"},
    "onnx": {"format": "onnx", "checkpoint": "~/.cache/hiwbc/blinkmimic/dance1_subject2.onnx"},
}
_OBS_NAMES = ("joint_pos", "joint_vel", "base_ang_vel", "projected_gravity", "actions",
              "motion_anchor_ori_b")


def check_onnx_metadata(meta: dict, source: str) -> None:
    """Raise ``ValueError`` if an export's metadata differs from this profile's G1 values."""
    if meta["joint_names"] != tuple(BLINKMIMIC_JOINT_NAMES):
        raise ValueError(f"{source}: joint_names {meta['joint_names']} are not BlinkMimic's "
                         "G1 order")
    for key, want in (("kp", G1_KP), ("kd", G1_KD), ("default_pos", G1_DEFAULT_POS),
                      ("action_scale", G1_ACTION_SCALE)):
        got = meta[key]
        if got.shape != want.shape or not np.allclose(got, want, rtol=1e-5, atol=1e-9):
            raise ValueError(f"{source}: metadata {key} {got.tolist()} differs from the G1 "
                             f"values {want.tolist()}")
    if (meta["history"], meta["goal_frames"]) != (HISTORY_LEN, FUTURE_FRAMES):
        raise ValueError(f"{source}: observation_history / goal_frames are "
                         f"{(meta['history'], meta['goal_frames'])}, need "
                         f"({HISTORY_LEN}, {FUTURE_FRAMES})")
    if abs(meta["control_dt"] - 0.02) > 1e-9:
        raise ValueError(f"{source}: control_dt is {meta['control_dt']}, need 0.02")
    if meta["observation_names"] != _OBS_NAMES:
        raise ValueError(f"{source}: observation_names {meta['observation_names']} differ "
                         f"from {_OBS_NAMES}")
    if meta["anchor_body_name"] not in (None, "pelvis"):
        raise ValueError(f"{source}: anchor body is {meta['anchor_body_name']!r}, not the pelvis")


def build(v: dict, options: dict) -> BlinkMimicController:
    path = checkpoint(options["checkpoint"] or v["checkpoint"])
    gains = {"default_pos": G1_DEFAULT_POS, "action_scale": G1_ACTION_SCALE, "kp": G1_KP,
             "kd": G1_KD, "effort": G1_EFFORT, "bound_torque": options["bound_torque"]}
    if v["format"] == "onnx":
        motion_fn, meta = load_onnx_motion(path, providers=options["providers"])
        check_onnx_metadata(meta, path.name)
        ctrl = BlinkMimicController(motion_fn=motion_fn, motion_name=meta["motion"],
                                    motion_frames=meta["frames"],
                                    anchor_quat_init=meta["anchor_quat_init"], **gains)
        ctrl.reset()
        return ctrl
    if v["format"] == "pt":
        fn = load_pt_runner(path, default_pos=G1_DEFAULT_POS, action_scale=G1_ACTION_SCALE)
    elif v["format"] == "npz":
        fn = load_npz_runner(path, default_pos=G1_DEFAULT_POS, action_scale=G1_ACTION_SCALE,
                             precision=options["precision"])
    else:
        raise ValueError(f"unknown checkpoint format {v['format']!r}")
    ctrl = BlinkMimicController(forward_fn=fn, **gains)
    ctrl.reset()
    return ctrl


register_profile(
    "blinkmimic/unitree_g1",
    build=build,
    variants=VARIANTS,
    default_options={"checkpoint": None, "precision": "bf16", "bound_torque": True,
                     "providers": None},
    description="BlinkMimic / XMimic tracker on the 29-DoF G1 (train your own checkpoint)",
)

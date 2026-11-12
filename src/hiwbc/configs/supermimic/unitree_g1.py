"""``supermimic`` on the 29-DoF Unitree G1, from a supermimic_deploy ``assets/`` bundle.

Unofficial. Joint order, default pose, action scale, gains, effort limits and the
control period come from the bundle's ``config.json``; ``policy.onnx`` is checked
against its sha256. See docs/guides/motion_tracking.md.
"""

from __future__ import annotations

from ...controllers.supermimic import SuperMimicController
from ...registry import register_profile


def build(_variant, options: dict) -> SuperMimicController:
    ctrl = SuperMimicController.from_bundle(
        options["checkpoint_dir"], check_hash=options["check_hash"], providers=options["providers"],
        bound_torque=options["bound_torque"])
    ctrl.reset()
    return ctrl


register_profile(
    "supermimic/unitree_g1",
    build=build,
    variants={"default": None},
    default_options={"checkpoint_dir": "~/.cache/hiwbc/supermimic/assets", "check_hash": True,
                     "providers": None, "bound_torque": True},
    description="SuperMimic tracker on the 29-DoF G1 (unofficial bundle)",
)

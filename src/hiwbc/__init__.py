"""hiWBC: a unified whole-body control layer for humanoid robots.

A controller maps ``(ControllerState, Reference) -> JointTarget`` — reference in,
balanced joint targets out — behind one contract, so the layers above never
touch balance, locomotion, or the vendor control loop.
"""

# Importing the controllers package registers the built-in controllers.
from . import controllers  # noqa: F401
from .controller import Controller
from .registry import (
    Profile,
    available_controllers,
    available_profiles,
    get_profile,
    load,
    make_controller,
    register_controller,
    register_profile,
)
from .runtime import load_onnx_runner, load_torchscript_runner
from .types import (
    BehaviorSpec,
    ControllerState,
    JointTarget,
    Reference,
    ReferenceKind,
    ReferenceSource,
)

__version__ = "0.0.3"   # the only place the version is written (pyproject reads it)

__all__ = [
    "BehaviorSpec",
    "Controller",
    "ControllerState",
    "JointTarget",
    "Profile",
    "Reference",
    "ReferenceKind",
    "ReferenceSource",
    "__version__",
    "available_controllers",
    "available_profiles",
    "get_profile",
    "load",
    "load_onnx_runner",
    "load_torchscript_runner",
    "make_controller",
    "register_controller",
    "register_profile",
]

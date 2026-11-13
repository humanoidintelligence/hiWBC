"""Core contract types for whole-body controllers.

A controller is a pure step: ``(ControllerState, Reference) -> JointTarget``. It
owns no hardware. The layer above (hiRobot) reads the robot, calls
:meth:`~hiwbc.controller.Controller.step`, and sends the returned joint targets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

import numpy as np


class ReferenceKind(str, Enum):
    """What a controller is being commanded with.

    The same controller contract carries every command source; a controller
    declares which kinds it accepts via
    :meth:`~hiwbc.controller.Controller.accepts`.
    """

    VELOCITY = "velocity"          # [vx, vy, wyaw] (+ optional gait / height), joystick-style
    MOTION_FRAMES = "motion_frames"  # future reference-motion frames / phase encoding (trackers)
    LATENT = "latent"              # an opaque motion latent / token (e.g. SONIC, VLA output)


@dataclass
class ControllerState:
    """Robot proprioception a controller reads each tick.

    Joint vectors are in the controller's own joint order; reconciling the
    robot's layout to the controller's is hiRobot's job (by joint name), not
    hiWBC's. All fields are plain ``float32`` arrays.
    """

    dof_pos: np.ndarray                      # (num_dofs,)
    dof_vel: np.ndarray                      # (num_dofs,)
    base_quat: np.ndarray                    # (4,) [x, y, z, w]
    base_ang_vel: np.ndarray                 # (3,) body frame
    base_lin_vel: np.ndarray | None = None   # (3,) optional, body frame
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Reference:
    """A high-level command handed to the controller.

    ``kind`` selects how ``value`` is interpreted. ``value`` is a float array for
    VELOCITY / LATENT; ``frames`` carries structured reference-motion data for
    MOTION_FRAMES. Unused fields stay ``None``.
    """

    kind: ReferenceKind
    value: np.ndarray | None = None          # VELOCITY command or LATENT token
    frames: dict[str, np.ndarray] | None = None  # MOTION_FRAMES payload
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class JointTarget:
    """What a controller emits each tick, consumed by hiRobot's send path.

    ``position`` is the full-body PD target in the controller's joint order.
    ``kp`` / ``kd`` are optional per-joint gain overrides; when ``None`` the
    robot keeps its configured gains.
    """

    position: np.ndarray                     # (num_actions,)
    kp: np.ndarray | None = None
    kd: np.ndarray | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BehaviorSpec:
    """A behavior a controller ships with (e.g. a motion baked into its checkpoint).

    ``WbcRobot`` exposes each one as a hiRobot layer skill. ``parameters`` is a JSON
    Schema for the keyword arguments of
    :meth:`~hiwbc.controller.Controller.start_behavior`.
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}, "additionalProperties": False}
    )
    #: Joint pose (controller order) the robot must be in when the behavior starts,
    #: e.g. a baked-in motion's first frame; None = start from wherever it is.
    start_pos: tuple[float, ...] | None = None


@runtime_checkable
class ReferenceSource(Protocol):
    """Yields one :class:`Reference` per tick until ``done`` (a running behavior)."""

    @property
    def done(self) -> bool: ...

    def next(self, state: ControllerState) -> Reference: ...

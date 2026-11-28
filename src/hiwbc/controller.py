"""The whole-body controller contract.

A :class:`Controller` turns ``(ControllerState, Reference)`` into a
:class:`~hiwbc.types.JointTarget`. Concrete controllers differ only in how they
build the policy observation and what reference they consume; the forward pass
and action post-processing are shared here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np

from .types import (
    BehaviorSpec,
    ControllerState,
    JointTarget,
    Reference,
    ReferenceKind,
    ReferenceSource,
)


class Controller(ABC):
    """Base class for whole-body controllers.

    Subclasses implement :meth:`observe` (build the policy input from state +
    reference) and :meth:`forward` (run the model). :meth:`step` wires them with
    the shared post-processing: EMA smoothing, clip, scale, and the default-pose
    offset that turns a residual action into an absolute PD target.
    """

    #: Reference kinds this controller accepts.
    accepts: tuple[ReferenceKind, ...] = ()
    #: ``MotionClip`` spaces (hiRobot ``MotionClip.space``) this controller can track; empty = none.
    motion_spaces: tuple[str, ...] = ()
    #: Keyword arguments for the ``ClipPlayer`` that feeds this controller
    #: (``ClipPlayer(clip, ctrl.control_dt, **ctrl.clip_player_options)``).
    clip_player_options: ClassVar[dict[str, Any]] = {}

    def __init__(
        self,
        *,
        num_actions: int,
        default_pos: Sequence[float],
        action_scale: float | Sequence[float] = 1.0,
        action_clip: float | None = None,
        action_beta: float = 1.0,
        joint_names: Sequence[str] | None = None,
        obs_joint_names: Sequence[str] | None = None,
        kp: Sequence[float] | None = None,
        kd: Sequence[float] | None = None,
        control_dt: float = 0.02,
        cartesian_targets: dict[str, str] | None = None,
        stand_command: Sequence[float] | None = None,
        velocity_limits: Sequence[Sequence[float]] | None = None,
    ) -> None:
        self.num_actions = num_actions
        self.default_pos = np.asarray(default_pos, dtype=np.float32)
        if self.default_pos.shape != (num_actions,):
            raise ValueError(
                f"default_pos has shape {self.default_pos.shape}, expected ({num_actions},)"
            )
        # Joint names (hiRobot spelling) in the controller's order; None when built
        # bare in a test. A profile always sets them; WbcRobot aligns by name.
        if joint_names is not None:
            joint_names = tuple(joint_names)
            if len(joint_names) != num_actions:
                raise ValueError(f"joint_names has {len(joint_names)} names, expected {num_actions}")
            if len(set(joint_names)) != num_actions:
                raise ValueError("joint_names repeats a joint")
        self.joint_names = joint_names
        # Joints the observation reads (ControllerState.dof_pos order). Usually the
        # driven joints; AMO also reads the arms it does not drive.
        if obs_joint_names is not None:
            obs_joint_names = tuple(obs_joint_names)
            if len(set(obs_joint_names)) != len(obs_joint_names):
                raise ValueError("obs_joint_names repeats a joint")
            if joint_names is None:
                raise ValueError("obs_joint_names needs joint_names")
        self.obs_joint_names = obs_joint_names if obs_joint_names is not None else joint_names
        # PD gains sent with each target; None leaves the robot's configured gains.
        self.kp = self._gains("kp", kp)
        self.kd = self._gains("kd", kd)
        if control_dt <= 0:
            raise ValueError("control_dt must be positive")
        self.control_dt = float(control_dt)
        #: End-effector name -> controller target name; empty = no CARTESIAN.
        self.cartesian_targets = dict(cartesian_targets or {})
        # VELOCITY command that means "stand still" for this checkpoint (not always zero).
        self.stand_command = (
            None if stand_command is None else np.asarray(stand_command, dtype=np.float32).reshape(-1)
        )
        # Per-axis (min, max) of a [vx, vy, wz] VELOCITY command tested to walk;
        # None = not given (or the command is not a zero-means-stand [vx, vy, wz]).
        self.velocity_limits = None
        if velocity_limits is not None:
            lim = np.asarray(velocity_limits, dtype=float)
            if lim.shape != (3, 2) or not np.all(lim[:, 0] <= 0) or not np.all(lim[:, 1] >= 0):
                raise ValueError("velocity_limits must be three (min, max) pairs with min <= 0 <= max")
            self.velocity_limits = tuple((float(a), float(b)) for a, b in lim)
        # A scalar, or one scale per joint (e.g. SONIC's effort-limit / stiffness scale).
        action_scale = np.asarray(action_scale, dtype=np.float32)
        if action_scale.shape not in ((), (num_actions,)):
            raise ValueError(
                f"action_scale has shape {action_scale.shape}, expected () or ({num_actions},)"
            )
        if np.any(action_scale <= 0):
            raise ValueError("action_scale must be positive")
        if action_clip is not None and action_clip <= 0:
            raise ValueError("action_clip must be positive")
        if not 0.0 < action_beta <= 1.0:
            raise ValueError("action_beta must be in (0, 1]")

        self.action_scale = action_scale
        self.action_clip = action_clip
        self.action_beta = action_beta
        self._last_action = np.zeros(num_actions, dtype=np.float32)

    def _gains(self, name: str, value) -> np.ndarray | None:
        if value is None:
            return None
        v = np.asarray(value, dtype=np.float32)
        if v.shape != (self.num_actions,):
            raise ValueError(f"{name} has shape {v.shape}, expected ({self.num_actions},)")
        if np.any(v < 0):
            raise ValueError(f"{name} must be non-negative")
        return v

    # ----- lifecycle --------------------------------------------------------
    def reset(self) -> None:
        """Clear per-episode state (smoothing buffer, timestep, history, recurrent state)."""
        self._last_action = np.zeros(self.num_actions, dtype=np.float32)
        for fn in self._model_fns():
            reset = getattr(fn, "reset", None)
            if callable(reset):
                reset()

    def _model_fns(self) -> list:
        """Injected model callables, so reset() can clear a recurrent model's state."""
        return [v for k, v in vars(self).items() if k.endswith("_fn") and callable(v)]

    # ----- reference helpers -------------------------------------------------
    def stand_reference(self, state: ControllerState) -> Reference:
        """The reference that holds the robot standing still from ``state``.

        ``WbcRobot`` uses it for ``hold()``, after a clip or behavior ends, and on a
        command timeout. A velocity controller returns its ``stand_command``;
        controllers with no standing command (the trackers) raise.
        """
        if self.stand_command is not None and ReferenceKind.VELOCITY in self.accepts:
            return Reference(ReferenceKind.VELOCITY, value=self.stand_command.copy())
        raise NotImplementedError(f"{type(self).__name__} has no standing reference")

    def cartesian_rest(self) -> dict[str, np.ndarray]:
        """Pose of each :attr:`cartesian_targets` target when none is commanded.

        Target name -> ``[x, y, z, qx, qy, qz, qw]`` in the base frame. ``WbcRobot``
        rate-limits ``CARTESIAN`` targets starting from these and returns to them
        when the command ends. Empty (the default) = start from the first target.
        """
        return {}

    def check_clip(self, player) -> None:
        """Raise ``ValueError`` if this controller cannot track ``player``'s clip.

        ``WbcRobot`` calls it when a clip arrives, so a clip that is missing data
        the controller needs is rejected up front instead of failing mid-motion.
        """
        if player.space not in self.motion_spaces:
            raise ValueError(
                f"{type(self).__name__} does not track {player.space!r} clips "
                f"(motion_spaces={self.motion_spaces})")

    def clip_reference(self, player, state: ControllerState) -> Reference:
        """This tick's reference from a :class:`~hiwbc.references.ClipPlayer`.

        Reads the player's current frame (and the future frames the model looks
        at); the caller advances the player after :meth:`step`.
        """
        raise NotImplementedError(f"{type(self).__name__} does not track motion clips")

    @property
    def behaviors(self) -> list[BehaviorSpec]:
        """Behaviors this controller ships with; ``WbcRobot`` lists them as layer skills."""
        return []

    def start_behavior(self, name: str, **args) -> ReferenceSource:
        """Start a behavior from :attr:`behaviors`; returns its per-tick reference source."""
        known = [b.name for b in self.behaviors]
        raise KeyError(f"unknown behavior {name!r}; available: {known}")

    # ----- contract to implement -------------------------------------------
    @abstractmethod
    def observe(self, state: ControllerState, reference: Reference):
        """Assemble the policy observation for this tick.

        Returns a flat ``np.ndarray`` for single-input models, or a tuple of
        arrays for models that take several named inputs (e.g. an actor obs plus
        a future-motion target and a history buffer).
        """

    @abstractmethod
    def forward(self, obs) -> np.ndarray:
        """Run the model on ``obs`` and return the raw action vector."""

    # ----- shared step ------------------------------------------------------
    def step(self, state: ControllerState, reference: Reference) -> JointTarget:
        """Produce a joint target for one control tick."""
        if self.accepts and reference.kind not in self.accepts:
            raise ValueError(
                f"{type(self).__name__} accepts {[k.value for k in self.accepts]}, "
                f"got {reference.kind.value}"
            )
        obs = self.observe(state, reference)
        raw = np.asarray(self.forward(obs), dtype=np.float32).reshape(-1)
        if raw.shape != (self.num_actions,):
            raise ValueError(
                f"forward returned shape {raw.shape}, expected ({self.num_actions},)"
            )
        action = self._postprocess(raw)
        return JointTarget(position=action + self.default_pos, kp=self.kp, kd=self.kd)

    def _postprocess(self, raw: np.ndarray) -> np.ndarray:
        """EMA-smooth, clip, then scale a raw action (RoboJuDo order)."""
        action = (1.0 - self.action_beta) * self._last_action + self.action_beta * raw
        self._last_action = action.copy()
        if self.action_clip is not None:
            action = np.clip(action, -self.action_clip, self.action_clip)
        return action * self.action_scale

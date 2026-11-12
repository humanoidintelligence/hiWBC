"""Velocity-command locomotion controller (Unitree family).

Replicates the observation layout of RoboJuDo's ``UnitreePolicy``: a flat vector
of base angular velocity, projected gravity, the [vx, vy, wyaw] command, joint
position residual, joint velocity, last action, and a gait-phase sin/cos. A
gaitless variant drops the phase terms. The model itself is a user-supplied
checkpoint loaded by a backend.

Observation field order is safety-critical: it must match the order the policy
was trained with, or the robot will be commanded with garbage. The order here
follows ``unitree_policy.py`` exactly.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from ..controller import Controller
from ..math_utils import gravity_orientation
from ..registry import register_controller
from ..types import ControllerState, Reference, ReferenceKind


@dataclass
class ObsScales:
    """Per-term observation scaling (RoboJuDo ``ObsScalesCfg`` defaults)."""

    dof_pos: float = 1.0
    dof_vel: float = 0.05
    ang_vel: float = 0.25
    command: tuple[float, float, float] = (2.0, 2.0, 0.25)


class UnitreeLocoController(Controller):
    """Residual-action velocity locomotion over a periodic gait clock.

    The forward function is injected (``forward_fn``) so the checkpoint backend
    is decoupled from the observation logic; a backend module supplies a
    torch-jit or ONNX runner, and tests supply a stub.
    """

    accepts = (ReferenceKind.VELOCITY,)

    def __init__(
        self,
        *,
        num_actions: int,
        default_pos: Sequence[float],
        forward_fn: Callable[[np.ndarray], np.ndarray],
        control_hz: int = 50,
        cycle_time: float = 0.8,
        with_gait: bool = True,
        obs_scales: ObsScales | None = None,
        max_cmd: Sequence[float] = (0.8, 0.5, 1.57),
        action_scale: float = 0.25,
        action_clip: float | None = None,
        action_beta: float = 0.8,
        history_length: int = 0,
        gravity_scale: float = 1.0,
        **base,
    ) -> None:
        super().__init__(
            num_actions=num_actions,
            default_pos=default_pos,
            action_scale=action_scale,
            action_clip=action_clip,
            action_beta=action_beta,
            control_dt=1.0 / control_hz,
            **{"stand_command": (0.0, 0.0, 0.0), **base},
        )
        if history_length and with_gait:
            raise ValueError("history_length is only supported with with_gait=False")
        self._forward_fn = forward_fn
        self.dt = 1.0 / control_hz
        self.cycle_time = cycle_time
        self.with_gait = with_gait
        self.obs_scales = obs_scales or ObsScales()
        self.max_cmd = np.asarray(max_cmd, dtype=np.float32)
        self.history_length = history_length
        self.gravity_scale = gravity_scale
        self._timestep = 0
        self._history: deque[list[np.ndarray]] = deque(maxlen=max(history_length, 1))
        self._reset_history()

    def _reset_history(self) -> None:
        if not self.history_length:
            return
        n = self.num_actions
        zeros = [np.zeros(d, dtype=np.float32) for d in (3, 3, 3, n, n, n)]
        self._history = deque([zeros] * self.history_length, maxlen=self.history_length)

    def reset(self) -> None:
        super().reset()
        self._timestep = 0
        self._reset_history()

    def _phase(self) -> float:
        return (self._timestep * self.dt) / self.cycle_time

    def observe(self, state: ControllerState, reference: Reference) -> np.ndarray:
        if reference.value is None:
            raise ValueError("VELOCITY reference requires a [vx, vy, wyaw] command in value")
        command = np.asarray(reference.value, dtype=np.float32).reshape(-1)
        if command.shape != (3,):
            raise ValueError(f"velocity command has shape {command.shape}, expected (3,)")

        scale = np.asarray(self.obs_scales.command, dtype=np.float32)
        terms = [
            state.base_ang_vel * self.obs_scales.ang_vel,
            gravity_orientation(state.base_quat) * self.gravity_scale,
            command * scale * self.max_cmd,
            (state.dof_pos - self.default_pos) * self.obs_scales.dof_pos,
            state.dof_vel * self.obs_scales.dof_vel,
            self._last_action,
        ]
        if self.history_length:
            # UnitreeWoGaitPolicy: push this tick, then lay out term-major
            # (all ang_vel frames, then all gravity frames, ...), oldest first.
            self._history.append([np.asarray(t, dtype=np.float32) for t in terms])
            return np.concatenate(
                [np.concatenate(items) for items in zip(*self._history, strict=True)]
            ).astype(np.float32)
        if self.with_gait:
            phase = self._phase()
            terms.append(np.array([np.sin(2 * np.pi * phase)], dtype=np.float32))
            terms.append(np.array([np.cos(2 * np.pi * phase)], dtype=np.float32))
        return np.concatenate(terms).astype(np.float32)

    def forward(self, obs: np.ndarray) -> np.ndarray:
        return self._forward_fn(obs)

    def step(self, state, reference):
        out = super().step(state, reference)
        self._timestep += 1
        return out


@register_controller("unitree_loco")
def _build_unitree(**kwargs) -> UnitreeLocoController:
    return UnitreeLocoController(**kwargs)

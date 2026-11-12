"""Velocity-command locomotion with roll/pitch and history (Smooth family).

Replicates the observation layout of RoboJuDo's ``SmoothPolicy``: like the
Unitree family but the proprio block carries base roll/pitch, and the full
observation appends a zeroed privileged-latent slot plus a flattened history of
the proprio block.

Observation order (``smooth_policy.py``):

    [proprio] = sin, cos, commands, base_ang_vel, roll_pitch,
                dof_pos - default, dof_vel, last_action
    obs = proprio, priv_latent(zeros), history(proprio x history_length)

Field order is safety-critical and matches the reference exactly.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from ..controller import Controller
from ..math_utils import quat_to_euler
from ..registry import register_controller
from ..types import ControllerState, Reference, ReferenceKind


@dataclass
class SmoothObsScales:
    """Per-term observation scaling (RoboJuDo ``SmoothPolicyCfg``)."""

    ang_vel: float = 0.25
    dof_vel: float = 0.05


class SmoothLocoController(Controller):
    """Velocity locomotion with a privileged-latent slot and proprio history."""

    accepts = (ReferenceKind.VELOCITY,)

    def __init__(
        self,
        *,
        num_actions: int,
        default_pos: Sequence[float],
        forward_fn: Callable[[np.ndarray], np.ndarray],
        control_hz: int = 50,
        cycle_time: float = 0.8,
        n_priv_latent: int = 0,
        history_length: int = 0,
        obs_scales: SmoothObsScales | None = None,
        action_scale: float = 0.25,
        action_clip: float | None = None,
        action_beta: float = 0.8,
        **base,
    ) -> None:
        super().__init__(
            num_actions=num_actions,
            default_pos=default_pos,
            action_scale=action_scale,
            action_clip=action_clip,
            action_beta=action_beta,
            control_dt=1.0 / control_hz,
            **{"stand_command": (-0.5, 0.0, -0.5), **base},
        )
        self._forward_fn = forward_fn
        self.dt = 1.0 / control_hz
        self.cycle_time = cycle_time
        self.n_priv_latent = n_priv_latent
        self.history_length = history_length
        self.obs_scales = obs_scales or SmoothObsScales()
        self._timestep = 0
        self._history: deque[np.ndarray] = deque(maxlen=history_length)
        self._reset_history()

    def reset(self) -> None:
        super().reset()
        self._timestep = 0
        self._reset_history()

    def _reset_history(self) -> None:
        # RoboJuDo seeds the buffer with zeros at construction, not only on reset.
        if self.history_length:
            size = self._proprio_size()
            self._history = deque(
                [np.zeros(size, dtype=np.float32)] * self.history_length,
                maxlen=self.history_length,
            )

    def _phase(self) -> float:
        return (self._timestep * self.dt) / self.cycle_time

    def _proprio_size(self) -> int:
        # sin(1)+cos(1)+commands(3)+ang_vel(3)+roll_pitch(2)+dof_pos(n)+dof_vel(n)+last_action(n)
        return 1 + 1 + 3 + 3 + 2 + self.num_actions * 3

    def observe(self, state: ControllerState, reference: Reference) -> np.ndarray:
        if reference.value is None:
            raise ValueError("VELOCITY reference requires a [vx, vy, wyaw] command in value")
        command = np.asarray(reference.value, dtype=np.float32).reshape(-1)
        if command.shape != (3,):
            raise ValueError(f"velocity command has shape {command.shape}, expected (3,)")

        phase = self._phase()
        rpy = quat_to_euler(state.base_quat)
        proprio = np.concatenate(
            [
                [np.sin(2 * np.pi * phase)],
                [np.cos(2 * np.pi * phase)],
                command,
                state.base_ang_vel * self.obs_scales.ang_vel,
                rpy[:2],
                state.dof_pos - self.default_pos,
                state.dof_vel * self.obs_scales.dof_vel,
                self._last_action,
            ]
        ).astype(np.float32)

        priv_latent = np.zeros(self.n_priv_latent, dtype=np.float32)
        history = (
            np.concatenate(list(self._history)).astype(np.float32)
            if self.history_length
            else np.zeros(0, dtype=np.float32)
        )
        obs = np.concatenate([proprio, priv_latent, history]).astype(np.float32)
        if self.history_length:
            self._history.append(proprio)
        return obs

    def forward(self, obs: np.ndarray) -> np.ndarray:
        return self._forward_fn(obs)

    def step(self, state, reference):
        out = super().step(state, reference)
        self._timestep += 1
        return out


@register_controller("smooth_loco")
def _build_smooth(**kwargs) -> SmoothLocoController:
    return SmoothLocoController(**kwargs)

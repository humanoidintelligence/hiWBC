"""Reference-motion tracking controller (ASAP family).

Replicates the observation layout of RoboJuDo's ``AsapPolicy`` (from
[ASAP](https://github.com/LeCAR-Lab/ASAP)): the tracked motion is baked into the
checkpoint and advanced by an internal phase clock, so the observation is
proprioception plus a scalar frame-encoding (phase in ``[0, 1]`` over
``motion_length_s``), optionally with a sorted history buffer.

Observation order (``get_observation`` in ``asap_policy.py``):

    last_action, base_ang_vel, dof_pos - default, dof_vel,
    [history], projected_gravity, phase

The history buffer stores the per-tick block *without* the history term, in the
same field order minus history. Field order is safety-critical and matches the
reference exactly. ASAP does no action EMA: ``last_action`` is the raw model
output, then clip, then scale — the base contract reproduces this at
``action_beta=1.0``.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from ..controller import Controller
from ..math_utils import quat_rotate_inverse
from ..registry import register_controller
from ..types import BehaviorSpec, ControllerState, Reference, ReferenceKind, ReferenceSource

_GRAVITY_VEC = np.array([0.0, 0.0, -1.0], dtype=np.float32)


@dataclass
class AsapObsScales:
    """Per-term observation scaling (RoboJuDo ``AsapPolicyCfg.ObsScalesCfg``)."""

    actions: float = 1.0
    base_ang_vel: float = 1.0
    dof_pos: float = 1.0
    dof_vel: float = 1.0
    projected_gravity: float = 1.0
    ref_motion_phase: float = 1.0
    history: float = 1.0


class AsapTrackerController(Controller):
    """Phase-clocked deepmimic tracker; proprio + frame-encoding observation."""

    accepts = (ReferenceKind.MOTION_FRAMES,)

    def __init__(
        self,
        *,
        num_actions: int,
        default_pos: Sequence[float],
        forward_fn: Callable[[np.ndarray], np.ndarray],
        motion_length_s: float,
        control_hz: int = 50,
        use_history: bool = False,
        history_length: int = 4,
        obs_scales: AsapObsScales | None = None,
        action_scale: float = 1.0,
        action_clip: float | None = None,
        motion_name: str | None = None,
        start_pos: Sequence[float] | None = None,
        **base,
    ) -> None:
        # ASAP applies no EMA smoothing: last_action is the raw action.
        super().__init__(
            num_actions=num_actions,
            default_pos=default_pos,
            action_scale=action_scale,
            action_clip=action_clip,
            action_beta=1.0,
            control_dt=1.0 / control_hz,
            **base,
        )
        self._forward_fn = forward_fn
        self.dt = 1.0 / control_hz
        self.motion_length_s = motion_length_s
        #: Name of the baked-in motion, exposed as a behavior; None = no behavior.
        self.motion_name = motion_name
        #: The motion's first pose (controller order); the robot must start there.
        self.start_pos = None if start_pos is None else np.asarray(start_pos, dtype=np.float32)
        if self.start_pos is not None and self.start_pos.shape != (num_actions,):
            raise ValueError(f"start_pos has shape {self.start_pos.shape}, expected ({num_actions},)")
        self.use_history = use_history
        self.history_length = history_length
        self.obs_scales = obs_scales or AsapObsScales()
        self._timestep = 0.0
        self._play_speed = 1.0
        self.motion_done = False
        self._history: deque[list[np.ndarray]] = deque(maxlen=max(history_length, 1))
        self._reset_history()

    def reset(self) -> None:
        super().reset()
        self._timestep = 0.0
        self._play_speed = 1.0
        self.motion_done = False
        self._reset_history()

    # ----- behavior: the motion baked into the checkpoint ---------------------
    @property
    def behaviors(self) -> list[BehaviorSpec]:
        if not self.motion_name:
            return []
        return [BehaviorSpec(
            name=self.motion_name,
            description=f"Play the motion trained into this checkpoint ({self.motion_length_s:g} s).",
            parameters={
                "type": "object",
                "properties": {"play_speed": {"type": "number", "exclusiveMinimum": 0,
                                              "maximum": 2, "default": 1.0}},
                "additionalProperties": False,
            },
            start_pos=None if self.start_pos is None else tuple(float(x) for x in self.start_pos),
        )]

    def start_behavior(self, name: str, **args) -> ReferenceSource:
        if name != self.motion_name or not name:
            return super().start_behavior(name, **args)
        unknown = set(args) - {"play_speed"}
        if unknown:
            raise TypeError(f"behavior {name!r} got unknown argument(s) {sorted(unknown)}")
        speed = float(args.get("play_speed", 1.0))
        if not 0 < speed <= 2:
            raise ValueError("play_speed must be in (0, 2]")
        # The phase clock starts from frame 0: a behavior always plays from the top.
        self.reset()
        return _BakedMotion(self, speed)

    def _reset_history(self) -> None:
        # Zero-filled at construction and on reset (RoboJuDo ``_init_history``); a
        # partially filled buffer would shift every term after the history block.
        if self.use_history:
            # entry = last_action, ang_vel, dof_pos, dof_vel, gravity, phase
            n = self.num_actions
            zeros = [np.zeros(d, dtype=np.float32) for d in (n, 3, n, n, 3, 1)]
            self._history = deque([zeros] * self.history_length, maxlen=self.history_length)

    def _phase(self) -> float:
        return (self._timestep * self.dt) / self.motion_length_s

    def _history_block_size(self) -> int:
        # last_action(n) + base_ang_vel(3) + dof_pos(n) + dof_vel(n) + gravity(3) + phase(1)
        return self.num_actions * 3 + 3 + 3 + 1

    def observe(self, state: ControllerState, reference: Reference) -> np.ndarray:
        ang_vel = state.base_ang_vel * self.obs_scales.base_ang_vel
        dof_pos = (state.dof_pos - self.default_pos) * self.obs_scales.dof_pos
        dof_vel = state.dof_vel * self.obs_scales.dof_vel
        projected_gravity = (
            quat_rotate_inverse(state.base_quat, _GRAVITY_VEC) * self.obs_scales.projected_gravity
        )
        phase = np.array([self._phase()], dtype=np.float32) * self.obs_scales.ref_motion_phase
        last_action = self._last_action * self.obs_scales.actions

        block = [np.asarray(t, dtype=np.float32)
                 for t in (last_action, ang_vel, dof_pos, dof_vel, projected_gravity, phase)]
        history_term = (
            self._get_history() * self.obs_scales.history
            if self.use_history
            else np.zeros(0, dtype=np.float32)
        )
        obs = np.concatenate(
            [last_action, ang_vel, dof_pos, dof_vel, history_term, projected_gravity, phase]
        ).astype(np.float32)

        if self.use_history:
            self._history.appendleft(block)
        return obs

    def _get_history(self) -> np.ndarray:
        # Term-major, newest first (RoboJuDo ``_get_obs_history``): all last_action
        # entries, then all ang_vel entries, ... each from newest to oldest.
        if not self._history:
            return np.zeros(self._history_block_size() * self.history_length, dtype=np.float32)
        return np.concatenate(
            [np.concatenate(items) for items in zip(*self._history, strict=True)]
        ).astype(np.float32)

    def forward(self, obs: np.ndarray) -> np.ndarray:
        return self._forward_fn(obs)

    def step(self, state, reference):
        out = super().step(state, reference)
        play_speed = reference.extra.get("play_speed", self._play_speed)
        if self._phase() >= 1.0:
            self.motion_done = True
        else:
            self._timestep += play_speed
        return out


class _BakedMotion:
    """Plays the checkpoint's own motion: an empty MOTION_FRAMES reference per tick."""

    def __init__(self, ctrl: AsapTrackerController, play_speed: float) -> None:
        self._ctrl = ctrl
        self._extra = {"play_speed": play_speed}

    @property
    def done(self) -> bool:
        return self._ctrl.motion_done

    def next(self, state: ControllerState) -> Reference:
        return Reference(ReferenceKind.MOTION_FRAMES, extra=dict(self._extra))


@register_controller("asap_tracker")
def _build_asap(**kwargs) -> AsapTrackerController:
    return AsapTrackerController(**kwargs)

"""AMO velocity + upper-body-command controller (RoboJuDo ``AMOPolicy``).

From AMO (Adaptive Motion Optimization). The policy reads all body joints
(lower body + arms) but only actuates the lower body; the arm joints feed the
observation and an adapter MLP. One tick:

* the adapter maps ``[height, torso_yaw, torso_pitch, torso_roll, arm_dof_pos]``
  (normalized) to a 15-dim latent that is part of the proprio frame;
* the proprio frame is ``ang_vel, roll/pitch, sin/cos(dyaw), dof_pos - default,
  dof_vel, last_action_full, gait, adapter_out``;
* the policy obs is ``proprio, demo, priv(zeros), history`` where ``history`` is
  the previous 10 proprio frames (oldest first, current excluded);
* the model takes a second input: the last 25 proprio frames (oldest first,
  current included).

The command is ``[c0, c1, c2, height, torso_yaw, torso_pitch, torso_roll]`` in
RoboJuDo's AMO order: ``c1`` drives the yaw term (``dyaw = -c1``); ``c0`` and
``c2`` go into the demo obs. Pass 3 values to keep the last four at
``default_command``.

Matches RoboJuDo behaviour, including two quirks: the ankle/waist velocity
zeroing that RoboJuDo computes is not applied to the obs (it computes
``obs_dof_vel`` but uses ``dof_vel``), and the demo height is
``0.75 + height``.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence

import numpy as np

from ..controller import Controller
from ..math_utils import quat_to_euler
from ..registry import register_controller
from ..types import ControllerState, Reference, ReferenceKind

_DEFAULT_COMMAND = (0.0, 0.0, 0.0, 0.75, 0.0, 0.0, 0.0)


class AmoLocoController(Controller):
    """AMO: lower-body velocity control with commanded height / torso orientation."""

    accepts = (ReferenceKind.VELOCITY,)

    def __init__(
        self,
        *,
        num_actions: int,
        default_pos: Sequence[float],
        obs_default_pos: Sequence[float],
        forward_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        adapter_fn: Callable[[np.ndarray], np.ndarray],
        adapter_input_mean: Sequence[float],
        adapter_input_std: Sequence[float],
        adapter_output_mean: Sequence[float],
        adapter_output_std: Sequence[float],
        control_hz: int = 50,
        ang_vel_scale: float = 0.25,
        dof_vel_scale: float = 0.05,
        action_scale: float = 0.25,
        raw_action_clip: float = 40.0,
        default_command: Sequence[float] = _DEFAULT_COMMAND,
        gait_freq: float = 1.3,
        n_priv: int = 3,
        history_length: int = 10,
        extra_history_length: int = 25,
        **base,
    ) -> None:
        super().__init__(
            num_actions=num_actions,
            default_pos=default_pos,
            action_scale=action_scale,
            action_clip=None,
            action_beta=1.0,
            control_dt=1.0 / control_hz,
            **{"stand_command": tuple(default_command), **base},
        )
        self.obs_default_pos = np.asarray(obs_default_pos, dtype=np.float32)
        self.num_dofs = self.obs_default_pos.size
        self.n_upper = self.num_dofs - num_actions
        if self.n_upper < 0:
            raise ValueError("obs_default_pos must cover at least the actuated joints")
        if not np.allclose(self.obs_default_pos[:num_actions], self.default_pos):
            raise ValueError(
                "actuated joints must be the leading obs joints: "
                "obs_default_pos[:num_actions] != default_pos"
            )
        self._forward_fn = forward_fn
        self._adapter_fn = adapter_fn
        self._in_mean = np.asarray(adapter_input_mean, dtype=np.float32)
        self._in_std = np.asarray(adapter_input_std, dtype=np.float32)
        self._out_mean = np.asarray(adapter_output_mean, dtype=np.float32)
        self._out_std = np.asarray(adapter_output_std, dtype=np.float32)
        if self._in_mean.shape != (4 + self.n_upper,):
            raise ValueError(
                f"adapter_input_mean has shape {self._in_mean.shape}, expected ({4 + self.n_upper},)"
            )
        self.n_adapter_out = self._out_mean.size
        self.dt = 1.0 / control_hz
        self.ang_vel_scale = ang_vel_scale
        self.dof_vel_scale = dof_vel_scale
        self.raw_action_clip = raw_action_clip
        self.default_command = np.asarray(default_command, dtype=np.float32)
        if self.default_command.shape != (7,):
            raise ValueError("default_command must have 7 entries")
        self.gait_freq = gait_freq
        self.n_priv = n_priv
        self.history_length = history_length
        self.extra_history_length = extra_history_length
        self._reset_buffers()

    # ----- state ------------------------------------------------------------
    @property
    def proprio_size(self) -> int:
        # ang_vel 3 + roll/pitch 2 + sin/cos dyaw 2 + 3 joint terms + gait 2 + adapter out
        return 3 + 2 + 2 + 3 * self.num_dofs + 2 + self.n_adapter_out

    def _reset_buffers(self) -> None:
        z = np.zeros(self.proprio_size, dtype=np.float32)
        self._history = deque([z] * self.history_length, maxlen=self.history_length)
        self._extra_history = deque(
            [z] * self.extra_history_length, maxlen=self.extra_history_length
        )
        self.gait_cycle = np.array([0.25, 0.25])

    def reset(self) -> None:
        super().reset()
        self._reset_buffers()

    def _command(self, reference: Reference) -> np.ndarray:
        if reference.value is None:
            raise ValueError("VELOCITY reference requires a command in value")
        value = np.asarray(reference.value, dtype=np.float32).reshape(-1)
        if value.shape == (3,):
            cmd = self.default_command.copy()
            cmd[:3] = value
            return cmd
        if value.shape == (7,):
            return value
        raise ValueError(f"AMO command has shape {value.shape}, expected (3,) or (7,)")

    # ----- contract -----------------------------------------------------------
    def observe(self, state: ControllerState, reference: Reference):
        cmd = self._command(reference)
        dof_pos = np.asarray(state.dof_pos, dtype=np.float32)
        if dof_pos.shape != (self.num_dofs,):
            raise ValueError(f"dof_pos has shape {dof_pos.shape}, expected ({self.num_dofs},)")
        dof_vel = np.asarray(state.dof_vel, dtype=np.float32)
        upper = dof_pos[self.num_actions:]

        in_place_stand = bool(np.all(np.abs(cmd[0:3]) < 0.1))
        dyaw = np.remainder(-cmd[1] + np.pi, 2 * np.pi) - np.pi
        gait_obs = np.sin(self.gait_cycle * 2 * np.pi)

        adapter_in = np.concatenate([cmd[3:7], upper]).astype(np.float32)
        adapter_in = (adapter_in - self._in_mean) / (self._in_std + 1e-8)
        adapter_out = np.asarray(self._adapter_fn(adapter_in), dtype=np.float32).reshape(-1)
        adapter_out = adapter_out * self._out_std + self._out_mean

        offset = dof_pos - self.obs_default_pos
        last_action_full = np.concatenate(
            [self._last_action, offset[self.num_actions:] / self.action_scale]
        )
        rpy = quat_to_euler(state.base_quat)
        proprio = np.concatenate(
            [
                state.base_ang_vel * self.ang_vel_scale,
                rpy[:2],
                [np.sin(dyaw), np.cos(dyaw)],
                offset,
                dof_vel * self.dof_vel_scale,
                last_action_full,
                gait_obs,
                adapter_out,
            ]
        ).astype(np.float32)

        demo = np.zeros(self.n_upper + 9, dtype=np.float32)
        demo[: self.n_upper] = upper
        demo[self.n_upper] = cmd[0]
        demo[self.n_upper + 1] = cmd[2]
        demo[self.n_upper + 3: self.n_upper + 6] = cmd[4:7]
        demo[self.n_upper + 6: self.n_upper + 9] = 0.75 + cmd[3]

        obs = np.concatenate(
            [proprio, demo, np.zeros(self.n_priv, dtype=np.float32), *self._history]
        ).astype(np.float32)

        self._history.append(proprio)
        self._extra_history.append(proprio)
        extra = np.concatenate(list(self._extra_history)).astype(np.float32)

        self._advance_gait(in_place_stand)
        return obs, extra

    def _advance_gait(self, in_place_stand: bool) -> None:
        gc = np.remainder(self.gait_cycle + self.dt * self.gait_freq, 1.0)
        near0 = abs(gc[0] - 0.25) < 0.05
        near1 = abs(gc[1] - 0.25) < 0.05
        if in_place_stand and (near0 or near1):
            gc = np.array([0.25, 0.25])
        if (not in_place_stand) and near0 and near1:
            gc = np.array([0.25, 0.75])
        self.gait_cycle = gc

    def forward(self, obs) -> np.ndarray:
        prop_obs, extra = obs
        raw = np.asarray(self._forward_fn(prop_obs, extra), dtype=np.float32).reshape(-1)
        return np.clip(raw, -self.raw_action_clip, self.raw_action_clip)


def load_amo_runners(
    policy_file: str, adapter_file: str, norm_file: str, *, device: str = "cpu"
) -> dict:
    """Load AMO's TorchScript policy + adapter and norm stats (needs ``torch``).

    Returns the ``forward_fn`` / ``adapter_fn`` / ``adapter_*`` keyword arguments
    for :class:`AmoLocoController`. The released policy has CUDA device constants
    baked in; on CPU they are rewritten, as RoboJuDo does.
    """
    import torch

    model = torch.jit.load(policy_file, map_location=device)
    if device == "cpu":
        for node in model.graph.nodes():
            if (
                node.kind() == "prim::Constant"
                and "value" in node.attributeNames()
                and isinstance(node.output().toIValue(), torch.device)
            ):
                node.removeAttribute("value")
                node.s_("value", "cpu")
    adapter = torch.jit.load(adapter_file, map_location=device)
    adapter.eval()
    stats = torch.load(norm_file, weights_only=False)

    def forward_fn(obs, extra):
        with torch.no_grad():
            o = torch.from_numpy(np.asarray(obs, np.float32)).unsqueeze(0).to(device)
            e = torch.from_numpy(np.asarray(extra, np.float32)).view(1, -1).to(device)
            return model(o, e).cpu().numpy().squeeze()

    def adapter_fn(x):
        with torch.no_grad():
            t = torch.from_numpy(np.asarray(x, np.float32)).view(1, -1).to(device)
            return adapter(t).cpu().numpy().squeeze()

    return {
        "forward_fn": forward_fn,
        "adapter_fn": adapter_fn,
        "adapter_input_mean": np.asarray(stats["input_mean"], np.float32),
        "adapter_input_std": np.asarray(stats["input_std"], np.float32),
        "adapter_output_mean": np.asarray(stats["output_mean"], np.float32),
        "adapter_output_std": np.asarray(stats["output_std"], np.float32),
    }


@register_controller("amo_loco")
def _build_amo(**kwargs) -> AmoLocoController:
    return AmoLocoController(**kwargs)

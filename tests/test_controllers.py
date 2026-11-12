"""Contract and observation-layout tests. No checkpoints, torch, or hardware."""

import numpy as np
import pytest

from hiwbc import (
    ControllerState,
    Reference,
    ReferenceKind,
    available_controllers,
    make_controller,
)
from hiwbc.controllers.unitree import ObsScales, UnitreeLocoController
from hiwbc.math_utils import gravity_orientation


def _state(num_dofs: int) -> ControllerState:
    return ControllerState(
        dof_pos=np.zeros(num_dofs, dtype=np.float32),
        dof_vel=np.zeros(num_dofs, dtype=np.float32),
        base_quat=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        base_ang_vel=np.zeros(3, dtype=np.float32),
    )


# ----- registry ------------------------------------------------------------
def test_builtin_controllers_registered():
    names = available_controllers()
    assert "echo" in names
    assert "unitree_loco" in names


# ----- echo controller -----------------------------------------------------
def test_echo_returns_default_pose():
    n = 29
    default = np.linspace(-0.3, 0.3, n).tolist()
    ctrl = make_controller("echo", num_actions=n, default_pos=default)
    out = ctrl.step(_state(n), Reference(kind=ReferenceKind.LATENT, value=np.zeros(8)))
    assert out.position.shape == (n,)
    np.testing.assert_allclose(out.position, default, rtol=1e-6)


def test_echo_adds_residual():
    n = 4
    ctrl = make_controller(
        "echo", num_actions=n, default_pos=[0.0] * n, residual=[1.0, 2.0, 3.0, 4.0]
    )
    out = ctrl.step(_state(n), Reference(kind=ReferenceKind.VELOCITY, value=np.zeros(3)))
    np.testing.assert_allclose(out.position, [1.0, 2.0, 3.0, 4.0])


# ----- unitree loco: observation layout ------------------------------------
def test_unitree_obs_field_order_and_size():
    """Obs = ang_vel(3) + gravity(3) + command(3) + dof_pos(n) + dof_vel(n) + last_action(n) + sincos(2)."""
    n = 12
    captured = {}

    def fake_forward(obs):
        captured["obs"] = obs
        return np.zeros(n, dtype=np.float32)

    scales = ObsScales()
    ctrl = UnitreeLocoController(
        num_actions=n,
        default_pos=[0.1] * n,
        forward_fn=fake_forward,
        with_gait=True,
        obs_scales=scales,
        max_cmd=(0.8, 0.5, 1.57),
    )

    state = ControllerState(
        dof_pos=np.full(n, 0.2, dtype=np.float32),
        dof_vel=np.full(n, 0.5, dtype=np.float32),
        base_quat=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        base_ang_vel=np.array([0.1, 0.2, 0.3], dtype=np.float32),
    )
    cmd = np.array([1.0, 0.0, -1.0], dtype=np.float32)
    ctrl.step(state, Reference(kind=ReferenceKind.VELOCITY, value=cmd))

    obs = captured["obs"]
    assert obs.shape == (3 + 3 + 3 + n + n + n + 2,)

    # ang_vel scaled
    np.testing.assert_allclose(obs[0:3], state.base_ang_vel * scales.ang_vel, rtol=1e-6)
    # projected gravity for identity quat is [0, 0, 1]
    np.testing.assert_allclose(obs[3:6], gravity_orientation(state.base_quat), rtol=1e-6)
    # command * command_scale * max_cmd
    np.testing.assert_allclose(
        obs[6:9], cmd * np.asarray(scales.command) * np.asarray([0.8, 0.5, 1.57]), rtol=1e-6
    )
    # dof_pos residual (0.2 - 0.1 = 0.1), scaled by dof_pos=1.0
    np.testing.assert_allclose(obs[9 : 9 + n], np.full(n, 0.1), rtol=1e-5)
    # dof_vel scaled by 0.05
    np.testing.assert_allclose(obs[9 + n : 9 + 2 * n], np.full(n, 0.5 * 0.05), rtol=1e-6)
    # last_action starts at zero on first tick
    np.testing.assert_allclose(obs[9 + 2 * n : 9 + 3 * n], np.zeros(n), atol=1e-7)
    # gait sin/cos at phase 0 -> sin=0, cos=1
    np.testing.assert_allclose(obs[-2:], [0.0, 1.0], atol=1e-7)


def test_unitree_gaitless_drops_phase_terms():
    n = 6
    ctrl = UnitreeLocoController(
        num_actions=n,
        default_pos=[0.0] * n,
        forward_fn=lambda obs: np.zeros(n, dtype=np.float32),
        with_gait=False,
    )
    obs = ctrl.observe(
        _state(n), Reference(kind=ReferenceKind.VELOCITY, value=np.zeros(3, dtype=np.float32))
    )
    assert obs.shape == (3 + 3 + 3 + n + n + n,)


def test_unitree_rejects_wrong_reference_kind():
    n = 6
    ctrl = UnitreeLocoController(
        num_actions=n,
        default_pos=[0.0] * n,
        forward_fn=lambda obs: np.zeros(n, dtype=np.float32),
    )
    with pytest.raises(ValueError, match="accepts"):
        ctrl.step(_state(n), Reference(kind=ReferenceKind.LATENT, value=np.zeros(8)))


def test_unitree_phase_advances_with_steps():
    n = 4
    ctrl = UnitreeLocoController(
        num_actions=n,
        default_pos=[0.0] * n,
        forward_fn=lambda obs: np.zeros(n, dtype=np.float32),
        control_hz=50,
        cycle_time=0.8,
    )
    ref = Reference(kind=ReferenceKind.VELOCITY, value=np.zeros(3, dtype=np.float32))
    o0 = ctrl.observe(_state(n), ref)
    ctrl.step(_state(n), ref)  # advances timestep
    o1 = ctrl.observe(_state(n), ref)
    # phase moved, so sin/cos differ
    assert not np.allclose(o0[-2:], o1[-2:])


# ----- post-processing math ------------------------------------------------
def test_postprocess_offset_and_scale():
    """step applies EMA(beta) -> clip -> scale, then adds default_pos."""
    n = 2
    ctrl = UnitreeLocoController(
        num_actions=n,
        default_pos=[1.0, 2.0],
        forward_fn=lambda obs: np.array([4.0, -4.0], dtype=np.float32),
        action_scale=0.25,
        action_clip=2.0,
        action_beta=1.0,
        with_gait=False,
    )
    ref = Reference(kind=ReferenceKind.VELOCITY, value=np.zeros(3, dtype=np.float32))
    out = ctrl.step(_state(n), ref)
    # raw [4,-4] -> clip to [2,-2] -> scale 0.25 -> [0.5,-0.5] -> + default [1,2]
    np.testing.assert_allclose(out.position, [1.5, 1.5], rtol=1e-6)


def test_reset_clears_smoothing():
    n = 2
    ctrl = UnitreeLocoController(
        num_actions=n,
        default_pos=[0.0, 0.0],
        forward_fn=lambda obs: np.array([1.0, 1.0], dtype=np.float32),
        action_beta=0.5,
        with_gait=False,
    )
    ref = Reference(kind=ReferenceKind.VELOCITY, value=np.zeros(3, dtype=np.float32))
    ctrl.step(_state(n), ref)  # last_action becomes 0.5
    ctrl.reset()
    assert np.allclose(ctrl._last_action, 0.0)

"""ASAP reference-tracking controller: observation layout and phase clock."""

import numpy as np
import pytest

from hiwbc import ControllerState, Reference, ReferenceKind, available_controllers
from hiwbc.controllers.asap import AsapObsScales, AsapTrackerController
from hiwbc.math_utils import quat_rotate_inverse


def _state(n: int, quat=None) -> ControllerState:
    return ControllerState(
        dof_pos=np.full(n, 0.3, dtype=np.float32),
        dof_vel=np.full(n, 0.7, dtype=np.float32),
        base_quat=np.array(quat if quat is not None else [0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        base_ang_vel=np.array([0.1, 0.2, 0.3], dtype=np.float32),
    )


def test_asap_registered():
    assert "asap_tracker" in available_controllers()


def test_asap_obs_layout_no_history():
    """Obs = last_action(n) + ang_vel(3) + dof_pos(n) + dof_vel(n) + gravity(3) + phase(1)."""
    n = 23
    captured = {}

    def fake_forward(obs):
        captured["obs"] = obs
        return np.zeros(n, dtype=np.float32)

    ctrl = AsapTrackerController(
        num_actions=n,
        default_pos=[0.1] * n,
        forward_fn=fake_forward,
        motion_length_s=10.0,
        use_history=False,
        obs_scales=AsapObsScales(),
    )
    ref = Reference(kind=ReferenceKind.MOTION_FRAMES)
    ctrl.step(_state(n), ref)

    obs = captured["obs"]
    assert obs.shape == (n + 3 + n + n + 3 + 1,)
    # last_action zero on first tick
    np.testing.assert_allclose(obs[0:n], np.zeros(n), atol=1e-7)
    # base_ang_vel
    np.testing.assert_allclose(obs[n : n + 3], [0.1, 0.2, 0.3], rtol=1e-6)
    # dof_pos residual 0.3 - 0.1 = 0.2
    np.testing.assert_allclose(obs[n + 3 : 2 * n + 3], np.full(n, 0.2), rtol=1e-5)
    # dof_vel
    np.testing.assert_allclose(obs[2 * n + 3 : 3 * n + 3], np.full(n, 0.7), rtol=1e-6)
    # projected gravity for identity quat -> [0, 0, -1]
    np.testing.assert_allclose(obs[3 * n + 3 : 3 * n + 6], [0.0, 0.0, -1.0], atol=1e-6)
    # phase 0 on first tick
    np.testing.assert_allclose(obs[-1], 0.0, atol=1e-7)


def test_asap_obs_layout_with_history():
    n = 5
    hist_len = 4
    captured = {}

    def fake_forward(obs):
        captured["obs"] = obs
        return np.zeros(n, dtype=np.float32)

    ctrl = AsapTrackerController(
        num_actions=n,
        default_pos=[0.0] * n,
        forward_fn=fake_forward,
        motion_length_s=5.0,
        use_history=True,
        history_length=hist_len,
    )
    ctrl.reset()
    ctrl.step(_state(n), Reference(kind=ReferenceKind.MOTION_FRAMES))
    block = n * 3 + 3 + 3 + 1  # last_action + ang_vel + dof_pos + dof_vel + gravity + phase
    # obs = proprio_front(last_action+ang_vel+dof_pos+dof_vel = n+3+n+n) + history(block*hist_len)
    #       + gravity(3) + phase(1)
    front = n + 3 + n + n
    assert captured["obs"].shape == (front + block * hist_len + 3 + 1,)


def test_asap_phase_advances_and_completes():
    n = 3
    ctrl = AsapTrackerController(
        num_actions=n,
        default_pos=[0.0] * n,
        forward_fn=lambda obs: np.zeros(n, dtype=np.float32),
        motion_length_s=0.1,  # 5 ticks at 50Hz -> 0.1s
        control_hz=50,
    )
    ref = Reference(kind=ReferenceKind.MOTION_FRAMES)
    assert ctrl._phase() == 0.0
    for _ in range(10):
        ctrl.step(_state(n), ref)
    assert ctrl.motion_done


def test_asap_rejects_wrong_reference_kind():
    n = 3
    ctrl = AsapTrackerController(
        num_actions=n,
        default_pos=[0.0] * n,
        forward_fn=lambda obs: np.zeros(n, dtype=np.float32),
        motion_length_s=5.0,
    )
    with pytest.raises(ValueError, match="accepts"):
        ctrl.step(_state(n), Reference(kind=ReferenceKind.VELOCITY, value=np.zeros(3)))


def test_quat_rotate_inverse_matches_gravity_for_rotations():
    """For a 90deg yaw, projected gravity stays [0,0,-1] (gravity is along z)."""
    # yaw 90deg about z: quat [0,0,sin(45),cos(45)]
    s = np.sin(np.pi / 4)
    quat = np.array([0.0, 0.0, s, s], dtype=np.float32)
    g = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0], dtype=np.float32))
    np.testing.assert_allclose(g, [0.0, 0.0, -1.0], atol=1e-6)
    # for a +90deg pitch about y, the inverse rotation maps world -z onto body +x
    quat_pitch = np.array([0.0, s, 0.0, s], dtype=np.float32)
    g2 = quat_rotate_inverse(quat_pitch, np.array([0.0, 0.0, -1.0], dtype=np.float32))
    np.testing.assert_allclose(g2, [1.0, 0.0, 0.0], atol=1e-6)

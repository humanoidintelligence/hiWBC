"""SuperMimic tracker: input layout, history order, action mapping (no weights)."""

import numpy as np
import pytest

from hiwbc import ControllerState, Reference, ReferenceKind, available_controllers
from hiwbc.controllers.supermimic import (
    GOAL_DIM,
    NUM_JOINTS,
    SuperMimicController,
    goal_features,
)

N = NUM_JOINTS


def _ctl(fn, **kw):
    kw.setdefault("bound_torque", False)
    return SuperMimicController(
        forward_fn=fn,
        default_pos=np.linspace(-0.3, 0.3, N),
        action_scale=np.full(N, 0.5),
        kp=np.full(N, 40.0),
        kd=np.full(N, 2.0),
        effort=np.full(N, 25.0),
        **kw,
    )


def _state(k: float) -> ControllerState:
    return ControllerState(
        dof_pos=np.full(N, k, dtype=np.float32),
        dof_vel=np.full(N, 10 * k, dtype=np.float32),
        base_quat=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        base_ang_vel=np.full(3, 100 * k, dtype=np.float32),
    )


def _ref(goal) -> Reference:
    return Reference(ReferenceKind.MOTION_FRAMES, frames={"goal": goal})


def test_registered():
    assert "supermimic" in available_controllers()


def test_shapes_and_first_tick_fills_history():
    seen = {}

    def fn(obs):
        seen["obs"] = obs
        return np.zeros(N, dtype=np.float32)

    ctl = _ctl(fn)
    ctl.step(_state(0.1), _ref(np.zeros((30, GOAL_DIM), np.float32)))
    hist, goal = seen["obs"]
    assert hist.shape == (930,) and goal.shape == (680,)
    # term-major: first 290 values are 10 copies of q - q0
    q_block = hist[: 10 * N].reshape(10, N)
    np.testing.assert_allclose(q_block, np.tile(0.1 - ctl.default_pos, (10, 1)), atol=1e-6)
    # gravity block for identity quaternion is [0, 0, -1] in every slot
    grav = hist[10 * (2 * N + 3): 10 * (2 * N + 6)].reshape(10, 3)
    np.testing.assert_allclose(grav, np.tile([0, 0, -1], (10, 1)), atol=1e-6)


def test_history_is_term_major_oldest_first():
    seen = {}

    def fn(obs):
        seen["obs"] = obs
        return np.zeros(N, dtype=np.float32)

    ctl = _ctl(fn)
    goal = np.zeros((30, GOAL_DIM), np.float32)
    for k in (0.1, 0.2, 0.3):
        ctl.step(_state(k), _ref(goal))
    hist, _ = seen["obs"]
    ang = hist[10 * 2 * N: 10 * (2 * N + 3)].reshape(10, 3)[:, 0]
    # 8 copies of tick-0 entry, then tick 1, then tick 2 (newest last)
    np.testing.assert_allclose(ang, [10.0] * 8 + [20.0, 30.0], rtol=1e-5)


def test_prev_action_is_raw_and_target_is_q0_plus_scale():
    seen = {}
    raw = np.linspace(-1, 1, N).astype(np.float32)

    def fn(obs):
        seen.setdefault("obs", []).append(obs)
        return raw

    ctl = _ctl(fn)
    goal = np.zeros((30, GOAL_DIM), np.float32)
    out = ctl.step(_state(0.0), _ref(goal))
    np.testing.assert_allclose(out.position, ctl.default_pos + 0.5 * raw, atol=1e-6)
    ctl.step(_state(0.0), _ref(goal))
    hist, _ = seen["obs"][1]
    prev = hist[10 * (2 * N + 6):].reshape(10, N)
    np.testing.assert_allclose(prev[-1], raw, atol=1e-6)
    np.testing.assert_allclose(prev[0], 0.0, atol=1e-6)
    assert out.kp is not None and out.kd is not None


def test_goal_window_holds_last_frame():
    seen = {}

    def fn(obs):
        seen["obs"] = obs
        return np.zeros(N, dtype=np.float32)

    ctl = _ctl(fn)
    goal = np.arange(3, dtype=np.float32)[:, None] * np.ones((3, GOAL_DIM), np.float32)
    ctl.step(_state(0.0), _ref(goal))
    win = seen["obs"][1].reshape(10, GOAL_DIM)[:, 0]
    np.testing.assert_allclose(win, [0, 1, 2, 2, 2, 2, 2, 2, 2, 2])


def test_bound_torque_clips_pd_torque():
    raw = np.full(N, 100.0, dtype=np.float32)  # huge action -> saturated torque
    ctl = _ctl(lambda obs: raw, bound_torque=True)
    st = _state(0.0)
    out = ctl.step(st, _ref(np.zeros((10, GOAL_DIM), np.float32)))
    tau = ctl.kp * (out.position - st.dof_pos) - ctl.kd * st.dof_vel
    np.testing.assert_allclose(tau, ctl.effort, rtol=1e-5)


def test_bad_goal_rejected():
    ctl = _ctl(lambda obs: np.zeros(N, np.float32))
    with pytest.raises(ValueError):
        ctl.step(_state(0.0), _ref(np.zeros((10, 67), np.float32)))
    with pytest.raises(ValueError):
        ctl.step(_state(0.0), Reference(ReferenceKind.MOTION_FRAMES))


def test_goal_features_static_pose():
    T = 4
    q0 = np.linspace(-0.3, 0.3, N)
    g = goal_features(
        joint_pos=np.tile(q0, (T, 1)), joint_vel=np.zeros((T, N)),
        root_quat=np.tile([0, 0, 0, 1.0], (T, 1)),
        root_lin_vel_w=np.tile([1.0, 0, 0], (T, 1)), root_ang_vel_w=np.zeros((T, 3)),
        root_z=np.full(T, 0.78), default_pos=q0,
    )
    assert g.shape == (T, GOAL_DIM)
    np.testing.assert_allclose(g[:, :2 * N], 0.0, atol=1e-6)
    np.testing.assert_allclose(g[0, 2 * N:], [0, 0, -1, 0, 0, 0, 1, 0, 0, 0.78], atol=1e-6)

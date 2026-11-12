"""AMO controller: obs layout, two-input forward, histories, gait clock, adapter path."""

import numpy as np
import pytest

from hiwbc import ControllerState, Reference, ReferenceKind, available_controllers
from hiwbc.controllers.amo import AmoLocoController

NA, NU = 15, 8          # actuated (lower body + waist), upper-body (arms)
ND = NA + NU
NOUT = 15               # adapter output


def _ctl(captured=None, **kw):
    captured = {} if captured is None else captured

    def fwd(obs, extra):
        captured["obs"], captured["extra"] = obs, extra
        return np.full(NA, 100.0, dtype=np.float32)  # exercise the +-40 raw clip

    def adapter(x):
        captured["adapter_in"] = x
        return np.ones(NOUT, dtype=np.float32)

    args = {
        "num_actions": NA, "default_pos": [0.1] * NA,
        "obs_default_pos": [0.1] * NA + [0.2] * NU,
        "forward_fn": fwd, "adapter_fn": adapter,
        "adapter_input_mean": np.zeros(4 + NU), "adapter_input_std": np.full(4 + NU, 2.0),
        "adapter_output_mean": np.full(NOUT, 0.5), "adapter_output_std": np.full(NOUT, 3.0),
    }
    args.update(kw)
    return AmoLocoController(**args), captured


def _state(tick=0):
    return ControllerState(
        dof_pos=np.full(ND, float(tick + 1), dtype=np.float32),
        dof_vel=np.full(ND, 2.0, dtype=np.float32),
        base_quat=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        base_ang_vel=np.array([4.0, 0.0, 0.0], dtype=np.float32),
    )


def _vel(*v):
    return Reference(ReferenceKind.VELOCITY, value=np.array(v, dtype=np.float32))


def test_registered():
    assert "amo_loco" in available_controllers()


def test_obs_layout_and_sizes():
    ctl, cap = _ctl()
    ctl.step(_state(), _vel(0.0, 0.0, 0.0))
    p = 3 + 2 + 2 + 3 * ND + 2 + NOUT
    assert ctl.proprio_size == p
    assert cap["obs"].shape == (p + (NU + 9) + 3 + 10 * p,)
    assert cap["extra"].shape == (25 * p,)
    o = cap["obs"]
    np.testing.assert_allclose(o[0:3], [1.0, 0.0, 0.0])          # ang_vel * 0.25
    np.testing.assert_allclose(o[5:7], [0.0, 1.0], atol=1e-6)     # sin/cos(dyaw=0)
    np.testing.assert_allclose(o[7:7 + NA], 0.9, rtol=1e-6)       # dof_pos - default
    np.testing.assert_allclose(o[7 + 2 * ND: 7 + 2 * ND + NA], 0.0)  # last_action (lower)
    np.testing.assert_allclose(o[7 + 2 * ND + NA: 7 + 3 * ND], 0.8 / 0.25, rtol=1e-6)
    np.testing.assert_allclose(o[p - NOUT: p], 3.5)               # 1 * std 3 + mean 0.5


def test_adapter_input_normalized():
    ctl, cap = _ctl()
    ctl.step(_state(), _vel(0.0, 0.0, 0.0, 0.6, 0.1, 0.2, 0.3))
    np.testing.assert_allclose(cap["adapter_in"][:4], np.array([0.6, 0.1, 0.2, 0.3]) / 2.0,
                               rtol=1e-6)
    np.testing.assert_allclose(cap["adapter_in"][4:], 0.5)        # arm dof_pos 1.0 / 2


def test_demo_obs_and_dyaw():
    ctl, cap = _ctl()
    ctl.step(_state(), _vel(0.3, 0.5, 0.7, 0.6, 0.1, 0.2, 0.3))
    o = cap["obs"]
    np.testing.assert_allclose(o[5:7], [np.sin(-0.5), np.cos(-0.5)], rtol=1e-6)
    p = ctl.proprio_size
    demo = o[p: p + NU + 9]
    np.testing.assert_allclose(demo[:NU], 1.0)
    np.testing.assert_allclose(demo[NU:], [0.3, 0.7, 0.0, 0.1, 0.2, 0.3, 1.35, 1.35, 1.35],
                               rtol=1e-6)


def test_three_value_command_uses_defaults():
    ctl, cap = _ctl(default_command=[0, 0, 0, 0.75, 0, 0, 0])
    ctl.step(_state(), _vel(0.0, 0.0, 0.0))
    np.testing.assert_allclose(cap["adapter_in"][0], 0.375, rtol=1e-6)


def test_histories_order():
    ctl, cap = _ctl()
    for t in range(3):
        ctl.step(_state(t), _vel(0.0, 0.0, 0.0))
    p = ctl.proprio_size
    hist = cap["obs"][p + NU + 9 + 3:].reshape(10, p)
    # previous frames, oldest first, current excluded: ..., zero, tick0, tick1
    np.testing.assert_allclose(hist[-2:, 7], [0.9, 1.9], rtol=1e-6)
    np.testing.assert_allclose(hist[:-2], 0.0)
    extra = cap["extra"].reshape(25, p)
    # last 25 frames incl. current, oldest first
    np.testing.assert_allclose(extra[-3:, 7], [0.9, 1.9, 2.9], rtol=1e-6)


def test_raw_clip_then_scale():
    ctl, _ = _ctl()
    out = ctl.step(_state(), _vel(0.0, 0.0, 0.0))
    np.testing.assert_allclose(out.position, 40.0 * 0.25 + 0.1, rtol=1e-6)
    np.testing.assert_allclose(ctl._last_action, 40.0)


def test_gait_clock_stand_vs_walk():
    ctl, _ = _ctl()
    for _ in range(50):
        ctl.step(_state(), _vel(0.0, 0.0, 0.0))
    np.testing.assert_allclose(ctl.gait_cycle, [0.25, 0.25])      # standing: phase locked
    walk, _ = _ctl()
    walk.step(_state(), _vel(0.5, 0.0, 0.0))
    np.testing.assert_allclose(walk.gait_cycle, [0.25, 0.75])     # start walking: antiphase


def test_reset_clears_state():
    ctl, cap = _ctl()
    for t in range(3):
        ctl.step(_state(t), _vel(0.5, 0.0, 0.0))
    ctl.reset()
    np.testing.assert_allclose(ctl.gait_cycle, [0.25, 0.25])
    ctl.step(_state(), _vel(0.0, 0.0, 0.0))
    p = ctl.proprio_size
    np.testing.assert_allclose(cap["obs"][p + NU + 9 + 3:], 0.0)


def test_rejects_bad_inputs():
    ctl, _ = _ctl()
    with pytest.raises(ValueError):
        ctl.step(_state(), _vel(0.0, 0.0))
    with pytest.raises(ValueError):
        ctl.step(_state(), Reference(ReferenceKind.MOTION_FRAMES))
    with pytest.raises(ValueError):
        _ctl(default_pos=[0.0] * NA)  # actuated defaults must lead obs defaults

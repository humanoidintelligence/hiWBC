"""History-buffer layouts, pinned to RoboJuDo.

Each family flattens its history differently; a lockstep comparison against
RoboJuDo found hiWBC had ASAP and KungfuBot wrong (frame-major instead
of term-major). These tests pin the layouts with distinguishable per-tick values,
so a regression shows up as a value mismatch, not just a size mismatch.

Layouts (RoboJuDo source):
  ASAP / KungfuBot2  term-major, newest first   (appendleft + zip(*buf))
  Unitree wo-gait    term-major, oldest first   (append + zip(*buf)), current tick included
  TWIST / Smooth     frame-major, oldest first  (append + flatten), current tick excluded
  ProtoMotions(H2H)  frame-major, newest first  (appendleft + flatten), current tick excluded
"""

import numpy as np

from hiwbc import ControllerState, Reference, ReferenceKind
from hiwbc.controllers.asap import AsapTrackerController
from hiwbc.controllers.kungfubot import KungfuBotTrackerController
from hiwbc.controllers.protomotions import ProtoMotionsTrackerController
from hiwbc.controllers.smooth import SmoothLocoController
from hiwbc.controllers.twist import TwistTrackerController
from hiwbc.controllers.unitree import UnitreeLocoController

N = 4


def _state(tick: int) -> ControllerState:
    """dof_pos encodes the tick so each history frame is identifiable."""
    return ControllerState(
        dof_pos=np.full(N, float(tick + 1), dtype=np.float32),
        dof_vel=np.zeros(N, dtype=np.float32),
        base_quat=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        base_ang_vel=np.zeros(3, dtype=np.float32),
    )


def _zero_fn(n):
    return lambda obs: np.zeros(n, dtype=np.float32)


def _run(ctl, ref, ticks):
    """Step ``ticks`` times and return the obs of the last tick."""
    captured = {}
    orig = ctl.observe

    def spy(s, r):
        captured["obs"] = orig(s, r)
        return captured["obs"]

    ctl.observe = spy
    for t in range(ticks):
        ctl.step(_state(t), ref)
    return captured["obs"]


def test_asap_history_term_major_newest_first():
    H = 3
    ctl = AsapTrackerController(num_actions=N, default_pos=[0.0] * N, forward_fn=_zero_fn(N),
                                motion_length_s=100.0, use_history=True, history_length=H)
    obs = _run(ctl, Reference(ReferenceKind.MOTION_FRAMES), ticks=3)  # ticks 0,1,2 -> pos 1,2,3
    # front: last_action(N) ang_vel(3) dof_pos(N) dof_vel(N), then history
    hist = obs[3 * N + 3: 3 * N + 3 + H * (3 * N + 3 + 3 + 1)]
    # term-major: [last_action x H][ang_vel x H][dof_pos x H]...; dof_pos block newest first
    dof_block = hist[H * N + H * 3: H * N + H * 3 + H * N].reshape(H, N)
    # current tick (pos=3) not yet in history; newest stored is tick 1 (pos 2), then tick 0, then zero-fill
    np.testing.assert_allclose(dof_block[:, 0], [2.0, 1.0, 0.0])


def test_kungfubot_history_term_major_newest_first():
    H = 3
    ref = Reference(ReferenceKind.MOTION_FRAMES, frames={
        "anchor_ref_rot": np.zeros(6), "next_step_ref_motion": np.zeros(N),
        "future_motion_base_lin_vel": np.zeros(2), "future_motion_base_yaw_vel": np.zeros(1),
        "future_motion_dof_pos": np.zeros(N), "future_motion_roll_pitch": np.zeros(2),
        "future_motion_root_height": np.zeros(1)})
    ctl = KungfuBotTrackerController(num_actions=N, default_pos=[0.0] * N,
                                     forward_fn=_zero_fn(N), history_length=H)
    _, _, hist = _run(ctl, ref, ticks=3)
    # entry = action(N) ang_vel(3) dof_pos(N) dof_vel(N) roll_pitch(2)
    dof_block = hist[H * N + H * 3: H * N + H * 3 + H * N].reshape(H, N)
    np.testing.assert_allclose(dof_block[:, 0], [2.0, 1.0, 0.0])


def test_kungfubot_reset_clears_history():
    ref = Reference(ReferenceKind.MOTION_FRAMES, frames={
        "anchor_ref_rot": np.zeros(6), "next_step_ref_motion": np.zeros(N),
        "future_motion_base_lin_vel": np.zeros(2), "future_motion_base_yaw_vel": np.zeros(1),
        "future_motion_dof_pos": np.zeros(N), "future_motion_roll_pitch": np.zeros(2),
        "future_motion_root_height": np.zeros(1)})
    ctl = KungfuBotTrackerController(num_actions=N, default_pos=[0.0] * N,
                                     forward_fn=_zero_fn(N), history_length=2)
    _run(ctl, ref, ticks=3)
    ctl.reset()
    _, _, hist = _run(ctl, ref, ticks=1)
    np.testing.assert_allclose(hist, 0.0)


def test_unitree_wogait_history_term_major_oldest_first_includes_current():
    H = 3
    ctl = UnitreeLocoController(num_actions=N, default_pos=[0.0] * N, forward_fn=_zero_fn(N),
                                with_gait=False, history_length=H)
    obs = _run(ctl, Reference(ReferenceKind.VELOCITY, value=np.zeros(3)), ticks=2)
    # blocks: ang_vel(3H) gravity(3H) command(3H) dof_pos(NH) ...
    dof_block = obs[9 * H: 9 * H + N * H].reshape(H, N)
    np.testing.assert_allclose(dof_block[:, 0], [0.0, 1.0, 2.0])  # zero-fill, tick0, current tick1
    assert obs.size == H * (9 + 3 * N)


def test_unitree_wogait_gravity_scale():
    ctl = UnitreeLocoController(num_actions=N, default_pos=[0.0] * N, forward_fn=_zero_fn(N),
                                with_gait=False, history_length=1, gravity_scale=0.5)
    obs = _run(ctl, Reference(ReferenceKind.VELOCITY, value=np.zeros(3)), ticks=1)
    np.testing.assert_allclose(obs[3:6], [0.0, 0.0, -0.5])


def test_twist_history_frame_major_oldest_first_and_ankle_mask():
    H = 3
    mask = np.zeros(N, dtype=np.float32)
    mask[1] = 1.0
    ctl = TwistTrackerController(num_actions=N, default_pos=[0.0] * N, forward_fn=_zero_fn(N),
                                 history_length=H, zero_dof_vel_mask=mask)
    ref = Reference(ReferenceKind.MOTION_FRAMES, frames={"mimic_target": np.zeros(2)})
    state = _state(0)
    state.dof_vel[:] = 1.0
    full_size = 2 + 3 + 2 + 3 * N
    obs = ctl.observe(state, ref)
    vel = obs[2 + 3 + 2 + N: 2 + 3 + 2 + 2 * N]
    np.testing.assert_allclose(vel, [1.0, 0.0, 1.0, 1.0])
    np.testing.assert_allclose(obs[full_size:], 0.0)  # first tick: zero history
    obs2 = _run(TwistTrackerController(num_actions=N, default_pos=[0.0] * N,
                                       forward_fn=_zero_fn(N), history_length=H),
                ref, ticks=3)
    frames = obs2[full_size:].reshape(H, full_size)
    dof = frames[:, 7]  # first dof_pos entry in each frame
    np.testing.assert_allclose(dof, [0.0, 1.0, 2.0])


def test_smooth_history_seeded_without_reset():
    ctl = SmoothLocoController(num_actions=N, default_pos=[0.0] * N, forward_fn=_zero_fn(N),
                               history_length=2)
    obs = ctl.observe(_state(0), Reference(ReferenceKind.VELOCITY, value=np.zeros(3)))
    size = 1 + 1 + 3 + 3 + 2 + 3 * N
    assert obs.size == size * 3
    np.testing.assert_allclose(obs[size:], 0.0)


def test_protomotions_history_frame_major_newest_first():
    H = 3
    ctl = ProtoMotionsTrackerController(num_actions=N, default_pos=[0.0] * N,
                                        forward_fn=_zero_fn(N), history_length=H)
    ref = Reference(ReferenceKind.MOTION_FRAMES, frames={"task_obs": np.zeros(5)})
    obs = _run(ctl, ref, ticks=3)
    block = N + N + 3 + 3 + N
    frames = obs[-H * block:].reshape(H, block)
    np.testing.assert_allclose(frames[:, 0], [2.0, 1.0, 0.0])


def test_tracker_action_beta_passthrough():
    ctl = ProtoMotionsTrackerController(num_actions=N, default_pos=[0.0] * N,
                                        forward_fn=lambda o: np.ones(N, np.float32),
                                        action_beta=0.8)
    ref = Reference(ReferenceKind.MOTION_FRAMES, frames={"task_obs": np.zeros(1)})
    ctl.step(_state(0), ref)
    np.testing.assert_allclose(ctl._last_action, 0.8)

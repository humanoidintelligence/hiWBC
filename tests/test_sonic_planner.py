"""SONIC locomotion planner: replan rules, resampling, blending, commands (stub model).

The stub predicts a constant pose with joint values that encode the call number, so
blending and resampling are checked exactly. One test runs the released
``planner_sonic.onnx`` when it is in the checkpoint cache.
"""

import os

import numpy as np
import pytest

from hiwbc.configs.sonic.unitree_g1 import (
    G1_DEFAULT_POS,
    G1_LOWER_BODY_INDEX,
    ISAACLAB_TO_MUJOCO_INDEX,
    SONIC_CHECKPOINTS,
    SONIC_PLANNER,
)
from hiwbc.controllers.sonic_planner import LocomotionMode, SonicPlanner, velocity_command

IDENT = np.array([1.0, 0.0, 0.0, 0.0])   # [w, x, y, z]


def _yaw_wxyz(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


class _Stub:
    """Planner model stub: ``frames`` 30 Hz frames, joints = call number, root at x = call."""

    def __init__(self, frames=30):
        self.frames = frames
        self.feeds: list[dict] = []

    def __call__(self, feeds):
        self.feeds.append(feeds)
        k = len(self.feeds)
        q = np.zeros((1, 64, 36), np.float32)
        q[0, :, 0] = k
        q[0, :, 2] = 0.79
        q[0, :, 3] = 1.0
        q[0, :, 7:] = k
        return q, np.array([self.frames], np.int64)


def _planner(frames=30, **kw):
    stub = _Stub(frames)
    p = SonicPlanner(stub, qpos_joint_index=ISAACLAB_TO_MUJOCO_INDEX,
                     lower_body_index=G1_LOWER_BODY_INDEX, **kw)
    return p, stub


def _ticks(p, n, q=None):
    for _ in range(n):
        p.advance(G1_DEFAULT_POS if q is None else q)


def test_start_plans_idle_from_the_measured_joints():
    p, stub = _planner()
    p.start(G1_DEFAULT_POS)
    f = stub.feeds[0]
    assert f["mode"][0] == LocomotionMode.IDLE and f["target_vel"][0] == -1.0
    ctx = f["context_mujoco_qpos"][0]
    assert ctx.shape == (4, 36)
    np.testing.assert_allclose(ctx[:, 2], p.default_height)
    np.testing.assert_allclose(ctx[:, 3:7], np.tile(IDENT, (4, 1)))
    # joints go into MuJoCo order: qpos column 7 + mujoco_index holds controller joint i
    np.testing.assert_allclose(ctx[0, 7 + ISAACLAB_TO_MUJOCO_INDEX], G1_DEFAULT_POS, atol=1e-6)


def test_prediction_resampled_from_30_to_50_hz():
    p, _ = _planner(frames=30)        # 1 s of prediction
    p.start(G1_DEFAULT_POS)
    assert len(p.motion) == 50
    np.testing.assert_allclose(p.motion.joint_vel, 0.0)   # constant pose


def test_bad_predictions_raise():
    for frames in (1, 65):
        p, _ = _planner(frames=frames)
        with pytest.raises(RuntimeError, match="predicted"):
            p.start(G1_DEFAULT_POS)

    def nan(feeds):
        return np.full((1, 64, 36), np.nan, np.float32), np.array([30])

    p = SonicPlanner(nan, qpos_joint_index=ISAACLAB_TO_MUJOCO_INDEX, lower_body_index=G1_LOWER_BODY_INDEX)
    with pytest.raises(RuntimeError, match="NaN"):
        p.start(G1_DEFAULT_POS)


def test_qpos_index_must_be_a_permutation():
    with pytest.raises(ValueError, match="permutation"):
        SonicPlanner(_Stub(), qpos_joint_index=np.zeros(29, int), lower_body_index=G1_LOWER_BODY_INDEX)


def test_replans_on_a_command_change_at_the_10_hz_check():
    p, stub = _planner()
    p.start(G1_DEFAULT_POS)
    p.set_command(LocomotionMode.WALK, 1.0, 0.0, 0.0)
    _ticks(p, 4)
    assert len(stub.feeds) == 1          # the check runs every 5 ticks (0.1 s)
    _ticks(p, 1)
    assert len(stub.feeds) == 2
    f = stub.feeds[1]
    assert f["mode"][0] == LocomotionMode.WALK and f["target_vel"][0] == 1.0
    np.testing.assert_allclose(f["movement_direction"][0], [1, 0, 0], atol=1e-7)


def test_moving_replans_every_second_running_every_tenth():
    p, stub = _planner(frames=64)
    p.start(G1_DEFAULT_POS)
    p.set_command(LocomotionMode.WALK, 1.0, 0.0, 0.0)
    _ticks(p, 5)
    n = len(stub.feeds)
    _ticks(p, 50)                          # 1 s more of the same command
    assert len(stub.feeds) == n + 1
    p.set_command(LocomotionMode.RUN, 2.0, 0.0, 0.0)
    _ticks(p, 5)
    n = len(stub.feeds)
    _ticks(p, 25)                          # 0.5 s: one replan per 0.1 s check
    assert len(stub.feeds) == n + 5


def test_idle_does_not_replan_on_the_timer():
    p, stub = _planner(frames=64)
    p.start(G1_DEFAULT_POS)
    p.set_command(*_idle(0.0))
    _ticks(p, 200)
    assert len(stub.feeds) == 1


def _idle(face):
    return (LocomotionMode.IDLE, -1.0, None, face)


def test_turning_in_idle_replans_on_the_facing_change():
    p, stub = _planner(frames=64)
    p.start(G1_DEFAULT_POS)
    p.set_command(*_idle(0.3))
    _ticks(p, 5)
    assert len(stub.feeds) == 2
    np.testing.assert_allclose(stub.feeds[1]["facing_direction"][0], [np.cos(0.3), np.sin(0.3), 0], atol=1e-6)


def test_blend_fades_the_new_plan_in_over_8_frames_after_look_ahead():
    p, _ = _planner(frames=30)
    p.start(G1_DEFAULT_POS)                # plan 1: joints = 1
    p.set_command(LocomotionMode.WALK, 1.0, 0.0, 0.0)
    _ticks(p, 4)
    p.advance(G1_DEFAULT_POS)              # plan 2 (joints = 2) blended at this tick
    # rebased to frame 0 at the replan, then advanced one frame
    assert p.frame == 1
    j = p.motion.joint_pos[:, 0]
    # frames 0, 1 (before look_ahead = 2) are the old plan; the new one fades in linearly
    np.testing.assert_allclose(j[:3], 1.0)
    np.testing.assert_allclose(j[2:11], 1.0 + np.arange(9) / 8)
    np.testing.assert_allclose(j[11:], 2.0)


def test_motion_holds_its_last_frame():
    p, _ = _planner(frames=6)              # 10 frames at 50 Hz
    p.start(G1_DEFAULT_POS)
    _ticks(p, 30)
    assert p.frame == len(p.motion) - 1


def test_idle_readaptation_pulls_the_legs_toward_the_measured_joints():
    p, _ = _planner(frames=6)
    p.start(G1_DEFAULT_POS)                # plan joints = 1
    legs = G1_LOWER_BODY_INDEX
    q = np.ones(29)
    q[legs] = 1.5                          # legs 0.5 rad off the reference: "adapting"
    _ticks(p, 12, q)
    held = p.motion.joint_pos[-1]
    assert np.all(held[legs] > 1.0) and np.all(held[legs] < 1.5)
    others = np.setdiff1d(np.arange(29), legs)
    np.testing.assert_allclose(held[others], 1.0)


def test_window_reanchors_heading_to_the_robot_and_clamps():
    p, _ = _planner(frames=6)
    p.start(G1_DEFAULT_POS)
    yaw = 1.0
    w = p.window(20, _yaw_wxyz(yaw))
    assert w["joint_pos"].shape == (20, 29) and w["root_z"].shape == (20,)
    qx, qy, qz, qw = w["root_quat"][0]
    assert np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)) == pytest.approx(yaw, abs=1e-6)
    assert p.robot_yaw(_yaw_wxyz(yaw)) == pytest.approx(0.0, abs=1e-6)
    # the anchor stays fixed: a robot that turned further reads as turned in the planner frame
    p.window(20, _yaw_wxyz(yaw + 0.5))
    assert p.robot_yaw(_yaw_wxyz(yaw + 0.5)) == pytest.approx(0.5, abs=1e-6)


@pytest.mark.parametrize("v,mode", [((0.0, 0.0, 0.0), LocomotionMode.IDLE),
                                    ((0.5, 0.0, 0.0), LocomotionMode.SLOW_WALK),
                                    ((1.2, 0.0, 0.0), LocomotionMode.WALK),
                                    ((2.0, 0.0, 0.0), LocomotionMode.RUN)])
def test_velocity_picks_the_mode_by_speed(v, mode):
    p, _ = _planner()
    p.start(G1_DEFAULT_POS)
    p.window(1, IDENT)
    velocity_command(p, v, IDENT, dt=0.02)
    assert p._pending[0] == mode


def test_velocity_direction_is_relative_to_the_facing():
    p, _ = _planner()
    p.start(G1_DEFAULT_POS)
    p.window(1, IDENT)
    velocity_command(p, (0.0, 0.5, 0.0), IDENT, dt=0.02)      # sideways left
    np.testing.assert_allclose(p._pending[2], [0, 1, 0], atol=1e-7)
    assert p._pending[1] == pytest.approx(0.5)


def test_yaw_rate_integrates_the_facing_with_a_bounded_lead():
    p, _ = _planner()
    p.start(G1_DEFAULT_POS)
    p.window(1, IDENT)
    for _ in range(50):
        velocity_command(p, (0.0, 0.0, 0.5), IDENT, dt=0.02)
    assert p.facing_yaw == pytest.approx(0.5)
    for _ in range(500):                   # the robot does not turn: the lead stops at pi/2
        velocity_command(p, (0.0, 0.0, 0.5), IDENT, dt=0.02)
    assert p.facing_yaw == pytest.approx(np.pi / 2, abs=0.011)


def test_background_replans_off_the_tick_thread():
    p, _ = _planner(background=True)
    p.start(G1_DEFAULT_POS)
    p.set_command(LocomotionMode.WALK, 1.0, 0.0, 0.0)
    _ticks(p, 5)
    p._future.result(timeout=5)            # the replan was submitted, not run inline
    _ticks(p, 1)                           # and is blended in at the next tick
    assert p._future is None and p.replans == 1
    assert p.motion.joint_pos[-1, 0] == pytest.approx(2.0)
    p.reset()


ONNX = os.path.join(os.path.expanduser(SONIC_CHECKPOINTS), SONIC_PLANNER)


@pytest.mark.skipif(not os.path.isfile(ONNX), reason="planner_sonic.onnx not downloaded")
def test_released_planner_walks_forward():
    pytest.importorskip("onnxruntime")
    from hiwbc.controllers.sonic_planner import load_planner_runner

    p = SonicPlanner(load_planner_runner(ONNX), qpos_joint_index=ISAACLAB_TO_MUJOCO_INDEX,
                     lower_body_index=G1_LOWER_BODY_INDEX)
    p.start(G1_DEFAULT_POS)
    p.window(1, IDENT)
    x0 = p.motion.root_pos[p.frame, 0]
    for _ in range(150):                    # 3 s at 0.5 m/s
        velocity_command(p, (0.5, 0.0, 0.0), IDENT, dt=0.02)
        p.window(1, IDENT)
        p.advance(G1_DEFAULT_POS)
    dx = p.motion.root_pos[p.frame, 0] - x0
    assert 0.8 < dx < 2.0
    assert 0.6 < p.motion.root_pos[p.frame, 2] < 0.9

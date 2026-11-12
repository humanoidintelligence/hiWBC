"""Smooth + tracking-family controllers: obs layout, multi-input, registry."""

import numpy as np
import pytest

from hiwbc import ControllerState, Reference, ReferenceKind, available_controllers
from hiwbc.controllers.beyondmimic import BeyondMimicTrackerController
from hiwbc.controllers.kungfubot import KungfuBotTrackerController
from hiwbc.controllers.protomotions import ProtoMotionsTrackerController
from hiwbc.controllers.smooth import SmoothLocoController
from hiwbc.controllers.twist import TwistTrackerController


def _state(n: int) -> ControllerState:
    return ControllerState(
        dof_pos=np.full(n, 0.3, dtype=np.float32),
        dof_vel=np.full(n, 0.7, dtype=np.float32),
        base_quat=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        base_ang_vel=np.array([0.1, 0.2, 0.3], dtype=np.float32),
    )


def _capture():
    box = {}

    def fn(obs):
        box["obs"] = obs
        # num_actions is closed over by each test via the controller
        return np.zeros(box["n"], dtype=np.float32)

    return box, fn


def test_all_registered():
    names = available_controllers()
    for name in [
        "smooth_loco",
        "twist_tracker",
        "beyondmimic_tracker",
        "protomotions_tracker",
        "kungfubot_tracker",
    ]:
        assert name in names


# ----- smooth --------------------------------------------------------------
def test_smooth_obs_layout():
    n = 19
    box, fn = _capture()
    box["n"] = n
    ctrl = SmoothLocoController(
        num_actions=n, default_pos=[0.1] * n, forward_fn=fn, n_priv_latent=4, history_length=3
    )
    ctrl.reset()
    ctrl.step(_state(n), Reference(kind=ReferenceKind.VELOCITY, value=np.zeros(3, dtype=np.float32)))
    proprio = 1 + 1 + 3 + 3 + 2 + n * 3
    assert box["obs"].shape == (proprio + 4 + proprio * 3,)


def test_smooth_rejects_motion_frames():
    n = 6
    ctrl = SmoothLocoController(
        num_actions=n, default_pos=[0.0] * n, forward_fn=lambda o: np.zeros(n, dtype=np.float32)
    )
    with pytest.raises(ValueError, match="accepts"):
        ctrl.step(_state(n), Reference(kind=ReferenceKind.MOTION_FRAMES))


# ----- twist ---------------------------------------------------------------
def test_twist_obs_layout_and_mimic_prefix():
    n = 23
    box, fn = _capture()
    box["n"] = n
    ctrl = TwistTrackerController(num_actions=n, default_pos=[0.0] * n, forward_fn=fn)
    mimic = np.arange(10, dtype=np.float32)
    ref = Reference(kind=ReferenceKind.MOTION_FRAMES, frames={"mimic_target": mimic})
    ctrl.step(_state(n), ref)
    obs = box["obs"]
    # mimic(10) + proprio(ang_vel 3 + rp 2 + dof_pos n + dof_vel n + last_action n)
    assert obs.shape == (10 + 3 + 2 + 3 * n,)
    np.testing.assert_allclose(obs[:10], mimic)


def test_twist_missing_frames_raises():
    n = 4
    ctrl = TwistTrackerController(
        num_actions=n, default_pos=[0.0] * n, forward_fn=lambda o: np.zeros(n, dtype=np.float32)
    )
    with pytest.raises(ValueError, match="frames"):
        ctrl.step(_state(n), Reference(kind=ReferenceKind.MOTION_FRAMES, frames={}))


# ----- beyondmimic ---------------------------------------------------------
def test_beyondmimic_with_and_without_state_estimator():
    n = 12
    box, fn = _capture()
    box["n"] = n
    cmd = np.arange(4, dtype=np.float32)
    anchor_pos = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    anchor_ori = np.arange(6, dtype=np.float32)
    lin_vel = np.array([0.5, 0.6, 0.7], dtype=np.float32)
    frames = {
        "command": cmd,
        "motion_anchor_pos_b": anchor_pos,
        "motion_anchor_ori_b": anchor_ori,
        "base_lin_vel": lin_vel,
    }
    ctrl = BeyondMimicTrackerController(num_actions=n, default_pos=[0.0] * n, forward_fn=fn)
    ctrl.step(_state(n), Reference(kind=ReferenceKind.MOTION_FRAMES, frames=frames))
    # cmd4 + anchor_pos3 + anchor_ori6 + lin_vel3 + ang_vel3 + dof_pos n + dof_vel n + last n
    assert box["obs"].shape == (4 + 3 + 6 + 3 + 3 + 3 * n,)

    ctrl2 = BeyondMimicTrackerController(
        num_actions=n, default_pos=[0.0] * n, forward_fn=fn, without_state_estimator=True
    )
    ctrl2.step(_state(n), Reference(kind=ReferenceKind.MOTION_FRAMES, frames=frames))
    # drops anchor_pos(3) and lin_vel(3)
    assert box["obs"].shape == (4 + 6 + 3 + 3 * n,)


# ----- protomotions --------------------------------------------------------
def test_protomotions_obs_layout():
    n = 23
    box, fn = _capture()
    box["n"] = n
    task_obs = np.arange(30, dtype=np.float32)
    ctrl = ProtoMotionsTrackerController(
        num_actions=n, default_pos=[0.1] * n, forward_fn=fn, history_length=2
    )
    ctrl.reset()
    ctrl.step(
        _state(n),
        Reference(kind=ReferenceKind.MOTION_FRAMES, frames={"task_obs": task_obs}),
    )
    # dof_pos n + dof_vel n + ang_vel 3 + gravity 3 + task_obs 30 + last n + history(block*2)
    block = n + n + 3 + 3 + n
    assert box["obs"].shape == (n + n + 3 + 3 + 30 + n + block * 2,)


# ----- kungfubot (multi-input) ---------------------------------------------
def test_kungfubot_returns_three_inputs():
    n = 23
    inputs = {}

    def fn(obs):
        inputs["obs"] = obs
        return np.zeros(n, dtype=np.float32)

    frames = {
        "anchor_ref_rot": np.arange(6, dtype=np.float32),
        "next_step_ref_motion": np.arange(n, dtype=np.float32),
        "future_motion_base_lin_vel": np.zeros(3, dtype=np.float32),
        "future_motion_base_yaw_vel": np.zeros(1, dtype=np.float32),
        "future_motion_dof_pos": np.zeros(n, dtype=np.float32),
        "future_motion_roll_pitch": np.zeros(2, dtype=np.float32),
        "future_motion_root_height": np.zeros(1, dtype=np.float32),
    }
    ctrl = KungfuBotTrackerController(
        num_actions=n, default_pos=[0.0] * n, forward_fn=fn, history_length=2
    )
    ctrl.reset()
    ctrl.step(_state(n), Reference(kind=ReferenceKind.MOTION_FRAMES, frames=frames))
    obs = inputs["obs"]
    assert isinstance(obs, tuple) and len(obs) == 3
    actor_obs, future_targets, history = obs
    block = n + 3 + n + n + 2
    # actor: action n + anchor 6 + ang_vel 3 + dof_pos n + dof_vel n + history(block*2) + next n + rp 2
    assert actor_obs.shape == (n + 6 + 3 + n + n + block * 2 + n + 2,)
    # future targets: lin_vel 3 + yaw_vel 1 + dof_pos n + rp 2 + root_height 1
    assert future_targets.shape == (3 + 1 + n + 2 + 1,)
    assert history.shape == (block * 2,)


# ----- beyondmimic behavior ------------------------------------------------
_BM_NAMES = ("left_hip_pitch", "waist_yaw", "waist_roll", "waist_pitch", "right_elbow")


def _bm_motion(calls, n=5, frames=4):
    """Fake BeyondMimic export: reference joint_pos = time_step everywhere."""

    def motion_fn(obs, t):
        calls.append((np.asarray(obs).copy(), t))
        t = min(int(t), frames - 1)
        q = np.zeros((3, 4))
        q[:, 0] = 1.0  # [w, x, y, z] identity
        q[1] = [np.cos(0.25), 0, 0, np.sin(0.25)]  # anchor (index 1) yawed 0.5 rad
        return {"actions": np.full(n, 0.1 * (t + 1), np.float32), "joint_pos": np.full(n, float(t)),
                "joint_vel": np.full(n, -float(t)), "body_quat_w": q}

    return motion_fn


def _bm(calls, wose=True):
    return BeyondMimicTrackerController(
        num_actions=5, default_pos=[0.0] * 5, joint_names=_BM_NAMES,
        forward_fn=lambda o: np.zeros(5, np.float32), without_state_estimator=wose,
        motion_fn=_bm_motion(calls), motion_name="jump", motion_frames=4, anchor_body_index=1)


def test_beyondmimic_behavior_plays_baked_motion():
    calls = []
    ctrl = _bm(calls)
    (spec,) = ctrl.behaviors
    assert spec.name == "jump" and spec.start_pos == (0.0,) * 5
    src = ctrl.start_behavior("jump")
    # warm-up forward on a zero obs at frame 0 sets last_action
    assert calls[-1][1] == 0 and not calls[-1][0].any()
    np.testing.assert_allclose(ctrl._last_action, 0.1)
    state = ControllerState(dof_pos=np.zeros(5, np.float32), dof_vel=np.zeros(5, np.float32),
                            base_quat=np.array([0, 0, np.sin(-0.5), np.cos(-0.5)], np.float32),
                            base_ang_vel=np.zeros(3, np.float32))
    steps = []
    while not src.done:
        ref = src.next(state)
        ctrl.step(state, ref)
        steps.append((ref.frames["command"][0], calls[-1][1]))
    # the obs reads the previous forward's frame; the forward runs at the current one
    assert steps == [(0, 0), (0, 1), (1, 2), (2, 3)]
    # the anchor heading is turned onto the robot's: relative orientation is identity
    np.testing.assert_allclose(ref.frames["motion_anchor_ori_b"], [1, 0, 0, 1, 0, 0], atol=1e-6)
    assert ref.extra["time_step"] == 3


def test_beyondmimic_behavior_torso_from_waist_joints():
    ctrl = _bm([])
    state = ControllerState(dof_pos=np.array([0, 0.3, 0, 0, 0], np.float32), dof_vel=np.zeros(5, np.float32),
                            base_quat=np.array([0, 0, 0, 1], np.float32), base_ang_vel=np.zeros(3, np.float32))
    np.testing.assert_allclose(ctrl.torso_quat(state), [0, 0, np.sin(0.15), np.cos(0.15)], atol=1e-7)
    state.extra["torso_quat"] = np.array([1.0, 0, 0, 0])
    np.testing.assert_array_equal(ctrl.torso_quat(state), [1, 0, 0, 0])


def test_beyondmimic_behavior_with_state_estimator_needs_lin_vel():
    ctrl = _bm([], wose=False)
    src = ctrl.start_behavior("jump")
    state = _state(5)
    with pytest.raises(ValueError, match="base_lin_vel"):
        src.next(state)
    state.base_lin_vel = np.array([0.1, 0, 0], np.float32)
    ref = src.next(state)
    np.testing.assert_array_equal(ref.frames["motion_anchor_pos_b"], 0)
    ctrl.step(state, ref)


def test_beyondmimic_without_motion_fn_has_no_behavior():
    ctrl = BeyondMimicTrackerController(num_actions=5, default_pos=[0.0] * 5,
                                        forward_fn=lambda o: np.zeros(5, np.float32))
    assert ctrl.behaviors == []
    with pytest.raises(KeyError):
        ctrl.start_behavior("jump")

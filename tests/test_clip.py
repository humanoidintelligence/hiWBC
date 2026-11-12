"""Clip playback (``hiwbc.references.clip``) and the controllers' ``clip_reference``."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from hiwbc import ControllerState, ReferenceKind
from hiwbc.controllers.supermimic import SUPERMIMIC_JOINT_NAMES, SuperMimicController, goal_features
from hiwbc.controllers.twist import TwistTrackerController
from hiwbc.references import ClipPlayer


@dataclass
class Clip:  # same fields as hirobot.MotionClip, so these tests don't need hiRobot
    fps: float
    space: str = "joint"
    joint_names: list | None = None
    joint_pos: np.ndarray | None = None
    joint_vel: np.ndarray | None = None
    smpl_joints: np.ndarray | None = None
    root_pos: np.ndarray | None = None
    root_quat: np.ndarray | None = None
    extra: dict = field(default_factory=dict)


def _yaw(a):
    return np.array([0, 0, np.sin(a / 2), np.cos(a / 2)])


def _state(n, yaw=0.0):
    return ControllerState(dof_pos=np.zeros(n, np.float32), dof_vel=np.zeros(n, np.float32),
                           base_quat=_yaw(yaw).astype(np.float32), base_ang_vel=np.zeros(3, np.float32))


def _walk(T=50, fps=50.0, yaw0=0.4, names=("a", "b")):
    t = np.arange(T) / fps
    pos = np.stack([0.5 * t * np.cos(yaw0), 0.5 * t * np.sin(yaw0), 0.75 + 0 * t], 1) + [3.0, -2.0, 0]
    quat = np.stack([_yaw(yaw0 + 0.2 * ti) for ti in t])
    jp = np.stack([np.sin(t), np.cos(t)], 1)[:, : len(names)]
    return Clip(fps=fps, joint_names=list(names), joint_pos=jp, root_pos=pos, root_quat=quat)


# ----- the player ----------------------------------------------------------------
def test_same_rate_plays_frames_exactly_and_holds_the_last():
    clip = _walk(T=5)
    p = ClipPlayer(clip, 0.02)
    assert p.num_frames == 5 and p.duration == pytest.approx(0.08)
    seen = []
    while not p.done:
        seen.append(p.window(3, _yaw(0.4))["joint_pos"])
        p.advance()
    assert len(seen) == 5
    np.testing.assert_array_equal(seen[0], clip.joint_pos[:3])
    np.testing.assert_array_equal(seen[-1], clip.joint_pos[[4, 4, 4]])  # clamped to the last frame


def test_resamples_to_the_control_rate():
    clip = _walk(T=31, fps=30.0)  # 1 s
    p = ClipPlayer(clip, 0.02)
    assert p.num_frames == 51
    jp = p.window(51, _yaw(0.4))["joint_pos"]
    t = np.arange(51) * 0.02
    np.testing.assert_allclose(jp[:, 0], np.interp(t * 30, np.arange(31), clip.joint_pos[:, 0]))
    np.testing.assert_allclose(jp[-1], clip.joint_pos[-1])


def test_heading_and_position_aligned_to_the_robot_at_start():
    clip = _walk(yaw0=0.4)
    p = ClipPlayer(clip, 0.02)
    w = p.window(50, _yaw(-1.0))
    np.testing.assert_allclose(w["root_quat"][0], _yaw(-1.0), atol=1e-12)
    np.testing.assert_allclose(w["root_pos"][0], [0, 0, 0.75], atol=1e-12)
    # walks straight ahead of the robot's heading, at 0.5 m/s, turning at 0.2 rad/s
    np.testing.assert_allclose(w["root_lin_vel"][10], [0.5 * np.cos(-1.0), 0.5 * np.sin(-1.0), 0], atol=1e-9)
    np.testing.assert_allclose(w["root_ang_vel"][10], [0, 0, 0.2], atol=1e-9)
    np.testing.assert_allclose(w["root_z"], 0.75)
    # alignment is fixed at the first window; later robot headings don't move the clip
    p.advance()
    np.testing.assert_allclose(p.window(1, _yaw(2.0))["root_quat"][0], w["root_quat"][1], atol=1e-12)


def test_joint_velocity_differentiated_when_missing():
    clip = _walk()
    p = ClipPlayer(clip, 0.02)
    jv = p.window(50, _yaw(0))["joint_vel"]
    np.testing.assert_allclose(jv[1:-1, 0], np.cos(np.arange(1, 49) * 0.02), atol=2e-4)
    clip.joint_vel = np.full_like(clip.joint_pos, 7.0)
    np.testing.assert_array_equal(ClipPlayer(clip, 0.02).window(2, _yaw(0))["joint_vel"], 7.0)


def test_no_root_data():
    clip = Clip(fps=50, joint_names=["a"], joint_pos=np.zeros((4, 1)))
    w = ClipPlayer(clip, 0.02).window(2, _yaw(0.3))
    np.testing.assert_allclose(w["root_quat"], [_yaw(0.3)] * 2)
    assert w["root_pos"] is None and w["root_z"] is None
    np.testing.assert_array_equal(w["root_lin_vel"], 0)


def test_joints_reorders_by_name_and_fills_missing():
    clip = Clip(fps=50, joint_names=["b_joint", "a"], joint_pos=np.array([[2.0, 1.0]]))
    p = ClipPlayer(clip, 0.02)
    out = p.joints(p.window(1, _yaw(0))["joint_pos"], ["a", "b", "c"], fill=[0, 0, 9])
    np.testing.assert_array_equal(out, [[1, 2, 9]])  # "_joint" suffix stripped
    assert p.covers(["a", "c"]) == ["c"]


def test_append_continues_the_clip():
    p = ClipPlayer(_walk(T=10), 0.02)
    first = p.window(1, _yaw(0))
    p.append(_walk(T=10))
    assert p.num_frames == 20
    np.testing.assert_array_equal(p.window(1, _yaw(0))["root_quat"], first["root_quat"])
    with pytest.raises(ValueError, match="same space"):
        p.append(_walk(T=10, names=("a",)))


def test_rejects_bad_dt():
    with pytest.raises(ValueError):
        ClipPlayer(_walk(), 0.0)


# ----- controllers -----------------------------------------------------------------
def _twist():
    names = tuple(f"j{i}" for i in range(23))
    return TwistTrackerController(num_actions=23, default_pos=[0.1] * 23, joint_names=names,
                                  forward_fn=lambda obs: np.zeros(23, np.float32))


def test_base_controller_rejects_unsupported_space():
    from hiwbc.controllers.unitree import UnitreeLocoController
    ctrl = UnitreeLocoController(num_actions=12, default_pos=[0.0] * 12,
                                 forward_fn=lambda o: np.zeros(12, np.float32))
    assert ctrl.motion_spaces == ()
    with pytest.raises(ValueError, match="does not track 'joint'"):
        ctrl.check_clip(ClipPlayer(_walk(), 0.02))


def test_twist_mimic_target_from_clip():
    ctrl = _twist()
    names = list(ctrl.joint_names[:2])
    clip = _walk(names=names)
    p = ClipPlayer(clip, 0.02)
    ctrl.check_clip(p)
    ref = ctrl.clip_reference(p, _state(23, yaw=0.4))
    assert ref.kind is ReferenceKind.MOTION_FRAMES
    m = ref.frames["mimic_target"]
    assert m.shape == (31,)
    assert m[0] == pytest.approx(0.75)                          # root z
    np.testing.assert_allclose(m[1:4], [0, 0, 0.4], atol=1e-6)  # rpy (aligned heading = robot's)
    np.testing.assert_allclose(m[4:7], [0.5, 0, 0], atol=1e-3)  # body-frame velocity, forward
    assert m[7] == pytest.approx(0.2, abs=1e-3)                 # yaw rate
    np.testing.assert_allclose(m[8:10], clip.joint_pos[0])
    np.testing.assert_allclose(m[10:], 0.1)                     # joints the clip lacks: default pose
    ctrl.reset()
    ctrl.step(_state(23), ref)


def test_twist_needs_root_pose():
    ctrl = _twist()
    clip = Clip(fps=50, joint_names=["j0"], joint_pos=np.zeros((3, 1)))
    with pytest.raises(ValueError, match="root_pos"):
        ctrl.check_clip(ClipPlayer(clip, 0.02))
    with pytest.raises(ValueError, match="none of TWIST's joints"):
        ctrl.check_clip(ClipPlayer(_walk(names=("x",)), 0.02))


def test_supermimic_goal_from_clip_matches_goal_features():
    ctrl = SuperMimicController(forward_fn=lambda o: np.zeros(29, np.float32), default_pos=[0.05] * 29,
                                action_scale=[0.25] * 29, bound_torque=False)
    clip = _walk(T=30, names=SUPERMIMIC_JOINT_NAMES[:2])
    p = ClipPlayer(clip, 0.02)
    ctrl.check_clip(p)
    p.advance(25)  # near the end: the window repeats the last frame
    goal = ctrl.clip_reference(p, _state(29, yaw=2.0)).frames["goal"]
    w = p.window(10, None)
    jp = p.joints(w["joint_pos"], SUPERMIMIC_JOINT_NAMES, [0.05] * 29)
    jv = p.joints(w["joint_vel"], SUPERMIMIC_JOINT_NAMES, np.zeros(29))
    want = goal_features(jp, jv, w["root_quat"], w["root_lin_vel"], w["root_ang_vel"], w["root_z"],
                         [0.05] * 29)
    np.testing.assert_array_equal(goal, want)
    assert goal.shape == (10, 68)
    np.testing.assert_array_equal(goal[5:], goal[[4] * 5])
    # heading-frame velocity: the clip walks straight while turning 0.2 rad/s, 0.1 rad by frame 25
    np.testing.assert_allclose(goal[0, 64:67], [0.5 * np.cos(0.1), -0.5 * np.sin(0.1), 0], atol=1e-3)


def test_sonic_clip_reference():
    pytest.importorskip("onnxruntime")
    from hiwbc.configs.sonic.unitree_g1 import SONIC_G1, SONIC_VARIANTS
    from hiwbc.controllers.sonic import SonicController

    ctrl = SonicController(decoder_fn=lambda o: np.zeros(29, np.float32),
                           encoder_fn=lambda o: np.zeros(64, np.float32),
                           robot=SONIC_G1, obs=SONIC_VARIANTS["default"])
    assert ctrl.motion_spaces == ("joint", "smpl")
    names = list(SONIC_G1.joint_names)
    T = 60
    clip = _walk(T=T, names=names[:2])
    p = ClipPlayer(clip, 0.02)
    ctrl.check_clip(p)
    state = _state(29, yaw=0.4)
    ref = ctrl.clip_reference(p, state)
    assert ref.extra["encoder_mode"] == "joint"
    assert ref.frames["joint_pos"].shape == (46, 29) and ref.frames["root_z"].shape == (46,)
    np.testing.assert_allclose(ref.frames["joint_pos"][:, 2:], np.broadcast_to(SONIC_G1.default_pos[2:], (46, 27)))
    obs = ctrl.encoder_obs(state, ref)
    assert obs.shape == (ctrl.encoder_dim,) and np.isfinite(obs).all()
    # without an encoder SONIC takes no clips
    no_enc = SonicController(decoder_fn=lambda o: np.zeros(29, np.float32), robot=SONIC_G1,
                             obs=SONIC_VARIANTS["default"])
    assert no_enc.motion_spaces == ()
    # no mode of the default checkpoint reads the root height (its slot stays zero)
    clip.root_pos = None
    p = ClipPlayer(clip, 0.02)
    ctrl.check_clip(p)
    assert "root_z" not in ctrl.clip_reference(p, state).frames
    clip.root_quat = None
    with pytest.raises(ValueError, match="root_quat"):
        ctrl.check_clip(ClipPlayer(clip, 0.02))
    clip.root_quat = _walk(T=T).root_quat
    # SMPL clips
    smpl = Clip(fps=50, space="smpl", smpl_joints=np.zeros((T, 24, 3)), root_quat=clip.root_quat)
    p = ClipPlayer(smpl, 0.02)
    ctrl.check_clip(p)
    ref = ctrl.clip_reference(p, state)
    assert ref.extra["encoder_mode"] == "smpl" and ref.frames["smpl_joints"].shape == (10, 24, 3)
    assert np.isfinite(ctrl.encoder_obs(state, ref)).all()


# ----- source-rate velocities (PHC style) ------------------------------------------
def test_source_velocity_matches_scipy_gaussian():
    ndimage = pytest.importorskip("scipy.ndimage")
    clip = _walk(T=40, fps=30.0)
    clip.root_pos = clip.root_pos + np.stack([0 * clip.root_pos[:, 0], np.sin(np.arange(40) / 3.0),
                                              0 * clip.root_pos[:, 0]], 1)
    p = ClipPlayer(clip, 1 / 30, source_velocity_sigma=2.0)
    w = p.window(40, _yaw(0.4))  # robot at the clip's heading: no rotation applied
    want = ndimage.gaussian_filter1d(np.gradient(clip.root_pos, axis=0) * 30.0, 2, axis=0, mode="nearest")
    np.testing.assert_allclose(w["root_lin_vel"], want, atol=1e-9)
    np.testing.assert_allclose(w["root_ang_vel"][:-10, 2], 0.2, atol=1e-6)  # constant yaw rate; the
    # last frame is 0 (forward difference) and smoothing spreads that over 4 sigma
    # default unchanged: control-rate differences
    q = ClipPlayer(clip, 1 / 30).window(40, _yaw(0.4))
    np.testing.assert_allclose(q["root_lin_vel"], np.gradient(clip.root_pos, axis=0) * 30.0, atol=1e-9)
    with pytest.raises(ValueError, match="sigma"):
        ClipPlayer(clip, 0.02, source_velocity_sigma=-1.0)


def test_source_velocity_rotated_by_heading_alignment():
    clip = _walk(T=30, fps=50.0, yaw0=0.0)
    w = ClipPlayer(clip, 0.02, source_velocity_sigma=1.0).window(30, _yaw(np.pi / 2))
    np.testing.assert_allclose(w["root_lin_vel"][5:20], [[0, 0.5, 0]] * 15, atol=1e-6)


# ----- KungfuBot ------------------------------------------------------------------
class _ToyKin:
    """Stand-in for hirobot Kinematics: body "root" at the base, "ahead" 1 m in front."""

    class _Poses:
        names = ("root", "ahead")

        def __init__(self, pos, quat):
            self.pos, self.quat = pos, quat

        def index(self, name):
            return self.names.index(name)

    def forward(self, dof, base_pos, base_quat, joint_names):
        from hiwbc.controllers._bodyfk import rotate

        base_pos, base_quat = np.asarray(base_pos, float), np.asarray(base_quat, float)
        ahead = base_pos + rotate(base_quat, np.broadcast_to([1.0, 0, 0], base_pos.shape))
        return self._Poses(np.stack([base_pos, ahead], -2), np.stack([base_quat, base_quat], -2))


def _kungfubot(**kw):
    from hiwbc.controllers.kungfubot import KungfuBotTrackerController

    names = ("a", "b", "c")
    return KungfuBotTrackerController(
        num_actions=3, default_pos=[0.1] * 3, joint_names=names,
        forward_fn=lambda *o: np.zeros(3, np.float32), history_length=2, **kw)


def test_kungfubot_clip_reference_layout():
    ctrl = _kungfubot(future_steps=(1, 3), kinematics=lambda: _ToyKin(),
                      key_bodies=[("root", (0, 0, 0)), ("ahead", (0, 0, 0.5))])
    assert ctrl.motion_spaces == ("joint",)
    assert ctrl.clip_player_options == {"source_velocity_sigma": 2.0}
    clip = _walk(T=20, names=("a", "b"))
    p = ClipPlayer(clip, 0.02, **ctrl.clip_player_options)
    ctrl.check_clip(p)
    f = ctrl.clip_reference(p, _state(3, yaw=0.4)).frames
    assert f["future_motion_root_height"].shape == (2,)
    assert f["future_motion_dof_pos"].shape == (6,)
    np.testing.assert_allclose(f["future_motion_dof_pos"][:2], clip.joint_pos[1], atol=1e-6)
    assert f["future_motion_dof_pos"][2] == pytest.approx(0.1)          # missing joint: default
    np.testing.assert_allclose(f["future_motion_roll_pitch"], 0, atol=1e-6)
    np.testing.assert_allclose(f["future_motion_base_lin_vel"][:3], [0.5, 0, 0], atol=2e-3)
    np.testing.assert_allclose(f["future_motion_base_yaw_vel"], 0.2, atol=2e-3)
    nxt = f["next_step_ref_motion"]
    assert nxt.shape == (1 + 2 + 3 + 1 + 3 + 6,)
    np.testing.assert_allclose(nxt[-6:], [0, 0, 0, 1, 0, 0.5], atol=1e-6)  # key points, anchor frame
    # first step's anchor vs the robot: yaw 0.2 rad/s * 1 tick; columns 0-1 of the matrix, row-major
    c, s_ = np.cos(0.2 * 0.02), np.sin(0.2 * 0.02)
    np.testing.assert_allclose(f["anchor_ref_rot"], [c, -s_, s_, c, 0, 0], atol=1e-5)
    ctrl.reset()
    out = ctrl.observe(_state(3), ctrl.clip_reference(p, _state(3)))
    assert out[1].shape == (2 * 3 + 2 + 6 + 4 + 2,)


def test_kungfubot_check_clip():
    with pytest.raises(ValueError, match="future_steps, kinematics and key_bodies"):
        _kungfubot().check_clip(ClipPlayer(_walk(names=("a",)), 0.02))
    ctrl = _kungfubot(future_steps=(1,), kinematics=_ToyKin(), key_bodies=[("root", (0, 0, 0))])
    clip = _walk(names=("a",))
    clip.root_pos = None
    with pytest.raises(ValueError, match="root_pos"):
        ctrl.check_clip(ClipPlayer(clip, 0.02))


def test_kungfubot_profile_key_bodies_exist_in_g1():
    import os

    pytest.importorskip("hirobot.geometry.kinematics")
    from hirobot.geometry.kinematics import Kinematics

    from hiwbc.configs.kungfubot_tracker.unitree_g1 import (
        FUTURE_STEPS,
        G1_MJCF,
        JOINT_NAMES,
        KEY_BODIES,
    )

    assert FUTURE_STEPS == tuple(np.linspace(1, 95, 20).astype(int))
    path = os.path.expanduser(G1_MJCF)
    if not os.path.isfile(path):
        pytest.skip("G1 MJCF not present")
    kin = Kinematics.from_mjcf(path)
    assert {b for b, _ in KEY_BODIES} <= set(kin.body_names)
    assert {j + "_joint" for j in JOINT_NAMES} <= set(kin.joint_names)


# ----- derived terms / H2H --------------------------------------------------------
def test_add_derived_resamples_source_rate_values():
    clip = _walk(T=31, fps=30.0)
    p = ClipPlayer(clip, 0.02)
    p.add_derived("x", lambda f: f["root_pos"][:, :1] * 0 + np.arange(len(f["root_pos"]))[:, None])
    w = p.window(4, _yaw(0.4))
    np.testing.assert_allclose(w["x"][:, 0], [0, 0.6, 1.2, 1.8], atol=1e-12)  # source frame index
    with pytest.raises(ValueError, match="already exists"):
        p.add_derived("x", lambda f: f["root_pos"])
    p.append(_walk(T=31, fps=30.0))  # derived terms follow appended clips
    assert len(p.window(1, _yaw(0.4))["x"]) == 1 and p._derived["x"].shape[0] == p.num_frames


def test_start_places_clip_under_base_pos():
    clip = _walk(T=10)
    w = ClipPlayer(clip, 0.02).window(1, _yaw(0.4), base_pos=[2.0, -1.0, 0.7])
    np.testing.assert_allclose(w["root_pos"][0], [2.0, -1.0, clip.root_pos[0, 2]], atol=1e-9)


def _h2h(**kw):
    from hiwbc.controllers.protomotions import ProtoMotionsTrackerController

    return ProtoMotionsTrackerController(
        num_actions=3, default_pos=[0.1] * 3, joint_names=("a", "b", "c"),
        forward_fn=lambda o: np.zeros(3, np.float32), history_length=1, **kw)


def test_h2h_task_obs_from_clip():
    assert _h2h().motion_spaces == ()
    ctrl = _h2h(kinematics=_ToyKin(), track_points=[("ahead", (0, 0, 0)), ("root", (0, 0, 0.5))])
    assert ctrl.motion_spaces == ("joint",)
    clip = _walk(T=30, names=("a", "b"))
    p = ClipPlayer(clip, 0.02)
    ctrl.check_clip(p)
    st = _state(3, yaw=0.4)
    with pytest.raises(ValueError, match="base_pos"):
        ctrl.clip_reference(p, st)
    st.extra["base_pos"] = np.array([1.0, 2.0, 0.75])
    task = ctrl.clip_reference(p, st).frames["task_obs"]
    assert task.shape == (18,)
    # first frame: clip root under the robot, same heading, so the reference points
    # equal the robot's (diff 0) and sit 1 m ahead / 0.5 m above the base (z offset 0)
    np.testing.assert_allclose(task[:6], 0, atol=1e-6)
    np.testing.assert_allclose(task[6:12], [1, 0, 0, 0, 0, 0.5], atol=1e-6)
    np.testing.assert_allclose(task[12:15], [0.5, 0.2, 0], atol=3e-2)  # 0.5 m/s fwd + yaw * 1 m
    ctrl.reset()
    ctrl.step(st, ctrl.clip_reference(p, st))


def test_h2h_profile_points_exist_in_g1():
    import os

    pytest.importorskip("hirobot.geometry.kinematics")
    from hirobot.geometry.kinematics import Kinematics

    from hiwbc.configs.protomotions_tracker.unitree_g1 import (
        G1_29,
        G1_MJCF,
        JOINT_NAMES,
        TRACK_POINTS,
    )

    assert set(JOINT_NAMES) <= set(G1_29)
    path = os.path.expanduser(G1_MJCF)
    if not os.path.isfile(path):
        pytest.skip("G1 MJCF not present")
    kin = Kinematics.from_mjcf(path)
    assert {b for b, _ in TRACK_POINTS} <= set(kin.body_names)
    assert tuple(j.removesuffix("_joint") for j in kin.joint_names) == G1_29

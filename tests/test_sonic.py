"""SONIC controller: observation layout, history, token paths, action mapping (stub models)."""

import numpy as np
import pytest

from hiwbc import ControllerState, Reference, ReferenceKind, available_controllers
from hiwbc.configs.sonic.unitree_g1 import (
    G1_ACTION_SCALE,
    G1_DEFAULT_POS,
    G1_JOINT_NAMES,
    G1_JOINT_NAMES_MUJOCO,
    G1_KP,
    G1_LOWER_BODY_INDEX,
    ISAACLAB_TO_MUJOCO_INDEX,
    SONIC_G1,
    SONIC_VARIANTS,
)
from hiwbc.controllers.sonic import (
    VR_3POINT_DEFAULT_POS,
    VR_3POINT_DEFAULT_QUAT,
    VR_3POINT_NAMES,
    SonicController,
    SonicRobotConfig,
    _rot6d,
    vr_3point,
)
from hiwbc.controllers.sonic_planner import SonicPlanner

TD = 64
DEC_DIM = 64 + 10 * (3 + 29 + 29 + 29 + 3)  # 994


def _quat_z(yaw):  # [x, y, z, w]
    return np.array([0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)], dtype=np.float32)


def _state(k: float, yaw: float = 0.0) -> ControllerState:
    return ControllerState(
        dof_pos=(G1_DEFAULT_POS + k).astype(np.float32),
        dof_vel=np.full(29, 10 * k, np.float32),
        base_quat=_quat_z(yaw),
        base_ang_vel=np.full(3, 100 * k, np.float32),
    )


class _Rec:
    """Stub model that records its input and returns a fixed output."""

    def __init__(self, out):
        self.out = np.asarray(out, np.float32)
        self.inputs = []

    def __call__(self, obs):
        self.inputs.append(np.asarray(obs, np.float32).copy())
        return self.out


def _frames(T=60, yaw=0.0):
    t = np.arange(T, dtype=np.float32)[:, None]
    return {
        "joint_pos": np.tile(t, (1, 29)) + np.arange(29) * 1e-3,
        "joint_vel": -np.tile(t, (1, 29)),
        "root_quat": np.tile(_quat_z(yaw), (T, 1)),
        "root_z": 0.7 + 0.001 * t[:, 0],
    }


def _ctl(variant="default", act=None, tok=None):
    dec = _Rec(np.zeros(29) if act is None else act)
    enc = _Rec(np.arange(TD) * 0.01 if tok is None else tok)
    return SonicController(decoder_fn=dec, encoder_fn=enc, robot=SONIC_G1, obs=SONIC_VARIANTS[variant]), dec, enc


def _sonic(**kw):
    kw.setdefault("robot", SONIC_G1)
    kw.setdefault("obs", SONIC_VARIANTS["default"])
    return SonicController(**kw)


def test_registered():
    assert "sonic" in available_controllers()


@pytest.mark.parametrize("variant,enc_dim", [("default", 1762), ("low_latency", 1247), ("sonic_v1_1", 1751)])
def test_dims_match_released_onnx(variant, enc_dim):
    c, _, _ = _ctl(variant)
    assert c.encoder_dim == enc_dim
    assert c.decoder_dim == DEC_DIM


def test_joint_order_is_isaaclab():
    assert G1_JOINT_NAMES[:6] == (
        "left_hip_pitch", "right_hip_pitch", "waist_yaw",
        "left_hip_roll", "right_hip_roll", "waist_roll",
    )
    assert sorted(ISAACLAB_TO_MUJOCO_INDEX) == list(range(29))
    assert set(G1_JOINT_NAMES) == set(G1_JOINT_NAMES_MUJOCO)


def test_decoder_layout_and_history_order_with_zero_fill():
    tok = np.arange(TD, dtype=np.float32)
    c, dec, _ = _ctl()
    for k in (1.0, 2.0, 3.0):
        c.step(_state(k * 1e-3), Reference(kind=ReferenceKind.LATENT, value=tok))
    obs = dec.inputs[-1]
    assert obs.shape == (DEC_DIM,)
    np.testing.assert_array_equal(obs[:TD], tok)
    off = TD
    ang = obs[off:off + 30].reshape(10, 3)
    off += 30
    q = obs[off:off + 290].reshape(10, 29)
    off += 290
    dq = obs[off:off + 290].reshape(10, 29)
    off += 290 + 290
    grav = obs[off:off + 30].reshape(10, 3)
    # oldest first: 7 zero-padded entries, then ticks 1, 2, 3
    np.testing.assert_array_equal(ang[:7], 0)
    np.testing.assert_allclose(ang[7:, 0], [0.1, 0.2, 0.3], rtol=1e-5)
    np.testing.assert_allclose(q[7:, 0], [1e-3, 2e-3, 3e-3], rtol=1e-4)
    np.testing.assert_allclose(dq[9], 0.03, rtol=1e-5)
    # padded gravity is +z (zero quaternion in the deployment), live gravity is -z
    np.testing.assert_allclose(grav[:7], np.tile([0, 0, 1.0], (7, 1)))
    np.testing.assert_allclose(grav[7:], np.tile([0, 0, -1.0], (3, 1)), atol=1e-6)


def test_last_action_history_is_previous_raw_output_and_reset_clears():
    act = np.linspace(-1, 1, 29).astype(np.float32)
    c, dec, _ = _ctl(act=act)
    ref = Reference(kind=ReferenceKind.LATENT, value=np.zeros(TD, np.float32))
    c.step(_state(0.0), ref)
    c.step(_state(0.0), ref)
    acts = dec.inputs[-1][TD + 30 + 580:TD + 30 + 870].reshape(10, 29)
    np.testing.assert_array_equal(acts[9], act)  # logged before tick 2's inference
    np.testing.assert_array_equal(acts[8], 0)  # tick 1 saw no previous action
    c.reset()
    c.step(_state(0.0), ref)
    np.testing.assert_array_equal(dec.inputs[-1][TD + 30 + 580:TD + 30 + 870], 0)


def test_action_mapping_per_joint_scale_and_gains():
    act = np.linspace(-1, 1, 29).astype(np.float32)
    c, _, _ = _ctl(act=act)
    t = c.step(_state(0.0), Reference(kind=ReferenceKind.LATENT, value=np.zeros(TD, np.float32)))
    np.testing.assert_allclose(t.position, G1_DEFAULT_POS + act * G1_ACTION_SCALE, rtol=1e-6)
    np.testing.assert_array_equal(t.kp, G1_KP)
    # hip pitch (7520_22): 0.25 * 139 / (0.025101925 * (20 pi)^2) ~= 0.3507
    assert G1_ACTION_SCALE[0] == pytest.approx(0.25 * 139 / (0.025101925 * (20 * 3.1415926535) ** 2), rel=1e-6)
    assert G1_ACTION_SCALE[0] != G1_ACTION_SCALE[-1]


def test_latent_skips_encoder_and_motion_frames_runs_it():
    tok = np.full(TD, 0.5, np.float32)
    c, dec, enc = _ctl(tok=tok)
    c.step(_state(0.0), Reference(kind=ReferenceKind.LATENT, value=tok * 2))
    assert enc.inputs == []
    np.testing.assert_array_equal(dec.inputs[-1][:TD], tok * 2)
    t = c.step(_state(0.0), Reference(kind=ReferenceKind.MOTION_FRAMES, frames=_frames()))
    assert len(enc.inputs) == 1 and enc.inputs[0].shape == (1762,)
    np.testing.assert_array_equal(dec.inputs[-1][:TD], tok)
    np.testing.assert_array_equal(t.extra["motion_token"], tok)


def test_latent_and_motion_frames_agree_given_same_token():
    """Feeding the encoder's token back as LATENT gives the same decoder input and action."""
    rng = np.random.default_rng(0)
    W = rng.normal(size=(29, DEC_DIM)).astype(np.float32) * 0.01
    c1 = _sonic(decoder_fn=lambda o: W @ o, encoder_fn=lambda o: o[:TD] * 0 + o.sum() * 1e-4)
    c2 = _sonic(decoder_fn=lambda o: W @ o)
    for k in range(5):
        st = _state(k * 1e-3)
        t1 = c1.step(st, Reference(kind=ReferenceKind.MOTION_FRAMES, frames=_frames()))
        t2 = c2.step(st, Reference(kind=ReferenceKind.LATENT, value=t1.extra["motion_token"]))
        np.testing.assert_array_equal(t1.position, t2.position)


def test_g1_encoder_layout_default():
    c, _, enc = _ctl()
    fr = _frames()
    c.step(_state(0.0), Reference(kind=ReferenceKind.MOTION_FRAMES, frames=fr))
    x = enc.inputs[0]
    np.testing.assert_array_equal(x[:4], [0, 0, 0, 0])  # encoder_mode_4, joint = 0
    jp = x[4:294].reshape(10, 29)
    np.testing.assert_allclose(jp[:, 0], np.arange(10) * 5)  # step-5 sampling
    jv = x[294:584].reshape(10, 29)
    np.testing.assert_allclose(jv[:, 0], -np.arange(10) * 5)
    # root_z (10) + root_z (1) + anchor (6) are not used by the g1 mode -> zeros
    np.testing.assert_array_equal(x[584:601], 0)
    anchor = x[601:661].reshape(10, 6)
    np.testing.assert_allclose(anchor, np.tile([1, 0, 0, 1, 0, 0], (10, 1)), atol=1e-6)
    np.testing.assert_array_equal(x[661:], 0)  # teleop / smpl terms


def test_short_window_holds_last_frame_and_pause_zeros_velocity():
    c, _, enc = _ctl()
    fr = {k: v[:12] for k, v in _frames().items()}
    c.step(_state(0.0), Reference(kind=ReferenceKind.MOTION_FRAMES, frames=fr))
    jp = enc.inputs[-1][4:294].reshape(10, 29)
    np.testing.assert_allclose(jp[:, 0], [0, 5, 10, 11, 11, 11, 11, 11, 11, 11])
    c.step(_state(0.0), Reference(kind=ReferenceKind.MOTION_FRAMES, frames=fr, extra={"play": False}))
    x = enc.inputs[-1]
    np.testing.assert_array_equal(x[4:294].reshape(10, 29)[:, 0], 0)
    np.testing.assert_array_equal(x[294:584], 0)


def test_anchor_orientation_relative_to_base_and_heading():
    yaw_ref, yaw_base = 0.6, 0.2
    expect = _rot6d(_quat_z(yaw_ref - yaw_base))
    # default: full base quaternion on the left
    c, _, enc = _ctl()
    c.step(_state(0.0, yaw_base), Reference(kind=ReferenceKind.MOTION_FRAMES, frames=_frames(yaw=yaw_ref)))
    np.testing.assert_allclose(enc.inputs[0][601:607], expect, atol=1e-6)
    # v1.1: heading-only base quaternion; a base pitch must not change the term
    c, _, enc = _ctl("sonic_v1_1")
    st = _state(0.0, yaw_base)
    pitch = np.array([0.0, np.sin(0.1), 0.0, np.cos(0.1)], np.float32)
    from hiwbc.math_utils import quat_mul

    st.base_quat = quat_mul(_quat_z(yaw_base), pitch).astype(np.float32)
    c.step(st, Reference(kind=ReferenceKind.MOTION_FRAMES, frames=_frames(yaw=yaw_ref)))
    names = SONIC_VARIANTS["sonic_v1_1"].encoder_terms
    assert names[3] == "motion_anchor_orientation_heading_10frame_step5"
    np.testing.assert_allclose(enc.inputs[0][584:590], expect, atol=1e-6)


def test_bad_inputs_rejected():
    c, _, _ = _ctl()
    with pytest.raises(ValueError):
        c.step(_state(0.0), Reference(kind=ReferenceKind.LATENT, value=np.zeros(8)))
    with pytest.raises(KeyError):
        c.step(_state(0.0), Reference(kind=ReferenceKind.MOTION_FRAMES, frames={"joint_pos": np.zeros((5, 29))}))
    with pytest.raises(ValueError):
        c.step(_state(0.0), Reference(kind=ReferenceKind.VELOCITY, value=np.zeros(3)))
    with pytest.raises(ValueError):
        _sonic(decoder_fn=lambda o: np.zeros(29), encoder_mode="nope")
    nofn = _sonic(decoder_fn=lambda o: np.zeros(29))
    with pytest.raises(ValueError):
        nofn.step(_state(0.0), Reference(kind=ReferenceKind.MOTION_FRAMES, frames=_frames()))


def test_robot_config_is_injected_not_hardcoded():
    """A smaller robot sizes every joint term from its own config."""
    n = 12
    robot = SonicRobotConfig(
        joint_names=tuple(f"j{i}" for i in range(n)), default_pos=np.zeros(n),
        action_scale=np.ones(n), kp=np.full(n, 50.0), kd=np.ones(n),
        lower_body_index=np.arange(6), wrist_index=np.arange(6, 8),
    )
    c = _sonic(decoder_fn=lambda o: np.zeros(n), robot=robot)
    assert c.num_actions == n and c.joint_names == robot.joint_names
    assert c.decoder_dim == TD + 10 * (3 + 3 * n + 3)
    t = c.step(
        ControllerState(dof_pos=np.zeros(n, np.float32), dof_vel=np.zeros(n, np.float32),
                        base_quat=_quat_z(0.0), base_ang_vel=np.zeros(3, np.float32)),
        Reference(kind=ReferenceKind.LATENT, value=np.zeros(TD, np.float32)),
    )
    assert t.position.shape == (n,)
    np.testing.assert_array_equal(t.kp, robot.kp)


def test_checkpoint_using_wrist_terms_needs_wrists():
    robot = SonicRobotConfig(
        joint_names=SONIC_G1.joint_names, default_pos=SONIC_G1.default_pos,
        action_scale=SONIC_G1.action_scale, kp=SONIC_G1.kp, kd=SONIC_G1.kd,
        lower_body_index=SONIC_G1.lower_body_index,
    )
    with pytest.raises(ValueError, match="wrists"):
        _sonic(decoder_fn=lambda o: np.zeros(29), robot=robot)


def test_robot_config_validates_shapes():
    with pytest.raises(ValueError, match="kp"):
        SonicRobotConfig(joint_names=("a", "b"), default_pos=[0, 0], action_scale=[1, 1],
                         kp=[1.0], kd=[1, 1], lower_body_index=[0])
    with pytest.raises(ValueError, match="outside"):
        SonicRobotConfig(joint_names=("a", "b"), default_pos=[0, 0], action_scale=[1, 1],
                         kp=[1, 1], kd=[1, 1], lower_body_index=[2])


# ----- planner path (VELOCITY, CARTESIAN via teleop) -------------------------------


def _stub_planner():
    calls = []

    def fn(feeds):
        calls.append(feeds)
        q = np.zeros((1, 64, 36), np.float32)
        q[0, :, 2], q[0, :, 3] = 0.79, 1.0
        q[0, :, 7:] = 0.01 * len(calls)
        return q, np.array([30], np.int64)

    p = SonicPlanner(fn, qpos_joint_index=ISAACLAB_TO_MUJOCO_INDEX, lower_body_index=G1_LOWER_BODY_INDEX)
    return p, calls


def _planned(**kw):
    p, calls = _stub_planner()
    enc = _Rec(np.zeros(TD))
    c = _sonic(decoder_fn=_Rec(np.zeros(29)), encoder_fn=enc, planner=p,
               cartesian_targets={"left_hand": "left_wrist", "right_hand": "right_wrist"}, **kw)
    return c, enc, calls


def _vel(v=(0.0, 0.0, 0.0), **extra):
    return Reference(ReferenceKind.VELOCITY, value=np.asarray(v, np.float32), extra=extra)


def test_planner_adds_velocity_and_stands_through_it():
    c, _, _ = _planned()
    assert ReferenceKind.VELOCITY in c.accepts
    ref = c.stand_reference(_state(0.0))
    assert ref.kind == ReferenceKind.VELOCITY
    np.testing.assert_allclose(ref.value, 0)
    with pytest.raises(ValueError, match="encoder"):
        _sonic(decoder_fn=_Rec(np.zeros(29)), planner=_stub_planner()[0])
    nop = _sonic(decoder_fn=_Rec(np.zeros(29)), encoder_fn=_Rec(np.zeros(TD)))
    assert ReferenceKind.VELOCITY not in nop.accepts
    with pytest.raises(ValueError, match="velocity"):
        nop.step(_state(0.0), _vel())


def test_velocity_runs_the_joint_encoder_on_planner_frames():
    c, enc, calls = _planned()
    c.step(_state(0.0), _vel((0.5, 0, 0)))
    assert len(calls) == 1                         # start(): the first IDLE plan
    obs = enc.inputs[-1]
    assert obs[:4].tolist() == [0, 0, 0, 0]        # encoder_mode_4 = [mode id, 0, 0, 0]: joint
    # motion_joint_positions (10 frames) follow encoder_mode_4: the planner's joints
    np.testing.assert_allclose(obs[4:4 + 29], 0.01, atol=1e-6)
    for _ in range(5):
        c.step(_state(0.0), _vel((0.5, 0, 0)))
    assert len(calls) == 2 and calls[1]["mode"][0] == 1   # SLOW_WALK replan at the 10 Hz check


def test_leaving_the_planner_reinitializes_it():
    c, _, calls = _planned()
    c.step(_state(0.0), _vel())
    c.step(_state(0.0), Reference(ReferenceKind.LATENT, value=np.zeros(TD, np.float32)))
    c.step(_state(0.0), _vel())
    assert len(calls) == 2                         # a second start()
    c.reset()
    c.step(_state(0.0), _vel())
    assert len(calls) == 3


def test_ee_targets_switch_to_teleop_with_vr_points():
    c, enc, _ = _planned()
    pose = np.r_[0.3, 0.2, 0.1, 0, 0, 0, 1.0]
    c.step(_state(0.0), _vel(ee_targets={"left_wrist": pose}))
    obs = enc.inputs[-1]
    assert obs[:4].tolist() == [1, 0, 0, 0]        # teleop
    pos, quat = vr_3point({"left_wrist": pose})
    np.testing.assert_allclose(pos[0], [0.3, 0.2, 0.1])
    np.testing.assert_allclose(pos[1:], VR_3POINT_DEFAULT_POS[1:])   # untouched points: defaults
    np.testing.assert_allclose(quat[1:], VR_3POINT_DEFAULT_QUAT[1:])


def test_cartesian_rest_is_the_deployment_default():
    c, _, _ = _planned()
    rest = c.cartesian_rest()
    assert set(rest) == {"left_wrist", "right_wrist"}
    i = VR_3POINT_NAMES.index("right_wrist")
    np.testing.assert_allclose(rest["right_wrist"], np.r_[VR_3POINT_DEFAULT_POS[i], VR_3POINT_DEFAULT_QUAT[i]])


def test_vr_3point_rejects_bad_targets():
    with pytest.raises(ValueError, match="tracks"):
        vr_3point({"left_hand": np.zeros(7)})
    with pytest.raises(ValueError, match="7 finite"):
        vr_3point({"head": np.zeros(6)})
    with pytest.raises(ValueError, match="zero quaternion"):
        vr_3point({"head": np.zeros(7)})

"""BlinkMimic / XMimic tracker: input layout, goal features, runners, G1 constants (no weights)."""

import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from hiwbc import ControllerState, Reference, ReferenceKind, available_controllers
from hiwbc.controllers.blinkmimic import (
    FRAME_DIM,
    G1_ACTION_SCALE,
    G1_DEFAULT_POS,
    G1_EFFORT,
    G1_KD,
    G1_KP,
    GOAL_DIM,
    NUM_JOINTS,
    OBS_DIM,
    BlinkMimicController,
    PolicyRunner,
    bf16_round,
    goal_features,
    load_npz_runner,
    load_pt_runner,
    mat_to_6d,
    self_check,
)
from hiwbc.math_utils import quat_mul

N = NUM_JOINTS
IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])


def _ctl(fn, **kw):
    kw.setdefault("bound_torque", False)
    return BlinkMimicController(
        forward_fn=fn,
        default_pos=np.linspace(-0.3, 0.3, N),
        action_scale=np.full(N, 0.5),
        kp=np.full(N, 40.0),
        kd=np.full(N, 2.0),
        effort=np.full(N, 25.0),
        **kw,
    )


def _state(k: float, quat=IDENTITY) -> ControllerState:
    return ControllerState(
        dof_pos=np.full(N, k, dtype=np.float32),
        dof_vel=np.full(N, 10 * k, dtype=np.float32),
        base_quat=np.asarray(quat, dtype=np.float32),
        base_ang_vel=np.full(3, 100 * k, dtype=np.float32),
    )


def _ref(goal=None, quat=IDENTITY) -> Reference:
    goal = np.zeros((10, FRAME_DIM), np.float32) if goal is None else goal
    return Reference(ReferenceKind.MOTION_FRAMES, frames={"goal": goal, "ref_root_quat": quat})


def _spy(out=None):
    seen = []

    def fn(obs):
        seen.append(obs)
        return np.zeros(N, np.float32) if out is None else out

    return fn, seen


def _yaw(a):
    return np.array([0.0, 0.0, np.sin(a / 2), np.cos(a / 2)])


def _rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _rotx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def test_registered():
    assert "blinkmimic" in available_controllers()


# ----- observation entry and history ------------------------------------------------
def test_entry_order_by_value():
    fn, seen = _spy()
    ctl = _ctl(fn)
    ctl.step(_state(0.1), _ref())
    hist, goal = seen[0]
    assert hist.shape == (10, OBS_DIM) and goal.shape == (GOAL_DIM,)
    e = hist[-1]
    np.testing.assert_allclose(e[:N], 0.1 - ctl.default_pos, atol=1e-6)    # q - q0
    np.testing.assert_allclose(e[N:2 * N], 1.0, atol=1e-6)                 # qd
    np.testing.assert_allclose(e[2 * N:2 * N + 3], 10.0, atol=1e-5)        # base ang vel
    np.testing.assert_allclose(e[2 * N + 3:2 * N + 6], [0, 0, -1], atol=1e-6)  # gravity
    np.testing.assert_allclose(e[2 * N + 6:3 * N + 6], 0.0)                # a_prev after reset
    np.testing.assert_allclose(e[3 * N + 6:], [1, 0, 0, 1, 0, 0], atol=1e-6)  # 6D of identity


def test_gravity_is_minus_third_row_of_R():
    fn, seen = _spy()
    q = np.array([0.1, -0.2, 0.3, 0.9])
    q /= np.linalg.norm(q)
    _ctl(fn).step(_state(0.0, q), _ref())
    x, y, z, w = q
    R2 = [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]
    np.testing.assert_allclose(seen[0][0][-1, 2 * N + 3:2 * N + 6], -np.array(R2), atol=1e-6)


def test_6d_term_is_robot_relative_reference_orientation():
    fn, seen = _spy()
    # robot yawed 0.5 rad, reference yawed 0.2 rad and rolled 0.3 rad
    ref_R = _rotz(0.2) @ _rotx(0.3)
    ref_q = quat_mul(_yaw(0.2), np.array([np.sin(0.15), 0.0, 0.0, np.cos(0.15)]))
    _ctl(fn).step(_state(0.0, _yaw(0.5)), _ref(quat=ref_q))
    rel = _rotz(0.5).T @ ref_R
    want = [rel[0, 0], rel[0, 1], rel[1, 0], rel[1, 1], rel[2, 0], rel[2, 1]]
    np.testing.assert_allclose(seen[0][0][-1, 3 * N + 6:], want, atol=1e-6)
    np.testing.assert_allclose(mat_to_6d(rel), want)


def test_history_frame_major_oldest_first():
    fn, seen = _spy()
    ctl = _ctl(fn)
    for k in (0.1, 0.2, 0.3):
        ctl.step(_state(k), _ref())
    np.testing.assert_allclose(seen[0][0][:, N], [1.0] * 10, rtol=1e-6)  # first tick fills
    np.testing.assert_allclose(seen[-1][0][:, N], [1.0] * 8 + [2.0, 3.0], rtol=1e-6)
    ctl.reset()
    ctl.step(_state(0.5), _ref())
    np.testing.assert_allclose(seen[-1][0][:, N], [5.0] * 10, rtol=1e-6)


def test_prev_action_is_raw_and_target_is_q0_plus_scale():
    raw = np.linspace(-1, 1, N).astype(np.float32)
    fn, seen = _spy(raw)
    ctl = _ctl(fn)
    out = ctl.step(_state(0.0), _ref())
    np.testing.assert_allclose(out.position, ctl.default_pos + 0.5 * raw, atol=1e-6)
    ctl.step(_state(0.0), _ref())
    prev = seen[1][0][:, 2 * N + 6:3 * N + 6]
    np.testing.assert_allclose(prev[-1], raw, atol=1e-6)
    np.testing.assert_allclose(prev[0], 0.0)
    np.testing.assert_allclose(out.kp, 40.0)


def test_bound_torque_clips_pd_torque():
    ctl = _ctl(lambda obs: np.full(N, 100.0, np.float32), bound_torque=True)
    st = _state(0.0)
    out = ctl.step(st, _ref())
    tau = ctl.kp * (out.position - st.dof_pos) - ctl.kd * st.dof_vel
    np.testing.assert_allclose(tau, ctl.effort, rtol=1e-5)
    kp = np.full(N, 40.0)
    kp[3] = 0.0
    with pytest.raises(ValueError, match="kp > 0"):
        BlinkMimicController(forward_fn=lambda obs: np.zeros(N, np.float32),
                             default_pos=np.zeros(N), action_scale=np.full(N, 0.5), kp=kp,
                             kd=np.full(N, 2.0), effort=np.full(N, 25.0), bound_torque=True)


def test_bad_references_and_actions_rejected():
    ctl = _ctl(lambda obs: np.zeros(N, np.float32))
    with pytest.raises(ValueError):
        ctl.step(_state(0.0), _ref(np.zeros((10, 68), np.float32)))
    with pytest.raises(ValueError):
        ctl.step(_state(0.0), _ref(np.zeros(729, np.float32)))
    with pytest.raises(ValueError):
        ctl.step(_state(0.0), _ref(quat=np.zeros(4)))
    with pytest.raises(ValueError):
        ctl.step(_state(0.0), _ref(quat=np.ones(3)))
    with pytest.raises(ValueError):
        ctl.step(_state(0.0), Reference(ReferenceKind.MOTION_FRAMES,
                                        frames={"goal": np.zeros(GOAL_DIM)}))
    ctl.step(_state(0.0), _ref(np.zeros(GOAL_DIM, np.float32)))  # flat goal is accepted
    bad = _ctl(lambda obs: np.full(N, np.nan, np.float32))
    with pytest.raises(ValueError, match="non-finite"):
        bad.step(_state(0.0), _ref())
    with pytest.raises(NotImplementedError):
        ctl.stand_reference(_state(0.0))


# ----- goal features -------------------------------------------------------------------
def _turning_clip(T, dt=0.02):
    """Root moves along x at 1 m/s while yawing at 1 rad/s; joint j = 0.01 j t^2."""
    t = np.arange(T) * dt
    pos = np.stack([t, 0 * t, 0.75 + 0 * t], 1)
    quat = np.stack([_yaw(0.3 + ti) for ti in t])
    q = 0.01 * np.arange(N)[None, :] * t[:, None] ** 2
    return t, pos, quat, q


def test_goal_terms_by_value_mid_clip():
    t, pos, quat, q = _turning_clip(40)
    f = 5
    g = goal_features(pos[f:f + 12], quat[f:f + 12], q[f:f + 12])
    assert g.shape == (10, FRAME_DIM)
    for i, k in enumerate(range(f + 1, f + 11)):
        H = _rotz(0.3 + t[f])
        np.testing.assert_allclose(g[i, :3], H.T @ (pos[k] - pos[f]), atol=1e-6)      # dp
        rel = _rotz(t[k] - t[f])
        np.testing.assert_allclose(g[i, 3:9], rel[:, :2].reshape(-1), atol=1e-6)     # rot6d
        Rk = _rotz(0.3 + t[k])
        np.testing.assert_allclose(g[i, 9:12], Rk.T @ [1.0, 0, 0], atol=1e-5)        # v_local
        np.testing.assert_allclose(g[i, 12:15], [0, 0, 1.0], atol=1e-5)              # w_local
        np.testing.assert_allclose(g[i, 15:15 + N], q[k], atol=1e-6)                 # raw q
        # central difference of t^2 is exact: 2 t, times 0.1
        np.testing.assert_allclose(g[i, 15 + N:], 0.1 * 0.02 * np.arange(N) * t[k], atol=1e-6)


def test_goal_clamps_to_last_frame_with_one_sided_velocity():
    t, pos, quat, q = _turning_clip(10)
    f = 5  # frames 6..9 are real; 10..15 repeat frame 9
    g = goal_features(pos[f:], quat[f:], q[f:])
    np.testing.assert_array_equal(g[4:], np.repeat(g[3:4], 6, 0))
    np.testing.assert_allclose(g[3, 15:15 + N], q[9], atol=1e-6)
    # last frame: backward difference (q9 - q8) / dt; frame 8 still central
    np.testing.assert_allclose(g[3, 15 + N:], 0.1 * (q[9] - q[8]) / 0.02, atol=1e-6)
    np.testing.assert_allclose(g[2, 15 + N:], 0.1 * (q[9] - q[7]) / 0.04, atol=1e-6)
    np.testing.assert_allclose(g[3, 12:15], [0, 0, 1.0], atol=1e-5)
    np.testing.assert_allclose(g[3, :3], _rotz(0.3 + t[f]).T @ (pos[9] - pos[f]), atol=1e-6)


def test_goal_feature_input_checks():
    _, pos, quat, q = _turning_clip(20)
    with pytest.raises(ValueError):
        goal_features(pos[:1], quat[:1], q[:1])
    with pytest.raises(ValueError):
        goal_features(pos[:13], quat[:13], q[:13])


# ----- network runners -----------------------------------------------------------------
Z, HID = 8, 16


def _weights(seed=0):
    rng = np.random.default_rng(seed)

    def layers(sizes):
        return [(rng.normal(0, 0.3, (i, o)).astype(np.float32),
                 rng.normal(0, 0.1, o).astype(np.float32)) for i, o in itertools.pairwise(sizes)]

    return {
        "enc": layers([GOAL_DIM, HID, Z]),
        "dec": layers([Z + 10 * OBS_DIM, HID, N]),
        "obs_norm": (rng.normal(0, 0.1, OBS_DIM).astype(np.float32),
                     rng.uniform(0.5, 2, OBS_DIM).astype(np.float32)),
        "goal_norm": (rng.normal(0, 0.1, GOAL_DIM).astype(np.float32),
                      rng.uniform(0.5, 2, GOAL_DIM).astype(np.float32)),
    }


def _inputs(seed=1):
    rng = np.random.default_rng(seed)
    return (rng.normal(0, 0.5, (10, OBS_DIM)).astype(np.float32),
            rng.normal(0, 0.5, GOAL_DIM).astype(np.float32))


def _reference(wts, hist, goal, q0, alpha, rnd=lambda x: x):
    """The policy written out directly (upstream ``pi_apply``)."""
    def mlp(layers, x):
        for i, (w, b) in enumerate(layers):
            x = rnd(x) @ rnd(w) + b
            if i < len(layers) - 1:
                x = np.where(x > 0, x, np.exp(np.minimum(x, 0)) - 1)
        return x

    gm, gv = wts["goal_norm"]
    om, ov = wts["obs_norm"]
    z = np.tanh(mlp(wts["enc"], (goal - gm) / np.sqrt(gv + 1e-4)))
    o = ((hist - om) / np.sqrt(ov + 1e-4)).reshape(-1)
    a = mlp(wts["dec"], np.concatenate([z, o]))
    return a + (goal[15:44] - q0) / alpha


def _save_npz(path, wts, args=None, test=None):
    flat = {}
    for name in ("enc", "dec"):
        for i, (w, b) in enumerate(wts[name]):
            flat[f"{name}.{i}.w"], flat[f"{name}.{i}.b"] = w, b
    for name in ("obs_norm", "goal_norm"):
        flat[f"{name}.mean"], flat[f"{name}.var"] = wts[name]
        flat[f"{name}.count"] = np.float32(1e6)
    if test is not None:
        flat["test_oh"], flat["test_g"], flat["test_a"] = test
    meta = {"iteration": 1, "args": args or {"hist": 10, "goal_frames": 10}}
    np.savez(path, **flat, meta=json.dumps(meta))
    return path


def _npz(tmp_path, wts, **kw):
    return load_npz_runner(_save_npz(tmp_path / "ckpt.npz", wts), default_pos=G1_DEFAULT_POS,
                           action_scale=G1_ACTION_SCALE, **kw)


def test_npz_runner_fp32_matches_direct_reference(tmp_path):
    wts = _weights()
    run = _npz(tmp_path, wts, precision="fp32")
    hist, goal = _inputs()
    want = _reference(wts, hist, goal, G1_DEFAULT_POS, G1_ACTION_SCALE)
    np.testing.assert_allclose(run((hist, goal)), want, rtol=1e-4, atol=1e-5)
    assert not getattr(run, "stateful", False)


def test_ref_skip_adds_reference_joint_angles_of_next_frame(tmp_path):
    wts = _weights()
    run = _npz(tmp_path, wts, precision="fp32")
    hist, goal = _inputs()
    g2 = goal.copy()
    g2[15:44] += 0.1  # q of frame f+1; also changes z, so compare against a no-skip runner
    plain = PolicyRunner(wts["enc"], wts["dec"], wts["obs_norm"], wts["goal_norm"])
    np.testing.assert_allclose(run((hist, g2)) - plain((hist, g2)),
                               (g2[15:44] - G1_DEFAULT_POS) / G1_ACTION_SCALE, rtol=1e-5, atol=1e-5)


def test_bf16_rounding():
    x = np.array([1.0, 1.00390625, 1.005859375, 1.0078125 + 2 ** -9, -3.0e-3], np.float32)
    # 1 + 2^-8 is a tie between 1 and 1 + 2^-7: round to even (1.0)
    np.testing.assert_array_equal(bf16_round(x)[:2], [1.0, 1.0])
    assert bf16_round(x)[2] == np.float32(1.0078125)
    assert bf16_round(x)[3] == np.float32(1.0078125)   # tie at 1 + 2^-7 + 2^-8 -> even (1+2^-7)
    assert (bf16_round(x).view(np.uint32) & 0xFFFF == 0).all()


def test_npz_runner_bf16_matches_hand_rounded_matmul(tmp_path):
    wts = _weights()
    hist, goal = _inputs()
    bf = _npz(tmp_path, wts, precision="bf16")((hist, goal))
    fp = _npz(tmp_path, wts, precision="fp32")((hist, goal))
    want = _reference(wts, hist, goal, G1_DEFAULT_POS, G1_ACTION_SCALE, rnd=bf16_round)
    np.testing.assert_allclose(bf, want, rtol=1e-4, atol=1e-5)
    assert np.abs(bf - fp).max() > 1e-4


def test_npz_self_check(tmp_path):
    wts = _weights()
    hist, goal = _inputs()
    ref = PolicyRunner(wts["enc"], wts["dec"], wts["obs_norm"], wts["goal_norm"],
                       ref_q0=G1_DEFAULT_POS, ref_alpha=G1_ACTION_SCALE, precision="bf16")
    test = (hist[None], goal[None], ref((hist, goal))[None])
    path = _save_npz(tmp_path / "t.npz", wts, test=test)
    run = load_npz_runner(path, default_pos=G1_DEFAULT_POS, action_scale=G1_ACTION_SCALE)
    assert self_check(run) < 1e-6
    with pytest.raises(ValueError, match="no test vectors"):
        self_check(_npz(tmp_path, wts))


def test_npz_rejects_wrong_history_and_shapes(tmp_path):
    wts = _weights()
    path = _save_npz(tmp_path / "h.npz", wts, args={"hist": 5, "goal_frames": 10})
    with pytest.raises(ValueError, match="hist"):
        load_npz_runner(path, default_pos=G1_DEFAULT_POS, action_scale=G1_ACTION_SCALE)
    bad = dict(wts, enc=[(np.zeros((GOAL_DIM - 1, HID), np.float32), np.zeros(HID, np.float32)),
                         wts["enc"][1]])
    with pytest.raises(ValueError, match="encoder"):
        _npz(tmp_path, bad)
    with pytest.raises(ValueError, match="precision"):
        _npz(tmp_path, wts, precision="fp16")
    run = _npz(tmp_path, wts)
    with pytest.raises(ValueError):
        run((np.zeros((9, OBS_DIM)), np.zeros(GOAL_DIM)))


def _save_pt(path, wts, *, ref_skip=True, q0=G1_DEFAULT_POS, alpha=G1_ACTION_SCALE):
    torch = pytest.importorskip("torch")
    sd = {}
    for name in ("enc", "dec"):
        for i, (w, b) in enumerate(wts[name]):  # torch Linear: (out, in), at Sequential 0, 2, ..
            sd[f"{name}.{2 * i}.weight"] = torch.tensor(w.T.copy())
            sd[f"{name}.{2 * i}.bias"] = torch.tensor(b)
    for name in ("obs_norm", "goal_norm"):
        sd[f"{name}.mean"], sd[f"{name}.var"] = (torch.tensor(x) for x in wts[name])
        sd[f"{name}.count"] = torch.tensor(1e6)
    if ref_skip:
        sd["ref_q0"] = torch.tensor(np.asarray(q0, np.float32))
        sd["ref_alpha"] = torch.tensor(np.asarray(alpha, np.float32))
    args = {"hist": 10, "goal_frames": 10, "enc_hidden": str(HID), "dec_hidden": str(HID),
            "pi_ref_skip": ref_skip, "amp": "bf16"}
    torch.save({"pi": sd, "args": args, "iteration": 1}, path)
    return path


def test_pt_runner_equals_npz_runner(tmp_path):
    wts = _weights()
    pt = load_pt_runner(_save_pt(tmp_path / "p.pt", wts), default_pos=G1_DEFAULT_POS,
                        action_scale=G1_ACTION_SCALE)
    npz = _npz(tmp_path, wts, precision="fp32")
    hist, goal = _inputs()
    np.testing.assert_allclose(pt((hist, goal)), npz((hist, goal)), rtol=1e-6, atol=1e-6)


def test_pt_runner_without_ref_skip_and_mismatch(tmp_path):
    wts = _weights()
    hist, goal = _inputs()
    pt = load_pt_runner(_save_pt(tmp_path / "n.pt", wts, ref_skip=False))
    plain = PolicyRunner(wts["enc"], wts["dec"], wts["obs_norm"], wts["goal_norm"])
    np.testing.assert_allclose(pt((hist, goal)), plain((hist, goal)), rtol=1e-6, atol=1e-6)
    path = _save_pt(tmp_path / "m.pt", wts, q0=np.zeros(N))
    with pytest.raises(ValueError, match="default_pos"):
        load_pt_runner(path, default_pos=G1_DEFAULT_POS)


# ----- G1 constants -----------------------------------------------------------------
def test_g1_constants_pinned():
    # actuator_gainprm / biasprm and action_scale from XMimic's G1 meta.json (MJCF exported
    # from BlinkMimic's mjlab scene), and its default_joint_pos.
    hip_pitch, hip_roll, ankle, shoulder, wrist_pitch = 0, 1, 4, 15, 20
    np.testing.assert_allclose(G1_KP[[hip_pitch, hip_roll, ankle, shoulder, wrist_pitch]],
                               [40.17923847137318, 99.09842777666113, 28.50124619574858,
                                14.25062309787429, 16.77832748089279], rtol=1e-12)
    np.testing.assert_allclose(G1_KD[[hip_pitch, hip_roll, ankle, shoulder]],
                               [2.5578897650279457, 6.3088018534966395, 1.814445686584846,
                                0.907222843292423], rtol=1e-12)
    np.testing.assert_allclose(G1_ACTION_SCALE[[0, 1, 4, 13, 20, 28]],
                               [0.5475464652142303, 0.3506614663788243, 0.43857731392336724,
                                0.43857731392336724, 0.07450087032950714, 0.07450087032950714],
                               rtol=1e-12)
    np.testing.assert_allclose(G1_EFFORT[[0, 1, 3, 4, 12, 13, 15, 20]],
                               [88, 139, 139, 50, 88, 50, 25, 5])
    np.testing.assert_allclose(G1_DEFAULT_POS, [
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0, -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        0.0, 0.0, 0.0, 0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0, 0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0])


# ----- real checkpoints, when present on this machine --------------------------------
_REAL_NPZ = Path("~/.cache/hiwbc/xmimic/policy.npz").expanduser()
_REAL_PT = Path("~/.cache/hiwbc/blinkmimic/policy.pt").expanduser()


@pytest.mark.skipif(not _REAL_NPZ.is_file(), reason="no XMimic checkpoint")
def test_real_xmimic_checkpoint_self_check():
    run = load_npz_runner(_REAL_NPZ, default_pos=G1_DEFAULT_POS, action_scale=G1_ACTION_SCALE)
    if run.test_vectors is None:
        pytest.skip("checkpoint has no test vectors")
    assert self_check(run) < 1e-4


@pytest.mark.skipif(not _REAL_PT.is_file(), reason="no BlinkMimic checkpoint")
def test_real_blinkmimic_checkpoint_loads():
    pytest.importorskip("torch")
    run = load_pt_runner(_REAL_PT, default_pos=G1_DEFAULT_POS, action_scale=G1_ACTION_SCALE)
    a = run(_inputs())
    assert a.shape == (N,) and np.isfinite(a).all()


# ----- clip playback -------------------------------------------------------------------
def test_clip_reference_matches_goal_features_and_stops_before_last_frame():
    from dataclasses import dataclass, field

    from hiwbc.controllers.blinkmimic import BLINKMIMIC_JOINT_NAMES
    from hiwbc.references import ClipPlayer

    @dataclass
    class Clip:
        fps: float
        space: str = "joint"
        joint_names: list | None = None
        joint_pos: np.ndarray | None = None
        joint_vel: np.ndarray | None = None
        smpl_joints: np.ndarray | None = None
        root_pos: np.ndarray | None = None
        root_quat: np.ndarray | None = None
        extra: dict = field(default_factory=dict)

    _, pos, quat, q = _turning_clip(15)
    clip = Clip(fps=50.0, joint_names=list(BLINKMIMIC_JOINT_NAMES), joint_pos=q, root_pos=pos,
                root_quat=quat)
    ctl = _ctl(lambda obs: np.zeros(N, np.float32))
    p = ClipPlayer(clip, 0.02)
    ctl.check_clip(p)
    # robot heading 0.3 = the clip's first heading, at the origin: aligned clip = clip - p0
    ref = ctl.clip_reference(p, _state(0.0, _yaw(0.3)))
    want = goal_features(pos[:12] - pos[0] * [1, 1, 0], quat[:12], q[:12])
    np.testing.assert_allclose(ref.frames["goal"], want, atol=1e-5)
    np.testing.assert_allclose(ref.frames["ref_root_quat"], quat[0], atol=1e-6)
    p.advance(14)  # the last frame: the goal is the one of frame 13
    last = ctl.clip_reference(p, _state(0.0, _yaw(0.3)))
    want = goal_features(pos[13:] - pos[0] * [1, 1, 0], quat[13:], q[13:])
    np.testing.assert_allclose(last.frames["goal"], want, atol=1e-5)
    np.testing.assert_allclose(last.frames["ref_root_quat"], quat[13], atol=1e-6)
    with pytest.raises(ValueError, match="root_pos"):
        ctl.check_clip(ClipPlayer(Clip(fps=50.0, joint_names=list(BLINKMIMIC_JOINT_NAMES),
                                       joint_pos=q, root_quat=quat), 0.02))

    # A new clip starts like an upstream episode: history refilled, previous action 0.
    raw = np.full(N, 0.7, np.float32)
    fn, seen = _spy(raw)
    ctl = _ctl(fn)
    st = _state(0.0, _yaw(0.3))
    first = ClipPlayer(clip, 0.02)
    ctl.step(st, ctl.clip_reference(first, st))
    first.advance(1)
    ctl.step(st, ctl.clip_reference(first, st))  # same player: a_prev carries over
    np.testing.assert_allclose(seen[1][0][-1, 2 * N + 6:3 * N + 6], raw, atol=1e-6)
    second = ClipPlayer(clip, 0.02)
    ctl.step(st, ctl.clip_reference(second, st))
    hist = seen[2][0]
    np.testing.assert_allclose(hist[:, 2 * N + 6:3 * N + 6], 0.0)
    np.testing.assert_array_equal(hist, np.broadcast_to(hist[-1], hist.shape))

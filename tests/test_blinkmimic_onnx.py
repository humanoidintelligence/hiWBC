"""BlinkMimic single-motion ONNX export (``blinkmimic/unitree_g1`` ``onnx``): the behavior.

A tiny ONNX with the export's inputs, outputs and metadata stands in for a real
export, so the observation, ``time_step`` and anchor-quaternion handling are
pinned by value without weights.
"""

import os
from pathlib import Path

import numpy as np
import pytest

import hiwbc
from hiwbc import ControllerState, Reference, ReferenceKind
from hiwbc.controllers.blinkmimic import (
    BLINKMIMIC_JOINT_NAMES,
    G1_ACTION_SCALE,
    G1_DEFAULT_POS,
    G1_KD,
    G1_KP,
    NUM_JOINTS,
    OBS_DIM,
    BlinkMimicController,
    load_onnx_motion,
    mat_to_6d,
)

N = NUM_JOINTS
FRAMES = 6
Q6 = slice(3 * N + 6, 3 * N + 12)       # the 6D term
APREV = slice(2 * N + 6, 3 * N + 6)     # previous action


def _yaw_wxyz(a):
    return np.array([np.cos(a / 2), 0.0, 0.0, np.sin(a / 2)])


def _rot_wxyz(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


#: The fake motion's root orientation per frame, [w, x, y, z]: yaw 0.1 f plus a growing
#: roll component, so the 6D term is not yaw only. Frame 0 is identity (heading 0).
ROOT = np.stack([_yaw_wxyz(0.1 * f) for f in range(FRAMES)])
ROOT[:, 1] = 0.02 * np.arange(FRAMES)
ROOT /= np.linalg.norm(ROOT, axis=1, keepdims=True)


def _fmt(xs):
    return ",".join(f"{float(x):.6g}" for x in xs)


def _metadata(**over):
    meta = {
        "motion": "fake_dance",
        "motion_num_frames": str(FRAMES),
        "joint_names": ",".join(f"{n}_joint" for n in BLINKMIMIC_JOINT_NAMES),
        "joint_stiffness": _fmt(G1_KP),
        "joint_damping": _fmt(G1_KD),
        "default_joint_pos": _fmt(G1_DEFAULT_POS),
        "action_scale": _fmt(G1_ACTION_SCALE),
        "command_names": "motion",
        "observation_names": ("joint_pos,joint_vel,base_ang_vel,projected_gravity,actions,"
                              "motion_anchor_ori_b"),
        "observation_history": "10",
        "goal_frames": "10",
        "control_dt": "0.02",
        "anchor_body_name": "pelvis",
        "body_names": "pelvis,torso_link",
        "anchor_quat_w_init": _fmt(ROOT[0]),
    }
    meta.update(over)
    return {k: v for k, v in meta.items() if v is not None}


def _export(path, **meta_over):
    """Fake export. ``actions`` = newest entry's ``q - q0`` + time_step;
    ``joint_pos`` = 0.01 * time_step; ``next_anchor_quat_w`` = ROOT[min(t + 1, last)]."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    def const(name, arr, dtype=TensorProto.INT64):
        arr = np.asarray(arr)
        return helper.make_node("Constant", [], [name], value=helper.make_tensor(
            name + "_v", dtype, list(arr.shape), arr.ravel().tolist()))

    nodes = [
        const("shape_flat", [1, 10 * OBS_DIM]),
        helper.make_node("Reshape", ["obs_history", "shape_flat"], ["flat"]),
        const("s", [9 * OBS_DIM]), const("e", [9 * OBS_DIM + N]), const("ax", [1]),
        helper.make_node("Slice", ["flat", "s", "e", "ax"], ["newest_q"]),
        helper.make_node("Add", ["newest_q", "time_step"], ["actions"]),
        const("ones", np.ones((1, N)), TensorProto.FLOAT),
        const("c01", [0.01], TensorProto.FLOAT),
        helper.make_node("Mul", ["time_step", "c01"], ["t01"]),
        helper.make_node("Mul", ["ones", "t01"], ["joint_pos"]),
        helper.make_node("Round", ["time_step"], ["tr"]),
        helper.make_node("Cast", ["tr"], ["ti"], to=TensorProto.INT64),
        const("one", [1]), const("lo", [0]), const("hi", [FRAMES - 1]),
        helper.make_node("Add", ["ti", "one"], ["t1"]),
        helper.make_node("Clip", ["t1", "lo", "hi"], ["t1c"]),
        const("shape1", [1]),
        helper.make_node("Reshape", ["t1c", "shape1"], ["idx"]),
        const("table", ROOT.astype(np.float32), TensorProto.FLOAT),
        helper.make_node("Gather", ["table", "idx"], ["next_anchor_quat_w"], axis=0),
    ]
    f = TensorProto.FLOAT
    graph = helper.make_graph(
        nodes, "fake_blinkmimic",
        [helper.make_tensor_value_info("obs_history", f, [1, 10, OBS_DIM]),
         helper.make_tensor_value_info("time_step", f, [1, 1])],
        [helper.make_tensor_value_info("actions", f, [1, N]),
         helper.make_tensor_value_info("joint_pos", f, [1, N]),
         helper.make_tensor_value_info("next_anchor_quat_w", f, [1, 4])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8   # what older onnxruntime builds read
    for k, v in _metadata(**meta_over).items():
        model.metadata_props.add(key=k, value=v)
    onnx.save(model, str(path))
    return path


@pytest.fixture
def export(tmp_path):
    pytest.importorskip("onnxruntime")
    return _export(tmp_path / "fake_dance.onnx")


def _load(path, **kw):
    return hiwbc.load("blinkmimic/unitree_g1", variant="onnx", checkpoint=str(path),
                      bound_torque=False, **kw)


def _state(k, quat_xyzw=(0.0, 0.0, 0.0, 1.0)):
    return ControllerState(
        dof_pos=(G1_DEFAULT_POS + k).astype(np.float32),
        dof_vel=np.zeros(N, np.float32),
        base_quat=np.asarray(quat_xyzw, np.float32),
        base_ang_vel=np.zeros(3, np.float32),
    )


def _spy(ctrl):
    """Record each (obs_history, time_step) the motion model gets."""
    seen, fn = [], ctrl._motion_fn

    def wrapped(obs, t):
        seen.append((np.asarray(obs).copy(), t))
        return fn(obs, t)

    ctrl._motion_fn = wrapped
    return seen


# ----- loading -------------------------------------------------------------------------
def test_load_onnx_motion_parses_metadata(export):
    fn, meta = load_onnx_motion(export)
    assert meta["joint_names"] == tuple(BLINKMIMIC_JOINT_NAMES)
    assert meta["frames"] == FRAMES and meta["motion"] == "fake_dance"
    np.testing.assert_allclose(meta["anchor_quat_init"], ROOT[0], atol=1e-6)
    np.testing.assert_allclose(meta["kp"], G1_KP, rtol=1e-5)
    out = fn(np.zeros((10, OBS_DIM)), 2)
    np.testing.assert_allclose(out["actions"], 2.0)
    np.testing.assert_allclose(out["next_anchor_quat_w"], ROOT[3], atol=1e-6)


def test_profile_builds_behavior_named_after_motion(export, tmp_path):
    ctrl = _load(export)
    (spec,) = ctrl.behaviors
    assert spec.name == "fake_dance"
    assert spec.start_pos == pytest.approx([0.0] * N)   # joint_pos at time_step 0
    assert ctrl.joint_names == tuple(BLINKMIMIC_JOINT_NAMES)
    np.testing.assert_allclose(ctrl.kp, G1_KP, rtol=1e-6)
    # no motion field: the behavior is the file stem
    p = _export(tmp_path / "my_clip.onnx", motion=None)
    assert [b.name for b in _load(p).behaviors] == ["my_clip"]


@pytest.mark.parametrize("over, match", [
    ({"joint_stiffness": _fmt(G1_KP * (1 + 1e-4))}, "kp"),
    ({"action_scale": _fmt(G1_ACTION_SCALE[::-1])}, "action_scale"),
    ({"joint_names": ",".join(BLINKMIMIC_JOINT_NAMES[::-1])}, "joint_names"),
    ({"observation_history": "5"}, "observation_history"),
    ({"control_dt": "0.01"}, "control_dt"),
    ({"anchor_body_name": "torso_link"}, "anchor"),
    ({"anchor_quat_w_init": None}, "metadata lacks"),
])
def test_profile_rejects_metadata_that_differs_from_g1(tmp_path, over, match):
    pytest.importorskip("onnxruntime")
    p = _export(tmp_path / "bad.onnx", **over)
    with pytest.raises(ValueError, match=match):
        _load(p)


def test_export_only_plays_its_behavior(export):
    ctrl = _load(export)
    ref = Reference(ReferenceKind.MOTION_FRAMES,
                    frames={"goal": np.zeros(730), "ref_root_quat": np.array([0, 0, 0, 1.0])})
    with pytest.raises(ValueError, match="only plays"):
        ctrl.step(_state(0.0), ref)
    with pytest.raises(ValueError, match="only plays"):
        ctrl.check_clip(None)
    with pytest.raises(KeyError):
        ctrl.start_behavior("other")
    with pytest.raises(TypeError):
        ctrl.start_behavior("fake_dance", speed=1)


# ----- the behavior: time_step, anchor quaternion, a_prev, history ---------------------
def test_behavior_time_steps_anchor_quat_and_end(export):
    ctrl = _load(export)
    seen = _spy(ctrl)
    src = ctrl.start_behavior("fake_dance")
    ticks = 0
    while not src.done:
        ctrl.step(_state(0.01 * ticks, (0, 0, 0, 1.0)), src.next(_state(0.01 * ticks)))
        ticks += 1
    # play_onnx: time_step 0 .. frames - 2, one per tick; the last frame is never current
    assert ticks == FRAMES - 1
    assert [t for _, t in seen] == list(range(FRAMES - 1))
    # the robot base is at identity and the motion starts at yaw 0: no heading turn, so
    # tick t's 6D term is against ROOT[t] (init at t = 0, then the previous call's
    # next_anchor_quat_w = ROOT[t])
    for t, (hist, _) in enumerate(seen):
        np.testing.assert_allclose(hist[-1, Q6], mat_to_6d(_rot_wxyz(ROOT[t])), atol=1e-6)


def test_behavior_a_prev_and_history(export):
    ctrl = _load(export)
    seen = _spy(ctrl)
    src = ctrl.start_behavior("fake_dance")
    for k in range(3):
        ctrl.step(_state(0.1 * (k + 1)), src.next(_state(0.0)))
    h0, h1, h2 = (h for h, _ in seen)
    assert h0.shape == (10, OBS_DIM)
    # tick 0: a_prev zero, history = 10 copies of the first entry
    np.testing.assert_allclose(h0, np.repeat(h0[-1:], 10, 0))
    np.testing.assert_allclose(h0[-1, APREV], 0.0)
    np.testing.assert_allclose(h0[-1, :N], 0.1, atol=1e-6)
    # tick 1: a_prev = tick 0's raw action (q - q0 + time_step = 0.1 + 0); oldest first
    np.testing.assert_allclose(h1[-1, APREV], 0.1, atol=1e-6)
    np.testing.assert_allclose(h1[:-1], h0[1:])
    np.testing.assert_allclose(h2[-1, APREV], 0.2 + 1.0, atol=1e-6)
    np.testing.assert_allclose(h2[-2], h1[-1])
    # the target is q0 + action_scale * a
    out = ctrl.step(_state(0.0), src.next(_state(0.0)))
    np.testing.assert_allclose(out.position, G1_DEFAULT_POS + G1_ACTION_SCALE * 3.0, atol=1e-5)


def test_behavior_aligns_motion_heading_to_robot(export):
    ctrl = _load(export)
    seen = _spy(ctrl)
    src = ctrl.start_behavior("fake_dance")
    robot_yaw = 0.7    # motion frame 0 has yaw 0: the motion is turned by 0.7 rad
    base = np.array([0.0, 0.0, np.sin(robot_yaw / 2), np.cos(robot_yaw / 2)])
    for _ in range(3):
        ctrl.step(_state(0.0, base), src.next(_state(0.0, base)))
    for t, (hist, _) in enumerate(seen):
        # R_base^T (turn R_ref(t)), turn = yaw(robot) - yaw(ROOT[0]) = 0.7
        want = _rotz(robot_yaw).T @ (_rotz(robot_yaw) @ _rot_wxyz(ROOT[t]))
        np.testing.assert_allclose(hist[-1, Q6], mat_to_6d(want), atol=1e-6)
        # unaligned, the term would carry the 0.7 rad heading error
        assert np.abs(hist[-1, Q6] - mat_to_6d(_rotz(robot_yaw).T @ _rot_wxyz(ROOT[t]))).max() > 0.1


def test_behavior_alignment_uses_the_motion_start_heading(tmp_path):
    pytest.importorskip("onnxruntime")
    root = ROOT.copy()
    turned = np.array([_quat_mul_wxyz(_yaw_wxyz(-0.4), q) for q in root])
    p = _export(tmp_path / "turned.onnx", anchor_quat_w_init=_fmt(turned[0]))
    ctrl = _load(p)
    seen = _spy(ctrl)
    src = ctrl.start_behavior("fake_dance")
    ctrl.step(_state(0.0), src.next(_state(0.0)))   # robot at heading 0
    # tick 0 uses anchor_quat_w_init (heading -0.4), turned by +0.4 onto the robot
    want = _rotz(0.4) @ _rot_wxyz(turned[0])
    np.testing.assert_allclose(seen[0][0][-1, Q6], mat_to_6d(want), atol=1e-6)


def _quat_mul_wxyz(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw])


def test_failed_tick_does_not_advance_and_restart_resets(export):
    ctrl = _load(export)
    seen = _spy(ctrl)
    src = ctrl.start_behavior("fake_dance")
    ctrl.step(_state(0.0), src.next(_state(0.0)))
    fn = ctrl._motion_fn

    def boom(obs, t):
        raise TimeoutError("late")

    ctrl._motion_fn = boom
    with pytest.raises(TimeoutError):
        ctrl.step(_state(0.0), src.next(_state(0.0)))
    ctrl._motion_fn = fn
    ctrl.step(_state(0.0), src.next(_state(0.0)))
    assert [t for _, t in seen] == [0, 1]
    # starting again plays from frame 0 with a fresh history
    src = ctrl.start_behavior("fake_dance")
    ctrl.step(_state(0.0), src.next(_state(0.0)))
    assert seen[-1][1] == 0
    np.testing.assert_allclose(seen[-1][0][-1, APREV], 0.0)


def test_motion_fn_needs_its_metadata():
    with pytest.raises(ValueError, match="needs"):
        BlinkMimicController(default_pos=G1_DEFAULT_POS, action_scale=G1_ACTION_SCALE,
                             bound_torque=False, motion_fn=lambda o, t: {})
    with pytest.raises(ValueError, match="forward_fn"):
        BlinkMimicController(default_pos=G1_DEFAULT_POS, action_scale=G1_ACTION_SCALE,
                             bound_torque=False)


# ----- the real export, when present on this machine -----------------------------------
_REAL = Path("~/.cache/hiwbc/blinkmimic/dance1_subject2.onnx").expanduser()
#: An optional recorded closed-loop run of that export (obs, time_step, actions, q, qd,
#: root_quat_wxyz, root_ang_vel_w, next_anchor_quat_w per tick).
_TRACE = os.environ.get("HIWBC_BLINKMIMIC_ONNX_TRACE")


@pytest.mark.skipif(not _REAL.is_file(), reason="no BlinkMimic ONNX export")
def test_real_export_loads_and_plays():
    pytest.importorskip("onnxruntime")
    ctrl = hiwbc.load("blinkmimic/unitree_g1", variant="onnx")
    (spec,) = ctrl.behaviors
    assert spec.name == "dance1_subject2" and len(spec.start_pos) == N
    src = ctrl.start_behavior(spec.name)
    for _ in range(3):
        out = ctrl.step(_state(0.0), src.next(_state(0.0)))
        assert np.isfinite(out.position).all()


def replay_trace(trace, ticks=None):
    """Drive the real export's behavior with a recorded run's robot states.

    Returns the largest differences of the observation, the action and the anchor
    quaternion from the recorded ones.
    """
    ctrl = hiwbc.load("blinkmimic/unitree_g1", variant="onnx", bound_torque=False)
    seen = _spy(ctrl)
    outs = []
    fn = ctrl._motion_fn

    def keep(obs, t):
        out = fn(obs, t)
        outs.append(out)
        return out

    ctrl._motion_fn = keep
    src = ctrl.start_behavior(ctrl.motion_name)
    n = len(trace["time_step"]) if ticks is None else ticks
    for i in range(n):
        qw = trace["root_quat_wxyz"][i].astype(np.float64)
        R = _rot_wxyz(qw)
        state = ControllerState(
            dof_pos=trace["q"][i], dof_vel=trace["qd"][i], base_quat=qw[[1, 2, 3, 0]],
            base_ang_vel=(R.T @ trace["root_ang_vel_w"][i]).astype(np.float32))
        ctrl.step(state, src.next(state))
    obs = np.stack([h[-1] for h, _ in seen])
    return {
        "ticks": n, "done": src.done,
        "time_step": int(np.abs(np.array([t for _, t in seen]) - trace["time_step"][:n]).max()),
        "obs": float(np.abs(obs - trace["obs"][:n]).max()),
        "actions": float(np.abs(np.stack([o["actions"] for o in outs])
                                - trace["actions"][:n]).max()),
        "anchor": float(np.abs(np.stack([o["next_anchor_quat_w"] for o in outs])
                               - trace["next_anchor_quat_w"][:n]).max()),
    }


@pytest.mark.skipif(not (_REAL.is_file() and _TRACE and Path(_TRACE).is_file()),
                    reason="no BlinkMimic ONNX export or recorded trace")
def test_real_export_matches_recorded_trace():
    pytest.importorskip("onnxruntime")
    with np.load(_TRACE, allow_pickle=False) as z:
        trace = {k: z[k] for k in z.files}
    d = replay_trace(trace)
    assert d["time_step"] == 0 and d["done"]   # ends with the recorded run's last tick
    assert d["obs"] < 1e-5 and d["actions"] < 1e-4 and d["anchor"] < 1e-6

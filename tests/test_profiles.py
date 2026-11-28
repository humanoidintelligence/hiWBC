"""Profiles and the shared controller interface.

The registry tests and the ``echo/default`` profile need nothing installed. The
per-profile interface test loads every shipped profile whose checkpoints are on
this machine and skips the rest (set ``HIWBC_ROBOJUDO_MODELS`` to RoboJuDo's
``assets/models/g1``).
"""

import subprocess
import sys

import numpy as np
import pytest

import hiwbc
from hiwbc import BehaviorSpec, ControllerState, Reference, ReferenceKind, ReferenceSource
from hiwbc.controllers.asap import AsapTrackerController
from hiwbc.controllers.unitree import UnitreeLocoController


def _state(n, yaw=0.0):
    return ControllerState(
        dof_pos=np.zeros(n, np.float32), dof_vel=np.zeros(n, np.float32),
        base_quat=np.array([0, 0, np.sin(yaw / 2), np.cos(yaw / 2)], np.float32),
        base_ang_vel=np.zeros(3, np.float32),
    )


# ----- registry ------------------------------------------------------------------
def test_available_profiles_lists_shipped_modules_without_importing_runtimes():
    names = hiwbc.available_profiles()
    for expected in ("echo/default", "unitree_loco/unitree_g1", "sonic/unitree_g1",
                     "amo_loco/unitree_g1", "asap_tracker/unitree_g1"):
        assert expected in names


def test_listing_and_echo_profile_import_no_model_runtime():
    code = ("import sys, hiwbc; hiwbc.available_profiles(); hiwbc.load('echo/default'); "
            "print('torch' in sys.modules, 'onnxruntime' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["False", "False"]


def test_load_rejects_unknown_profile_option_and_variant():
    with pytest.raises(KeyError, match="unknown profile"):
        hiwbc.load("echo/unitree_h1")
    with pytest.raises(KeyError, match="unknown profile"):
        hiwbc.load("not_a_profile")
    with pytest.raises(TypeError, match="unknown option"):
        hiwbc.load("echo/default", bogus=1)
    with pytest.raises(ValueError, match="no variant"):
        hiwbc.load("echo/default", variant="nope")


@pytest.mark.parametrize("name, options", [
    ("sonic/unitree_g1", {}),
    ("sonic/unitree_g1", {"planner": False}),
    ("unitree_loco/unitree_g1", {}),
    ("unitree_loco/unitree_g1", {"variant": "wo_gait"}),
])
def test_profile_info_matches_the_built_controller(name, options):
    info = hiwbc.profile_info(name, options)
    assert info["description"] == hiwbc.get_profile(name).description
    if info["missing_files"]:
        pytest.skip(f"checkpoints not on this machine: {info['missing_files']}")
    assert info["missing_files"] == []
    try:
        ctrl = hiwbc.load(name, **options)
    except ImportError as e:
        pytest.skip(f"{name}: runtime not installed ({e})")
    assert info["variant"] == ctrl.variant
    assert info["joint_names"] == list(ctrl.joint_names)
    assert info["velocity_limits"] == ctrl.velocity_limits


def test_profile_info_loads_nothing_and_lists_missing_files(tmp_path):
    code = ("import sys, hiwbc; "
            f"i = hiwbc.profile_info('sonic/unitree_g1', {{'checkpoint_dir': {str(tmp_path)!r}}}); "
            f"j = hiwbc.profile_info('unitree_loco/unitree_g1', {{'checkpoint_dir': {str(tmp_path)!r}}}); "
            "print(len(i['missing_files']), len(j['missing_files']), "
            "'torch' in sys.modules, 'onnxruntime' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["3", "1", "False", "False"]

    (tmp_path / "unitree").mkdir()
    (tmp_path / "unitree" / "policy_lstm_1.pt").write_bytes(b"")
    info = hiwbc.profile_info("unitree_loco/unitree_g1", {"checkpoint_dir": str(tmp_path)})
    assert info["missing_files"] == []
    no_planner = hiwbc.profile_info("sonic/unitree_g1", {"checkpoint_dir": str(tmp_path),
                                                         "planner": False, "encoder": False})
    assert no_planner["missing_files"] == [str(tmp_path / "model_decoder.onnx")]
    assert no_planner["velocity_limits"] is None
    assert len(no_planner["joint_names"]) == 29


def test_profile_info_ignores_layer_keys_and_rejects_unknown_names():
    info = hiwbc.profile_info("unitree_loco/unitree_g1", {"type": "wbc", "start_seconds": 0.0})
    assert info["variant"] == "default" and len(info["joint_names"]) == 12
    echo = hiwbc.profile_info("echo/default")
    assert echo["description"] and echo["joint_names"] is None and echo["missing_files"] is None
    with pytest.raises(KeyError):
        hiwbc.profile_info("nope/unitree_g1")
    with pytest.raises(ValueError):
        hiwbc.profile_info("unitree_loco/unitree_g1", {"variant": "nope"})


def test_register_profile_validates_name_and_sets_profile_on_controller():
    with pytest.raises(ValueError, match="<controller>/<robot>"):
        hiwbc.register_profile("bad", build=lambda v, o: None, variants={"default": 1})
    calls = []

    def build(v, o):
        calls.append((v, o))
        return hiwbc.make_controller("echo", num_actions=2, default_pos=[0.1, 0.2])

    hiwbc.register_profile("echo/test_robot_x", build=build, variants={"a": 1, "b": 2},
                           default_options={"k": 0})
    ctrl = hiwbc.load("echo/test_robot_x", variant="b", k=5)
    assert calls == [(2, {"k": 5})]
    assert (ctrl.profile, ctrl.variant) == ("echo/test_robot_x", "b")
    hiwbc.load("echo/test_robot_x")
    assert calls[-1] == (1, {"k": 0})  # no "default" variant: the first one
    with pytest.raises(ValueError, match="already registered"):
        hiwbc.register_profile("echo/test_robot_x", build=build, variants={"a": 1})


def test_echo_profile_stands_and_has_g1_joint_names():
    ctrl = hiwbc.load("echo/default")
    assert len(ctrl.joint_names) == 29 and ctrl.joint_names[0] == "left_hip_pitch"
    assert ctrl.obs_joint_names == ctrl.joint_names
    assert ctrl.kp is None and ctrl.kd is None
    out = ctrl.step(_state(29), ctrl.stand_reference(_state(29)))
    np.testing.assert_array_equal(out.position, np.zeros(29))


# ----- base-class contract ---------------------------------------------------------
def test_base_defaults():
    ctrl = hiwbc.make_controller("echo", num_actions=3, default_pos=[0, 0, 0])
    assert ctrl.joint_names is None and ctrl.kp is None and ctrl.control_dt == 0.02
    assert ctrl.motion_spaces == () and ctrl.cartesian_targets == {} and ctrl.behaviors == []
    with pytest.raises(NotImplementedError):
        ctrl.stand_reference(_state(3))
    with pytest.raises(NotImplementedError):
        ctrl.clip_reference(None, _state(3))
    with pytest.raises(KeyError, match="unknown behavior"):
        ctrl.start_behavior("dance")


def test_base_validates_joint_names_and_gains():
    mk = lambda **kw: hiwbc.make_controller("echo", num_actions=2, default_pos=[0, 0], **kw)
    with pytest.raises(ValueError, match="joint_names has 1"):
        mk(joint_names=["a"])
    with pytest.raises(ValueError, match="repeats"):
        mk(joint_names=["a", "a"])
    with pytest.raises(ValueError, match="kp has shape"):
        mk(joint_names=["a", "b"], kp=[1, 2, 3])
    with pytest.raises(ValueError, match="non-negative"):
        mk(joint_names=["a", "b"], kd=[1, -1])
    with pytest.raises(ValueError, match="needs joint_names"):
        mk(obs_joint_names=["a", "b"])
    ctrl = mk(joint_names=["a", "b"], kp=[10, 20], kd=[1, 2])
    out = ctrl.step(_state(2), Reference(ReferenceKind.LATENT, value=np.zeros(1)))
    np.testing.assert_array_equal(out.kp, [10, 20])
    np.testing.assert_array_equal(out.kd, [1, 2])


def test_velocity_controllers_stand_with_their_neutral_command():
    fwd = lambda obs: np.zeros(12, np.float32)
    ctrl = UnitreeLocoController(num_actions=12, default_pos=[0.0] * 12, forward_fn=fwd)
    ref = ctrl.stand_reference(_state(12))
    assert ref.kind is ReferenceKind.VELOCITY
    np.testing.assert_array_equal(ref.value, [0, 0, 0])
    ctrl = UnitreeLocoController(num_actions=12, default_pos=[0.0] * 12, forward_fn=fwd,
                                 stand_command=(0.5, 0, 0))
    np.testing.assert_array_equal(ctrl.stand_reference(_state(12)).value, [0.5, 0, 0])


def test_reset_resets_stateful_model_fns():
    resets = []

    def fwd(obs):
        return np.zeros(12, np.float32)

    fwd.reset = lambda: resets.append(1)
    ctrl = UnitreeLocoController(num_actions=12, default_pos=[0.0] * 12, forward_fn=fwd)
    ctrl.reset()
    ctrl.reset()
    assert len(resets) == 2


def test_torchscript_runner_reset_restores_recurrent_state(tmp_path):
    torch = pytest.importorskip("torch")

    class Acc(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("h", torch.zeros(1, 2))

        def forward(self, x):
            self.h.add_(x)
            return self.h.clone()

    path = tmp_path / "acc.pt"
    torch.jit.save(torch.jit.script(Acc()), str(path))
    fn = hiwbc.load_torchscript_runner(path)
    assert fn.stateful
    one = np.ones(2, np.float32)
    np.testing.assert_array_equal(fn(one), [1, 1])
    np.testing.assert_array_equal(fn(one), [2, 2])
    fn.reset()
    np.testing.assert_array_equal(fn(one), [1, 1])


# ----- behaviors -------------------------------------------------------------------
def _asap(**kw):
    return AsapTrackerController(num_actions=23, default_pos=[0.0] * 23,
                                 forward_fn=lambda obs: np.zeros(23, np.float32),
                                 motion_length_s=0.2, **kw)


def test_asap_without_motion_name_has_no_behavior():
    assert _asap().behaviors == []


def test_asap_behavior_plays_baked_motion_to_done():
    start = tuple(float(x) for x in np.linspace(0, 0.2, 23))
    ctrl = _asap(motion_name="clip", start_pos=start)
    (spec,) = ctrl.behaviors
    assert isinstance(spec, BehaviorSpec) and spec.name == "clip" and spec.start_pos == pytest.approx(start)
    assert spec.parameters["properties"]["play_speed"]["maximum"] == 2
    src = ctrl.start_behavior("clip")
    assert isinstance(src, ReferenceSource)
    ticks = 0
    while not src.done:
        ctrl.step(_state(23), src.next(_state(23)))
        ticks += 1
        assert ticks < 100
    assert ticks == 11  # 0.2 s at 50 Hz; done on the tick after the phase reaches 1
    # restarting plays from the top again
    src = ctrl.start_behavior("clip", play_speed=2.0)
    assert not src.done
    with pytest.raises(TypeError, match="unknown argument"):
        ctrl.start_behavior("clip", speed=1)
    with pytest.raises(ValueError, match="play_speed"):
        ctrl.start_behavior("clip", play_speed=5)
    with pytest.raises(KeyError):
        ctrl.start_behavior("other")


# ----- every shipped profile: the shared interface ----------------------------------
def _cases():
    out = []
    for name in hiwbc.available_profiles():
        if name.startswith("echo/test_"):
            continue
        out.append(name)
    return out


@pytest.mark.parametrize("name", _cases())
def test_profile_interface(name):
    profile = hiwbc.get_profile(name)
    for variant in profile.variants:
        try:
            ctrl = hiwbc.load(name, variant=variant)
        except FileNotFoundError as e:
            pytest.skip(f"{name}:{variant} checkpoint missing ({e})")
        except ImportError as e:
            pytest.skip(f"{name}: runtime not installed ({e})")
        n = ctrl.num_actions
        assert ctrl.joint_names is not None and len(ctrl.joint_names) == n
        assert set(ctrl.joint_names) <= set(ctrl.obs_joint_names)
        assert ctrl.control_dt > 0
        for g in (ctrl.kp, ctrl.kd):
            assert g is None or (g.shape == (n,) and np.all(g >= 0))
        state = _state(len(ctrl.obs_joint_names), yaw=0.7)
        for b in ctrl.behaviors:
            assert b.start_pos is None or len(b.start_pos) == n
        try:
            ref = ctrl.stand_reference(state)
        except NotImplementedError:
            continue
        ctrl.reset()
        out = ctrl.step(state, ref)
        assert out.position.shape == (n,) and np.isfinite(out.position).all()

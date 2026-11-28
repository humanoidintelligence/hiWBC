"""WbcRobot (the hiRobot ``wbc`` layer) on hiRobot's toy simulator and MuJoCo G1."""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

hirobot = pytest.importorskip("hirobot")
from hirobot import (
    Action,
    ControlMode,
    Embodiment,
    MotionClip,
    RobotBusyError,
    RobotFaultError,
    SafetyError,
    make_robot,
)
from hirobot.drivers.sim.toy.driver import ToyRobot

from hiwbc.controllers.echo import EchoController
from hiwbc.robot import WbcRobot
from hiwbc.types import BehaviorSpec, Reference, ReferenceKind


def _act(**kw) -> Action:
    """Action with list fields turned into arrays (hiRobot takes ndarrays only)."""
    for k in ("joint_target", "kp", "kd", "base_velocity", "latent"):
        if isinstance(kw.get(k), (list, tuple)):
            kw[k] = np.asarray(kw[k], float)
    return Action(**kw)

N = 6
NAMES = [f"j{i}" for i in range(N)]
DRIVEN = NAMES[:4]          # j4, j5 are left to the caller (direct_joints)


def _cfg(n=N, names=None, name="toy", **extra):
    """An ``assembly:`` config: one toy body with joints ``names`` (default ``j0..``)."""
    names = NAMES[:n] if names is None else names
    body = {"device": "body", "control_hz": 50, "joints": [{"name": n_} for n_ in names],
            "driver": {"type": "toy"}}
    return {"assembly": {"name": name, "parts": {"body": body}}, **extra}


def _emb(n=N):
    return Embodiment(name="toy", control_hz=50, joints=[{"name": n_} for n_ in NAMES[:n]])


class Recorder(EchoController):
    """Echo controller that records the references it was stepped with."""

    motion_spaces = ("joint",)

    def __init__(self, fail_at=None, **kw):
        super().__init__(num_actions=len(DRIVEN), default_pos=[0.1, 0.2, 0.3, 0.4],
                         joint_names=DRIVEN, stand_command=(0.0, 0.0, 0.0), **kw)
        self.refs: list[Reference] = []
        self.fail_at = fail_at

    def step(self, state, reference):
        if self.fail_at is not None and len(self.refs) >= self.fail_at:
            raise RuntimeError("boom")
        self.refs.append(reference)
        return super().step(state, reference)

    def clip_reference(self, player, state):
        w = player.window(1, state.base_quat)
        return Reference(ReferenceKind.MOTION_FRAMES, frames={"joint_pos": w["joint_pos"]})

    @property
    def behaviors(self):
        return [BehaviorSpec("wave", "Wave once.", start_pos=(0.0, 0.0, 0.0, 0.0))]

    def start_behavior(self, name, **args):
        return _Countdown(3)


class _Countdown:
    def __init__(self, n):
        self.n = n

    @property
    def done(self):
        return self.n <= 0

    def next(self, state):
        self.n -= 1
        return Reference(ReferenceKind.MOTION_FRAMES, frames={}, extra={"behavior": True})


def _robot(ctrl=None, start_seconds=0.1, **kw):
    inner = make_robot(_cfg())
    robot = WbcRobot(inner, controller=ctrl or Recorder(), start_seconds=start_seconds, **kw)
    robot.connect()
    return robot, inner


def _joint_clip(frames=10):
    return MotionClip(fps=50, joint_names=DRIVEN, joint_pos=np.zeros((frames, 4)),
                      root_pos=np.zeros((frames, 3)), root_quat=np.tile([0, 0, 0, 1.0], (frames, 1)))


def test_capabilities_from_the_controller():
    robot, inner = _robot()
    assert robot.action_modes == (ControlMode.BASE_VELOCITY, ControlMode.MOTION_LATENT,
                                  ControlMode.MOTION_CLIP)
    assert robot.motion_spaces == ("joint",)
    assert robot.direct_joints == ("j4", "j5")
    assert robot.simulated and robot.control_period == pytest.approx(0.02)
    assert [s.name for s in robot.list_layer_skills()] == ["stand", "wave"]
    assert robot.list_onboard_skills() == [] and not robot.has_onboard_skills   # the vendor's: none
    assert [s.name for s in robot.list_robot_skills()] == ["stand", "wave"]     # layer + onboard
    assert robot.now() == inner.now()                    # the clock Observation.stamps are on


def test_connect_ramps_to_default_then_stands():
    ctrl = Recorder()
    robot, inner = _robot(ctrl, start_seconds=0.1)       # 5 ticks
    q = inner.get_observation().joint_pos
    np.testing.assert_allclose(q[:4], ctrl.default_pos, atol=1e-6)
    np.testing.assert_allclose(q[4:], 0)                 # direct joints keep the start pose
    assert ctrl.refs == []                               # the ramp does not call step()
    robot.send_action(_act(mode=ControlMode.BASE_VELOCITY, base_velocity=[0.5, 0, 0]))
    assert ctrl.refs[-1].kind == ReferenceKind.VELOCITY
    np.testing.assert_allclose(ctrl.refs[-1].value, [0.5, 0, 0])
    assert robot.get_status().extra["wbc"]["mode"] == "command"


def test_command_times_out_to_stand_in_sim_time():
    ctrl = Recorder()
    robot, _ = _robot(ctrl, timeout_s=0.1)
    robot.send_action(_act(mode=ControlMode.BASE_VELOCITY, base_velocity=[0.5, 0, 0]))
    for _ in range(5):
        robot._idle_step()
    assert ctrl.refs[-1].value[0] == pytest.approx(0.5)  # 0.1 s after the command
    robot._idle_step()
    np.testing.assert_allclose(ctrl.refs[-1].value, [0, 0, 0])
    assert robot.get_status().extra["wbc"]["mode"] == "stand"


def test_hold_stands_and_keeps_direct_joint_targets():
    ctrl = Recorder()
    robot, inner = _robot(ctrl)
    robot.send_action(_act(mode=ControlMode.BASE_VELOCITY, base_velocity=[0.3, 0, 0],
                             joint_names=["j5"], joint_target=[0.7]))
    robot.hold()
    robot._idle_step()
    np.testing.assert_allclose(ctrl.refs[-1].value, [0, 0, 0])
    np.testing.assert_allclose(inner.get_observation().joint_pos[5], 0.7)


def test_joint_commands_only_for_direct_joints():
    robot, inner = _robot()
    with pytest.raises(SafetyError, match="not free"):
        robot.send_action(_act(mode=ControlMode.BASE_VELOCITY, base_velocity=[0, 0, 0],
                                 joint_names=["j0"], joint_target=[0.0]))
    with pytest.raises(ValueError, match="owns the joints"):
        robot.send_action(_act(mode=ControlMode.POSITION, joint_target=np.zeros(N)))
    robot.send_action(_act(mode=ControlMode.BASE_VELOCITY, base_velocity=[0, 0, 0],
                           joint_names=["j4"], joint_target=[-0.2]))
    np.testing.assert_allclose(inner.get_observation().joint_pos[4], -0.2)
    with pytest.raises(ValueError, match="takes"):
        robot.send_action(_act(mode=ControlMode.CARTESIAN, ee_targets={}))


def test_direct_joints_alone_keep_the_reference_and_its_timeout():
    ctrl = Recorder()
    robot, inner = _robot(ctrl)
    robot.send_action(_act(mode=ControlMode.BASE_VELOCITY, base_velocity=[0.3, 0, 0]))
    for _ in range(10):   # 0.2 s of direct-joint-only actions: still walking
        robot.send_action(_act(mode=ControlMode.BASE_VELOCITY, joint_names=["j5"],
                               joint_target=[0.7]))
    np.testing.assert_allclose(ctrl.refs[-1].value, [0.3, 0, 0])
    np.testing.assert_allclose(inner.get_observation().joint_pos[5], 0.7)
    for _ in range(20):   # past timeout_s (0.5 s): they do not renew the command
        robot.send_action(_act(mode=ControlMode.BASE_VELOCITY, joint_names=["j5"],
                               joint_target=[0.6]))
    np.testing.assert_allclose(ctrl.refs[-1].value, [0, 0, 0])
    assert robot.get_status().extra["wbc"]["mode"] == "stand"
    np.testing.assert_allclose(inner.get_observation().joint_pos[5], 0.6)


def test_base_velocity_limits_come_from_the_controller():
    robot, _ = _robot()
    assert robot.base_velocity_limits is None
    lim = ((-0.5, 1.0), (-0.3, 0.3), (-1.0, 1.0))
    robot, _ = _robot(Recorder(velocity_limits=lim))
    assert robot.base_velocity_limits == lim
    with pytest.raises(ValueError, match="min <= 0 <= max"):
        Recorder(velocity_limits=((0.1, 1.0), (-0.3, 0.3), (-1.0, 1.0)))


def test_robot_configs_are_shipped_and_load():
    from hirobot.loading.config import load_config

    import hiwbc

    configs = hiwbc.robot_configs()
    assert {"g1_sonic", "g1_unitree_loco"} <= set(configs)
    for name, path in configs.items():
        layers = load_config(str(path))["layers"]
        assert [entry["type"] for entry in layers] == ["wbc"], name
        assert hiwbc.get_profile(layers[0]["profile"])


def test_robot_configs_are_found_by_name_through_the_entry_point():
    from hirobot.loading import config

    import hiwbc

    if not hasattr(config, "plugin_roots"):
        pytest.skip("this hiRobot has no hirobot.configs entry points")
    if not any(p.name == "hirobot_configs" for p in config.plugin_roots()):
        pytest.skip("hiwbc is not installed with its entry points")
    for name, path in hiwbc.robot_configs().items():
        assert config.resolve(name) == path


def test_clip_plays_then_stands_and_reports_progress():
    ctrl = Recorder()
    robot, _ = _robot(ctrl)
    robot.send_action(_act(mode=ControlMode.MOTION_CLIP, motion=_joint_clip(10)))
    st = robot.get_status().extra["wbc"]
    assert st["mode"] == "clip" and st["clip_done"] is False
    assert ctrl.refs[-1].kind == ReferenceKind.MOTION_FRAMES
    robot.send_action(_act(mode=ControlMode.MOTION_CLIP, motion=_joint_clip(5), append=True))
    for _ in range(14):
        robot._idle_step()
    st = robot.get_status().extra["wbc"]
    assert st["mode"] == "stand" and st["clip_done"] is True
    assert st["clip_duration"] == pytest.approx(14 * 0.02)
    assert sum(r.kind == ReferenceKind.MOTION_FRAMES for r in ctrl.refs) == 15
    with pytest.raises(ValueError, match="tracks"):
        robot.send_action(_act(mode=ControlMode.MOTION_CLIP, motion=MotionClip(
            fps=50, space="smpl", smpl_joints=np.zeros((3, 24, 3)))))


def test_sending_the_same_clip_again_keeps_playing_it():
    robot, _ = _robot(Recorder())
    clip = _act(mode=ControlMode.MOTION_CLIP, motion=_joint_clip(10))
    for _ in range(5):
        robot.send_action(clip)                    # a simulation advances on each send
    assert robot.get_status().extra["wbc"]["clip_time"] == pytest.approx(5 * 0.02)
    for _ in range(5):                             # 10 frames: done on the 10th send
        robot.send_action(clip)
    assert robot.get_status().extra["wbc"]["clip_done"] is True
    robot.send_action(clip)                        # done: the same clip plays again
    assert robot.get_status().extra["wbc"]["clip_time"] == pytest.approx(0.02)


def test_behavior_ramps_to_its_start_pose_then_plays():
    ctrl = Recorder()
    robot, inner = _robot(ctrl, start_seconds=0.1)
    robot.run_layer_skill("wave")
    assert robot.get_status().extra["wbc"]["mode"] == "ramp"
    assert robot.layer_skill_running
    with pytest.raises(RobotBusyError):                  # the skill holds the command stream
        robot.send_action(_act(mode=ControlMode.BASE_VELOCITY, base_velocity=[0, 0, 0]))
    for _ in range(5):
        robot._idle_step()
    np.testing.assert_allclose(inner.get_observation().joint_pos[:4], 0, atol=1e-6)
    for _ in range(3):
        robot._idle_step()
    assert [r.extra.get("behavior") for r in ctrl.refs[-3:]] == [True] * 3
    robot._idle_step()
    assert robot.get_status().extra["wbc"]["mode"] == "stand"
    assert not robot.layer_skill_running                 # ended on its own: stream given back
    robot.send_action(_act(mode=ControlMode.BASE_VELOCITY, base_velocity=[0, 0, 0]))
    robot.run_layer_skill("stand")
    assert not robot.layer_skill_running


def test_stop_layer_skill_stands_and_gives_the_stream_back():
    ctrl = Recorder()
    robot, _ = _robot(ctrl, start_seconds=0.0)
    robot.run_layer_skill("wave")
    robot._idle_step()
    assert robot.get_status().extra["wbc"]["mode"] == "behavior"
    robot.stop_layer_skill()
    assert robot.get_status().extra["wbc"]["mode"] == "stand" and not robot.layer_skill_running
    robot.send_action(_act(mode=ControlMode.BASE_VELOCITY, base_velocity=[0.2, 0, 0]))
    assert robot.get_status().extra["wbc"]["mode"] == "command"


def test_outer_hz_runs_ticks_on_the_running_total():
    ctrl = Recorder()
    robot, _ = _robot(ctrl, outer_hz=30)
    counts = []
    for _ in range(6):
        before = len(ctrl.refs)
        robot.send_action(_act(mode=ControlMode.BASE_VELOCITY, base_velocity=[0, 0, 0]))
        counts.append(len(ctrl.refs) - before)
    assert sum(counts) == 10 and set(counts) == {1, 2}   # 6 / 30 s = 10 ticks
    assert robot.control_period == pytest.approx(1 / 30)


def test_step_error_damps_and_faults_until_reset():
    ctrl = Recorder(fail_at=2)
    robot, _inner = _robot(ctrl)
    act = _act(mode=ControlMode.BASE_VELOCITY, base_velocity=[0, 0, 0])
    robot.send_action(act)
    robot.send_action(act)
    with pytest.raises(RobotFaultError, match="boom"):
        robot.send_action(act)
    assert "boom" in robot.fault and "boom" in robot.get_status().fault
    with pytest.raises(RobotFaultError):
        robot.send_action(act)
    ctrl.fail_at = None
    robot.reset()
    assert robot.fault is None
    robot.send_action(act)


def test_tracker_without_stand_reference_holds_default_pose():
    class Tracker(Recorder):
        accepts = (ReferenceKind.MOTION_FRAMES,)

        def stand_reference(self, state):
            raise NotImplementedError

    ctrl = Tracker()
    robot, inner = _robot(ctrl)
    assert robot.action_modes == (ControlMode.MOTION_CLIP,)
    robot._idle_step()
    assert ctrl.refs == []
    np.testing.assert_allclose(inner.get_observation().joint_pos[:4], ctrl.default_pos, atol=1e-6)


def test_controller_gains_need_direct_gains():
    inner = make_robot(_cfg())
    kw = {"kp": [50.0] * 4, "kd": [1.0] * 4}
    with pytest.raises(ValueError, match="direct_kp"):
        WbcRobot(inner, controller=Recorder(**kw))
    WbcRobot(inner, controller=Recorder(**kw), direct_kp=20.0, direct_kd=0.5)
    with pytest.raises(ValueError, match="not in the toy embodiment"):
        WbcRobot(make_robot(_cfg(n=3)), controller=Recorder())


def test_make_robot_builds_the_wbc_layer_from_a_profile():
    from hiwbc.configs.echo.default import G1_29

    cfg = _cfg(names=G1_29, name="g1",
               layers=[{"type": "wbc", "profile": "echo/default", "start_seconds": 0.0}])
    robot = make_robot(cfg)
    layer = robot
    while not isinstance(layer, WbcRobot):
        layer = layer._inner          # the stack from outside (tests only; layers use .inner)
    robot.connect()
    robot.send_action(_act(mode=ControlMode.BASE_VELOCITY, base_velocity=[0, 0, 0]))
    assert robot.get_status().extra["wbc"]["profile"] == "echo/default"
    with pytest.raises(TypeError, match="unknown option"):
        make_robot({**cfg, "layers": [{"type": "wbc", "profile": "echo/default", "nope": 1}]})


def test_real_time_thread_ticks_and_disconnect_damps():
    class RealToy(ToyRobot):
        simulated = False
        damped = False

        def damp(self):
            self.damped = True


    inner = RealToy(_emb())
    ctrl = Recorder()
    robot = WbcRobot(inner, controller=ctrl, start_seconds=0.0)
    robot.connect()
    robot.send_action(_act(mode=ControlMode.BASE_VELOCITY, base_velocity=[0.2, 0, 0]))
    time.sleep(0.2)
    robot.disconnect()
    n = robot.get_status().extra["wbc"]["ticks"]
    assert 5 <= n <= 15 and inner.damped
    assert any(r.value is not None and r.value[0] == pytest.approx(0.2) for r in ctrl.refs)


def test_estop_stops_the_robot_below_first_then_the_thread():
    class RealToy(ToyRobot):
        simulated = False
        stopped_with_thread_alive = None

        def _estop(self):
            t = layer._thread
            self.stopped_with_thread_alive = t is not None and t.is_alive()


    inner = RealToy(_emb())
    layer = WbcRobot(inner, controller=Recorder(), start_seconds=0.0)
    layer.connect()
    time.sleep(0.05)
    layer.estop()
    assert inner.stopped_with_thread_alive          # the hardware went first
    assert layer.fault == "estop()" and inner.fault == "estop()"
    layer._thread.join(1.0)
    assert not layer._thread.is_alive()
    with pytest.raises(RobotFaultError):
        layer.send_action(_act(mode=ControlMode.BASE_VELOCITY, base_velocity=[0, 0, 0]))
    layer.reset()
    assert layer.fault is None
    layer.disconnect()


def test_tick_failure_estops_the_stack():
    robot, inner = _robot(Recorder(fail_at=0))
    with pytest.raises(RobotFaultError, match="boom"):
        robot.send_action(_act(mode=ControlMode.BASE_VELOCITY, base_velocity=[0, 0, 0]))
    assert "boom" in robot.fault
    assert inner.fault == "estop()"                 # the robot below was e-stopped


def test_wbc_layer_uses_only_the_public_api_of_inner():
    from hirobot.testing import assert_layer_uses_public_api

    used = assert_layer_uses_public_api(
        WbcRobot, {"controller": Recorder(), "start_seconds": 0.1}, inner=make_robot(_cfg()))
    assert {"send_action", "get_observation"} <= used


G1_MJCF = os.path.expanduser("~/.cache/hirobot/assets/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.xml")


@pytest.mark.skipif(not os.path.isfile(G1_MJCF), reason="unitree_ros G1 MJCF not present")
def test_mujoco_g1_echo_ramps_and_holds_on_a_gantry():
    pytest.importorskip("mujoco")
    robot = make_robot("unitree_g1_mujoco", sim={"realtime": False, "ramp_seconds": 0.0})
    default = np.zeros(29)
    default[[3, 9]] = 0.3
    robot = WbcRobot(robot, "echo/default", default_pos=list(default), start_seconds=0.5)
    robot.connect()
    for _ in range(50):
        robot.send_action(_act(mode=ControlMode.BASE_VELOCITY, base_velocity=[0, 0, 0]))
    q = robot.get_observation().joint_pos
    np.testing.assert_allclose(q[[3, 9]], 0.3, atol=0.05)
    robot.disconnect()


# ----- CARTESIAN (rate-limited) and BASE_POSE -----------------------------------------
REST = {"lw": np.array([0.1, 0.2, -0.2, 0.0, 0.0, 0.0, 1.0]),
        "rw": np.array([0.1, -0.2, -0.2, 0.0, 0.0, 0.0, 1.0])}


class Reacher(Recorder):
    """Recorder with two CARTESIAN targets and rest poses, like SONIC with its planner."""

    def __init__(self, rest=REST, **kw):
        super().__init__(cartesian_targets={"left_hand": "lw", "right_hand": "rw"}, **kw)
        self._rest = rest

    def cartesian_rest(self):
        return dict(self._rest)


def _ee(ref, name):
    return ref.extra.get("ee_targets", {}).get(name)


def _reach(robot, targets, vel=(0.0, 0.0, 0.0)):
    robot.send_action(Action(mode=ControlMode.CARTESIAN,
                             ee_targets={k: np.asarray(v, float) for k, v in targets.items()},
                             base_velocity=np.asarray(vel, float)))


def test_cartesian_mode_and_targets_from_the_controller():
    robot, _ = _robot(Reacher())
    assert ControlMode.CARTESIAN in robot.action_modes
    assert robot.cartesian_targets == ("left_hand", "right_hand")
    plain, _ = _robot()
    assert ControlMode.CARTESIAN not in plain.action_modes and plain.cartesian_targets == ()


def test_cartesian_target_moves_at_the_speed_limit_from_rest():
    ctrl = Reacher()
    robot, _ = _robot(ctrl, cartesian_max_speed=0.5)     # 1 cm per 20 ms tick
    goal = REST["lw"] + [0.2, 0, 0, 0, 0, 0, 0]
    _reach(robot, {"left_hand": goal}, vel=(0.3, 0, 0))
    ref = ctrl.refs[-1]
    assert ref.kind == ReferenceKind.VELOCITY
    np.testing.assert_allclose(ref.value, [0.3, 0, 0])
    np.testing.assert_allclose(_ee(ref, "lw")[:3], REST["lw"][:3] + [0.01, 0, 0], atol=1e-9)
    assert _ee(ref, "rw") is None                         # only commanded targets are sent
    for _ in range(19):
        _reach(robot, {"left_hand": goal})
    np.testing.assert_allclose(_ee(ctrl.refs[-1], "lw")[:3], goal[:3], atol=1e-9)   # 20 ticks
    _reach(robot, {"left_hand": goal})
    np.testing.assert_allclose(_ee(ctrl.refs[-1], "lw"), goal, atol=1e-9)


def test_cartesian_rotation_is_rate_limited_along_the_short_path():
    ctrl = Reacher()
    robot, _ = _robot(ctrl, cartesian_max_rate=1.0)      # 0.02 rad per tick
    yaw = 0.5
    goal = np.r_[REST["lw"][:3], 0, 0, np.sin(yaw / 2), np.cos(yaw / 2)]
    _reach(robot, {"left_hand": np.r_[goal[:3], -goal[3:]]})   # same rotation, negated quaternion
    q = _ee(ctrl.refs[-1], "lw")[3:]
    angle = 2 * np.arccos(min(1.0, abs(q[3])))
    assert angle == pytest.approx(0.02, abs=1e-9) and q[2] * q[3] > 0   # toward +yaw, not the long way
    for _ in range(30):
        _reach(robot, {"left_hand": goal})
    np.testing.assert_allclose(abs(np.dot(_ee(ctrl.refs[-1], "lw")[3:], goal[3:])), 1.0, atol=1e-9)


def test_cartesian_returns_to_rest_then_drops_the_target():
    ctrl = Reacher()
    robot, _ = _robot(ctrl, cartesian_max_speed=0.5)
    goal = REST["lw"] + [0.05, 0, 0, 0, 0, 0, 0]
    for _ in range(5):
        _reach(robot, {"left_hand": goal})
    robot.send_action(_act(mode=ControlMode.BASE_VELOCITY, base_velocity=[0, 0, 0]))
    np.testing.assert_allclose(_ee(ctrl.refs[-1], "lw")[:3], goal[:3] - [0.01, 0, 0], atol=1e-9)
    for _ in range(3):
        robot.send_action(_act(mode=ControlMode.BASE_VELOCITY, base_velocity=[0, 0, 0]))
    assert _ee(ctrl.refs[-1], "lw") is not None
    robot.send_action(_act(mode=ControlMode.BASE_VELOCITY, base_velocity=[0, 0, 0]))
    assert "ee_targets" not in ctrl.refs[-1].extra       # back at rest: plain velocity
    assert robot.get_status().extra["wbc"]["ee_targets"] == {}


def test_cartesian_ends_at_once_for_a_clip_or_without_a_rest_pose():
    ctrl = Reacher()
    robot, _ = _robot(ctrl)
    _reach(robot, {"left_hand": REST["lw"] + [0.1, 0, 0, 0, 0, 0, 0]})
    robot.send_action(Action(mode=ControlMode.MOTION_CLIP, motion=_joint_clip()))
    assert ctrl.refs[-1].kind == ReferenceKind.MOTION_FRAMES
    assert robot.get_status().extra["wbc"]["ee_targets"] == {}

    ctrl = Reacher(rest={})                               # no rest: start at the first target
    robot, _ = _robot(ctrl)
    goal = REST["lw"] + [0.3, 0, 0, 0, 0, 0, 0]
    _reach(robot, {"left_hand": goal})
    np.testing.assert_allclose(_ee(ctrl.refs[-1], "lw"), goal)
    robot.send_action(_act(mode=ControlMode.BASE_VELOCITY, base_velocity=[0, 0, 0]))
    assert "ee_targets" not in ctrl.refs[-1].extra


def test_cartesian_rejects_bad_targets():
    robot, _ = _robot(Reacher())
    with pytest.raises(ValueError, match="tracks"):
        _reach(robot, {"head": REST["lw"]})
    with pytest.raises(ValueError, match="7 finite"):
        _reach(robot, {"left_hand": REST["lw"][:6]})
    with pytest.raises(ValueError, match="7 finite"):
        _reach(robot, {"left_hand": np.r_[np.nan, REST["lw"][1:]]})
    with pytest.raises(ValueError, match="zero quaternion"):
        _reach(robot, {"left_hand": np.r_[REST["lw"][:3], 0, 0, 0, 0]})
    with pytest.raises(ValueError, match="ee_targets"):
        robot.send_action(Action(mode=ControlMode.CARTESIAN, base_velocity=np.zeros(3)))
    with pytest.raises(ValueError, match="positive"):
        WbcRobot(make_robot(_cfg()), controller=Reacher(), cartesian_max_speed=0.0)


def test_cartesian_quaternion_is_normalized():
    ctrl = Reacher(rest={})
    robot, _ = _robot(ctrl)
    _reach(robot, {"left_hand": np.r_[REST["lw"][:3], 0, 0, 0, 2.0]})
    np.testing.assert_allclose(_ee(ctrl.refs[-1], "lw")[3:], [0, 0, 0, 1])


class _Based:
    """Wraps the toy sim and reports a settable floating-base pose (for BASE_POSE)."""

    def __init__(self, inner):
        self._inner = inner
        self.pos = np.zeros(3)
        self.yaw = 0.0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def get_observation(self):
        obs = self._inner.get_observation()
        obs.base_position = self.pos.copy()
        obs.base_orientation = np.array([0, 0, np.sin(self.yaw / 2), np.cos(self.yaw / 2)])
        obs.base_angular_velocity = np.zeros(3)
        return obs


def _based_robot(ctrl=None, **kw):
    base = _Based(make_robot(_cfg()))
    robot = WbcRobot(base, controller=ctrl or Recorder(), start_seconds=0.0, **kw)
    robot.connect()
    return robot, base


def _goal(robot, x, y, yaw):
    robot.send_action(Action(mode=ControlMode.BASE_POSE, base_target=np.array([x, y, yaw], float)))


def test_base_pose_needs_a_base_position():
    robot, _ = _robot()
    assert ControlMode.BASE_POSE not in robot.action_modes
    robot, _ = _based_robot()
    assert ControlMode.BASE_POSE in robot.action_modes


def test_base_pose_walks_in_the_body_frame_with_limits():
    ctrl = Recorder()
    robot, base = _based_robot(ctrl, goto={"max_speed": 0.5, "max_yaw_rate": 0.4})
    base.yaw = np.pi / 2                                 # facing +y
    _goal(robot, 0.0, 2.0, np.pi / 2)                    # 2 m straight ahead
    np.testing.assert_allclose(ctrl.refs[-1].value, [0.5, 0, 0], atol=1e-6)
    _goal(robot, 0.2, 0.0, np.pi / 2)                    # 0.2 m to the robot's right
    np.testing.assert_allclose(ctrl.refs[-1].value, [0, -0.2, 0], atol=1e-6)
    _goal(robot, 0.0, 0.0, -np.pi / 2)                   # turn only, wrap to the short way
    v = ctrl.refs[-1].value
    np.testing.assert_allclose(v[:2], 0, atol=1e-6)
    assert abs(v[2]) == pytest.approx(0.4)
    assert robot.get_status().extra["wbc"]["mode"] == "goto"


def test_base_pose_arrives_and_stands():
    ctrl = Recorder()
    robot, base = _based_robot(ctrl)
    _goal(robot, 1.0, 0.0, 0.0)
    st = robot.get_status().extra["wbc"]
    assert st["goal"] == [1.0, 0.0, 0.0] and not st["arrived"]
    base.pos[:] = [0.95, 0.02, 0.0]                      # inside pos_tol and yaw_tol
    robot._idle_step()
    st = robot.get_status().extra["wbc"]
    assert st["arrived"] and st["mode"] == "stand"
    np.testing.assert_allclose(ctrl.refs[-1].value, [0, 0, 0])   # the stand command
    robot._idle_step()                                    # standing: no timeout, no new goal
    assert robot.get_status().extra["wbc"]["mode"] == "stand"


def test_a_velocity_command_replaces_the_goal():
    ctrl = Recorder()
    robot, _ = _based_robot(ctrl)
    _goal(robot, 3.0, 0.0, 0.0)
    robot.send_action(_act(mode=ControlMode.BASE_VELOCITY, base_velocity=[0, 0.1, 0]))
    assert robot.get_status().extra["wbc"]["mode"] == "command"
    np.testing.assert_allclose(ctrl.refs[-1].value, [0, 0.1, 0])


def test_base_pose_rejects_bad_targets():
    robot, _ = _based_robot()
    with pytest.raises(ValueError, match="finite"):
        _goal(robot, np.nan, 0, 0)
    with pytest.raises(TypeError, match="goto"):
        WbcRobot(make_robot(_cfg()), controller=Recorder(), goto={"kp": 1.0})



# ----- RobotStatus.command (hiRobot's generic progress of the last high-level command) ---
def test_command_progress_base_pose_walks_then_arrives():
    robot, base = _based_robot()
    assert robot.get_status().command is None              # nothing commanded yet
    goal = Action(mode=ControlMode.BASE_POSE, base_target=np.array([1.0, 0.0, 0.0]))
    robot.send_action(goal)
    cmd = robot.get_status().command
    assert cmd.mode == ControlMode.BASE_POSE and cmd.done is False
    assert cmd.goal == [1.0, 0.0, 0.0] and cmd.clip_time is None
    base.pos[:] = [0.95, 0.02, 0.0]                        # inside pos_tol and yaw_tol
    robot._idle_step()
    cmd = robot.get_status().command
    assert cmd.done is True and cmd.goal == [1.0, 0.0, 0.0]   # kept after arrival
    robot._idle_step()
    assert robot.get_status().command.done is True        # until the next command
    robot.send_action(goal)                                # sent again while there
    assert robot.get_status().command.done is True        # (this send's tick re-arrives)


def test_command_progress_clip_plays_then_ends():
    robot, _ = _robot(Recorder())
    robot.send_action(_act(mode=ControlMode.MOTION_CLIP, motion=_joint_clip(10)))
    cmd = robot.get_status().command
    assert cmd.mode == ControlMode.MOTION_CLIP and cmd.done is False and cmd.goal is None
    assert cmd.clip_duration == pytest.approx(9 * 0.02)
    assert cmd.clip_time == pytest.approx(0.02)
    robot.send_action(_act(mode=ControlMode.MOTION_CLIP, motion=_joint_clip(5), append=True))
    cmd = robot.get_status().command
    assert cmd.clip_duration == pytest.approx(14 * 0.02)  # the whole append queue
    for _ in range(14):
        robot._idle_step()
    cmd = robot.get_status().command
    assert cmd.mode == ControlMode.MOTION_CLIP and cmd.done is True
    assert cmd.clip_time == pytest.approx(cmd.clip_duration)
    assert robot.get_status().extra["wbc"]["mode"] == "stand"


def test_command_progress_velocity_has_no_end():
    robot, _ = _robot(Recorder(), timeout_s=0.1)
    robot.send_action(_act(mode=ControlMode.BASE_VELOCITY, base_velocity=[0.5, 0, 0]))
    cmd = robot.get_status().command
    assert cmd.mode == ControlMode.BASE_VELOCITY and cmd.done is None
    assert cmd.goal is None and cmd.clip_time is None and cmd.clip_duration is None
    for _ in range(10):                                    # timed out to a stand
        robot._idle_step()
    assert robot.get_status().command.mode == ControlMode.BASE_VELOCITY
    robot.hold()                                           # the caller ended it
    assert robot.get_status().command is None


def test_command_progress_is_none_while_a_layer_skill_runs():
    robot, _ = _robot(Recorder(), start_seconds=0.0)
    robot.send_action(_act(mode=ControlMode.MOTION_CLIP, motion=_joint_clip(10)))
    robot.run_layer_skill("wave")                          # preempts the caller's clip
    robot._idle_step()
    assert robot.layer_skill_running and robot.get_status().command is None
    robot.stop_layer_skill()
    assert robot.get_status().command is None
    robot.send_action(_act(mode=ControlMode.BASE_VELOCITY, base_velocity=[0.2, 0, 0]))
    assert robot.get_status().command.mode == ControlMode.BASE_VELOCITY


class _FrozenToy(ToyRobot):
    """Toy body whose joint state stops arriving once ``frozen`` is set (stamp stuck)."""

    frozen_at = None

    def get_observation(self):
        obs = super().get_observation()
        if self.frozen_at is not None:
            obs.stamps["body"] = self.frozen_at
        return obs

    def freeze(self):
        self.frozen_at = super().get_observation().stamps["body"]


def test_stale_body_state_faults_and_estops():
    inner = _FrozenToy(_emb())
    robot = WbcRobot(inner, controller=Recorder(), start_seconds=0.0, stale_s=0.1)
    robot.connect()
    walk = _act(mode=ControlMode.BASE_VELOCITY, base_velocity=[0.1, 0, 0])
    for _ in range(10):
        robot.send_action(walk)
    inner.freeze()
    for _ in range(5):                       # 0.1 s at 50 Hz: still fresh enough
        robot.send_action(walk)
    with pytest.raises(RobotFaultError, match="body state is .* old"):
        for _ in range(3):
            robot.send_action(walk)
    assert "stale_s" in robot.fault and inner.fault == "estop()"


def test_stale_check_can_be_turned_off():
    inner = _FrozenToy(_emb())
    robot = WbcRobot(inner, controller=Recorder(), start_seconds=0.0, stale_s=None)
    robot.connect()
    inner.freeze()
    for _ in range(20):
        robot.send_action(_act(mode=ControlMode.BASE_VELOCITY, base_velocity=[0.1, 0, 0]))
    assert robot.fault is None
    with pytest.raises(ValueError, match="stale_s"):
        WbcRobot(_FrozenToy(_emb()), controller=Recorder(), stale_s=0.0)

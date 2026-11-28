"""The ``wbc`` layer on one part of a hiRobot assembly (``layers: [{type: wbc, part: g1}]``).

A toy G1 (the echo controller drives its 29 joints) next to a toy arm, built by
``make_robot`` from an ``assembly:`` config. The layer sees only the G1; the arm
is commanded directly beside it.
"""

from __future__ import annotations

import numpy as np
import pytest
from hirobot import Action, ControlMode, RobotFaultError, make_robot

from hiwbc.configs.echo.default import G1_29
from hiwbc.robot import WbcRobot

ARM = [f"a{i}" for i in range(3)]


def _body(names, floating=False):
    return {"device": "body", "control_hz": 50, "floating_base": floating,
            "joints": [{"name": n} for n in names], "driver": {"type": "toy"}}


def _cfg(**layer):
    g1 = {"assembly": {"parts": {"body": _body(G1_29, floating=True)}}}
    arm = {"assembly": {"parts": {"body": _body(ARM)}}}
    return {"assembly": {"name": "cell", "parts": {"g1": g1, "arm": arm}},
            "layers": [{"type": "wbc", "part": "g1", "profile": "echo/default",
                        "start_seconds": 0.0, **layer}]}


def _robot(**layer):
    robot = make_robot(_cfg(**layer))
    robot.connect()
    return robot


def _wbc(robot) -> WbcRobot:
    stack = robot
    while not hasattr(stack, "layers"):
        stack = stack._inner          # tests only: reach the PartLayers in the stack
    layer = stack.layers["g1"]
    assert isinstance(layer, WbcRobot)
    return layer


def _arm(q):
    return Action(mode=ControlMode.POSITION, joint_target=np.full(3, q))


def _walk():
    return Action(mode=ControlMode.BASE_VELOCITY, base_velocity=np.zeros(3))


def test_the_wbc_layer_sees_only_its_part():
    robot = _robot()
    wbc = _wbc(robot)
    assert list(wbc.embodiment.joint_names) == list(G1_29)   # the part's own names
    assert wbc.direct_joints == ()                           # echo drives all 29
    assert ControlMode.BASE_VELOCITY in wbc.action_modes
    robot.disconnect()


def test_the_arm_is_commanded_beside_the_controlled_g1():
    robot = _robot()
    wbc = _wbc(robot)
    robot.send_action(Action(parts={"g1": _walk(), "arm": _arm(0.2)}))
    assert np.allclose(robot.part("arm").get_observation().joint_pos, 0.2)
    assert wbc.get_status().extra["wbc"]["mode"] == "command"
    with pytest.raises(ValueError):                          # joint modes are not the layer's
        robot.send_action(Action(parts={"g1": Action(mode=ControlMode.POSITION,
                                                     joint_target=np.zeros(29))}))
    robot.disconnect()


def test_the_controller_still_ticks_when_only_the_arm_is_commanded():
    robot = _robot()
    wbc = _wbc(robot)
    robot.send_action(Action(parts={"g1": _walk(), "arm": _arm(0.2)}))
    ticks = wbc.get_status().extra["wbc"]["ticks"]
    robot.send_action(Action(parts={"arm": _arm(0.3)}))
    assert np.allclose(robot.part("arm").get_observation().joint_pos, 0.3)
    assert wbc.get_status().extra["wbc"]["ticks"] == ticks + 1


def test_layer_skills_are_prefixed_and_hold_only_the_g1():
    robot = _robot()
    assert [s.name for s in robot.list_layer_skills()] == ["g1/stand"]
    robot.run_layer_skill("g1/stand")                        # immediate: the stream comes back
    robot.send_action(Action(parts={"g1": _walk()}))
    robot.disconnect()


def test_a_controller_error_stops_the_whole_tree_until_reset():
    robot = _robot()
    wbc = _wbc(robot)

    def broken(state, ref):
        raise RuntimeError("injected failure")

    real_step = wbc.controller.step
    wbc.controller.step = broken
    with pytest.raises(RobotFaultError):
        robot.send_action(Action(parts={"g1": _walk(), "arm": _arm(0.1)}))
    assert robot.fault is not None
    with pytest.raises(RobotFaultError):                     # the arm is refused too
        robot.send_action(Action(parts={"arm": _arm(0.1)}))
    wbc.controller.step = real_step
    robot.reset()
    assert robot.fault is None
    robot.send_action(Action(parts={"g1": _walk(), "arm": _arm(0.1)}))
    assert np.allclose(robot.part("arm").get_observation().joint_pos, 0.1)
    robot.disconnect()


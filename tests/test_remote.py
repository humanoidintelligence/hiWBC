"""Remote model forward: wire format, server, client timeouts, WbcRobot misses."""

from __future__ import annotations

import socket
import struct
import threading
import time

import numpy as np
import pytest

from hiwbc import remote
from hiwbc.controllers.echo import EchoController
from hiwbc.remote import ForwardTimeout, RemoteError, RemoteFn, decode, encode, make_remote
from hiwbc.serve import ModelServer
from hiwbc.types import ReferenceKind


# ----- wire format ------------------------------------------------------------
def _roundtrip(msg):
    return decode(encode(msg)[4:])


def test_encode_roundtrips_arrays_dicts_tuples():
    msg = {"op": "call", "args": [
        np.arange(6, dtype=np.float32).reshape(2, 3),
        (np.ones(2, np.float64), np.array([3], np.int64)),
        {"mode": np.array([1], np.int64), "flag": np.array([True])},
        None, 1.5, "x",
    ]}
    out = _roundtrip(msg)
    a, t, d, n, f, s = out["args"]
    np.testing.assert_array_equal(a, msg["args"][0])
    assert a.dtype == np.float32 and a.shape == (2, 3)
    assert isinstance(t, tuple) and t[1].dtype == np.int64
    assert d["flag"].dtype == bool and (n, f, s) == (None, 1.5, "x")
    assert np.float16(1).dtype.name not in remote._DTYPES
    assert _roundtrip({"x": np.float16(1.0)})["x"].dtype == np.float32   # cast, not refused


def test_decoded_arrays_are_writable_copies():
    a = _roundtrip({"a": np.zeros(3, np.float32)})["a"]
    a[0] = 1.0


@pytest.mark.parametrize("payload", [
    b"",
    struct.pack(">I", 3) + b"{x}",
    struct.pack(">I", 2) + b"[]",
])
def test_malformed_payloads_raise_value_error(payload):
    with pytest.raises(ValueError):
        decode(payload)


def test_array_specs_are_checked():
    good = encode({"a": np.zeros(4, np.float32)})[4:]
    (n,) = struct.unpack_from(">I", good)
    head = good[4:4 + n].decode()
    for bad in (head.replace('"float32"', '"object"'), head.replace("[4]", "[400]"),
                head.replace('{"__nd__":0}', '{"__nd__":5}')):
        b = bad.encode()
        with pytest.raises(ValueError):
            decode(struct.pack(">I", len(b)) + b + good[4 + n:])
    with pytest.raises(ValueError, match="trailing"):
        decode(good + b"\0")


def test_unsendable_values_raise():
    with pytest.raises(TypeError):
        encode({"f": object()})
    with pytest.raises(TypeError):
        encode({"d": {1: 2}})


def test_parse_endpoint():
    assert remote.parse_endpoint("tcp://10.0.0.2:5555") == ("10.0.0.2", 5555)
    assert remote.parse_endpoint("localhost:1") == ("localhost", 1)
    for bad in ("tcp://host", "5555", "tcp://:5"):
        with pytest.raises(ValueError):
            remote.parse_endpoint(bad)


# ----- server + client with stub models ----------------------------------------------
class Stub(EchoController):
    """Echo controller whose model callables are plain functions (as runtime runners are)."""

    def __init__(self, delay=0.0):
        super().__init__(num_actions=3, default_pos=[0.0, 0.0, 0.0])
        self.delay = delay
        self.calls = 0

        def forward_fn(obs):
            self.calls += 1
            if self.delay:
                time.sleep(self.delay)
            return np.asarray(obs, np.float32) * 2.0

        def pair_fn(a, b):
            return {"sum": a + b}

        self._forward_fn = forward_fn
        self._pair_fn = pair_fn
        self.profile, self.variant = "stub/robot", "default"

    def observe(self, state, reference):
        return np.asarray(reference.value, np.float32)

    def forward(self, obs):
        return self._forward_fn(obs)


@pytest.fixture
def served():
    started = []

    def start(ctrl=None):
        ctrl = ctrl or Stub()
        server = ModelServer(ctrl, "127.0.0.1", 0).start()
        started.append(server)
        return ctrl, server, f"tcp://127.0.0.1:{server.port}"

    yield start
    for s in started:
        s.stop()


def test_remote_call_matches_local(served):
    ctrl, _, ep = served()
    robot_side = Stub()
    local = robot_side._forward_fn(np.array([1.0, 2.0, 3.0]))
    fns = make_remote(robot_side, ep, timeout_s=1.0)
    assert set(fns) == {"_forward_fn", "_pair_fn"}
    assert isinstance(robot_side._forward_fn, RemoteFn)
    out = robot_side._forward_fn(np.array([1.0, 2.0, 3.0], np.float32))
    np.testing.assert_array_equal(out, local)
    assert ctrl.calls == 1 and robot_side.calls == 1        # the second call ran on the server
    r = robot_side._pair_fn(np.ones(2), np.ones(2))
    np.testing.assert_array_equal(r["sum"], [2, 2])


def test_server_refuses_another_profile_or_variant(served):
    _, _, ep = served()
    other = Stub()
    other.variant = "fast"
    with pytest.raises(RemoteError, match="stub/robot:default"):
        make_remote(other, ep)
    nofn = RemoteFn(ep, name="_nope_fn", profile="stub/robot", variant="default", timeout_s=1.0)
    with pytest.raises(RemoteError, match="no model"):
        nofn.connect()


def test_server_reports_model_errors_and_keeps_serving(served):
    _, _, ep = served()
    robot_side = Stub()
    make_remote(robot_side, ep, timeout_s=1.0)
    with pytest.raises(RemoteError, match="TypeError"):
        robot_side._pair_fn(np.ones(2))                        # wrong arity, raised server-side
    np.testing.assert_array_equal(robot_side._pair_fn(np.ones(1), np.ones(1))["sum"], [2])


def test_make_remote_needs_a_loaded_stateless_controller():
    bare = Stub()
    bare.profile = None
    with pytest.raises(ValueError, match="hiwbc.load"):
        make_remote(bare, "tcp://127.0.0.1:1", connect=False)
    rnn = Stub()
    rnn._forward_fn.stateful = True
    with pytest.raises(ValueError, match="recurrent"):
        make_remote(rnn, "tcp://127.0.0.1:1", connect=False)
    with pytest.raises(ValueError, match="recurrent"):
        ModelServer(rnn, "127.0.0.1", 0)


def test_slow_server_times_out_then_reconnects(served):
    slow = Stub(delay=0.2)
    _, _, ep = served(slow)
    robot_side = Stub()
    fns = make_remote(robot_side, ep, timeout_s=0.05)
    with pytest.raises(ForwardTimeout):
        robot_side._forward_fn(np.ones(3, np.float32))
    slow.delay = 0.0
    time.sleep(0.25)                                          # the late reply is dropped with the socket
    np.testing.assert_array_equal(robot_side._forward_fn(np.ones(3, np.float32)), [2, 2, 2])
    assert fns["_forward_fn"].reconnects == 1


def test_server_gone_is_a_timeout_not_a_hang(served):
    _, server, ep = served()
    robot_side = Stub()
    make_remote(robot_side, ep, timeout_s=0.05)
    server.stop()
    t0 = time.monotonic()
    for _ in range(3):
        with pytest.raises(ForwardTimeout):
            robot_side._forward_fn(np.ones(3, np.float32))
    assert time.monotonic() - t0 < 1.0


def test_bad_first_message_is_refused(served):
    _, server, _ = served()
    with socket.create_connection(("127.0.0.1", server.port), timeout=1.0) as s:
        remote.send(s, {"op": "call", "args": []})
        assert remote.recv(s)["ok"] is False


def test_concurrent_clients_share_the_model(served):
    ctrl, _, ep = served()
    errors = []

    def worker():
        try:
            side = Stub()
            make_remote(side, ep, timeout_s=2.0)
            for i in range(20):
                np.testing.assert_array_equal(side._forward_fn(np.full(3, i, np.float32)), np.full(3, 2 * i))
        except Exception as e:  # noqa: BLE001 - collected and asserted below
            errors.append(e)

    ts = [threading.Thread(target=worker) for _ in range(4)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors and ctrl.calls == 80


# ----- hiwbc.load(runtime="remote") ------------------------------------------------
def test_load_checks_runtime_options():
    import hiwbc

    with pytest.raises(ValueError, match="endpoint"):
        hiwbc.load("echo/default", runtime="remote")
    with pytest.raises(ValueError, match="endpoint"):
        hiwbc.load("echo/default", endpoint="tcp://127.0.0.1:1")
    with pytest.raises(ValueError, match="runtime"):
        hiwbc.load("echo/default", runtime="gpu")
    ctrl = hiwbc.load("echo/default")
    assert not hasattr(ctrl, "remote_fns")


# ----- WbcRobot: a timed-out tick repeats the last target, too many damp ----------
hirobot = pytest.importorskip("hirobot")


class _Flaky(Stub):
    """Raises ForwardTimeout on the ticks in ``fail``, as a RemoteFn would."""

    motion_spaces = ()

    def __init__(self, fail=()):
        super().__init__()
        self.fail = set(fail)
        self.n = 0
        self.joint_names = self.obs_joint_names = ("j0", "j1", "j2")
        self.accepts = (ReferenceKind.VELOCITY,)
        self.stand_command = np.zeros(3, np.float32)

    def forward(self, obs):
        self.n += 1
        if self.n in self.fail:
            raise ForwardTimeout("stub timeout")
        return np.full(3, 0.1 * self.n, np.float32)


def _wbc(ctrl, **kw):
    from hirobot import make_robot

    from hiwbc.robot import WbcRobot

    body = {"device": "body", "control_hz": 50, "joints": [{"name": f"j{i}"} for i in range(3)],
            "driver": {"type": "toy"}}
    inner = make_robot({"assembly": {"name": "toy", "parts": {"body": body}}})
    robot = WbcRobot(inner, controller=ctrl, start_seconds=0.0, **kw)
    robot.connect()
    return robot, inner


def _vel(robot):
    from hirobot import Action, ControlMode

    robot.send_action(Action(mode=ControlMode.BASE_VELOCITY, base_velocity=np.zeros(3)))


def test_timed_out_tick_repeats_the_last_target():
    ctrl = _Flaky(fail={2})
    robot, inner = _wbc(ctrl, max_missed_ticks=2)
    _vel(robot)
    np.testing.assert_allclose(inner.get_observation().joint_pos, 0.1, atol=1e-6)
    _vel(robot)                                               # tick 2 times out
    np.testing.assert_allclose(inner.get_observation().joint_pos, 0.1, atol=1e-6)
    _vel(robot)
    np.testing.assert_allclose(inner.get_observation().joint_pos, 0.3, atol=1e-6)
    st = robot.get_status()
    assert st.fault is None and st.extra["wbc"]["missed_ticks"] == 1


def test_a_good_tick_resets_the_run_of_misses():
    ctrl = _Flaky(fail={2, 3, 5, 6, 8, 9})            # 2 in a row, three times
    robot, _ = _wbc(ctrl, max_missed_ticks=2)
    for _ in range(10):
        _vel(robot)
    st = robot.get_status()
    assert st.fault is None and st.extra["wbc"]["missed_ticks"] == 6


def test_too_many_timeouts_in_a_row_damp():
    from hirobot import RobotFaultError

    ctrl = _Flaky(fail={2, 3, 4})
    robot, _ = _wbc(ctrl, max_missed_ticks=2)
    _vel(robot)
    _vel(robot)
    _vel(robot)                                               # 2 misses: still holding
    with pytest.raises(RobotFaultError, match="timed out 3"):
        _vel(robot)
    assert "timed out" in robot.get_status().fault


def test_first_tick_timeout_damps_with_no_target_to_repeat():
    from hirobot import RobotFaultError

    robot, _ = _wbc(_Flaky(fail={1}))
    with pytest.raises(RobotFaultError, match="timed out 1"):
        _vel(robot)

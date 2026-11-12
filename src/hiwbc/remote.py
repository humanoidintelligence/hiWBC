"""Remote model forward: the 50 Hz loop stays on the robot, the models run elsewhere.

The robot builds its controller as usual, then :func:`make_remote` swaps each
injected model callable (``Controller._model_fns()``, plus SONIC's planner) for a
:class:`RemoteFn` that sends the call to ``python -m hiwbc.serve`` on a workstation.
Observation assembly, history, post-processing and the safety rules stay local;
the server only runs the forward pass, so it keeps no state between calls.

Wire format: TCP, each message ``[u32 length][u32 header length][JSON header][array bytes]``.
Arrays are referenced from the JSON as ``{"__nd__": i}`` and carried raw after the
header (``dtype`` from a fixed list, C order). Nothing is unpickled. The first
message on a connection is a ``hello`` naming the profile, variant and model
function; the server refuses one that does not match what it loaded.

The service has no authentication. Bind it to a trusted wired network
(``--host 127.0.0.1`` is the default).
"""

from __future__ import annotations

import json
import socket
import struct
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np

PROTOCOL = 1
#: Largest message either side accepts (the biggest SONIC input is ~7 KB).
MAX_MESSAGE = 16 * 1024 * 1024
_DTYPES = ("float32", "float64", "int64", "int32", "bool")
_U32 = struct.Struct(">I")


class ForwardTimeout(TimeoutError):
    """A remote forward pass did not return within its timeout.

    ``WbcRobot`` repeats the last joint target for that tick and damps after
    ``max_missed_ticks`` in a row.
    """


class RemoteError(RuntimeError):
    """The server refused the connection or the call failed on the server."""


# ----- encoding ---------------------------------------------------------------
def _pack(value: Any, arrays: list[np.ndarray]) -> Any:
    if isinstance(value, (np.ndarray, np.generic)):
        arr = np.ascontiguousarray(value)
        if arr.dtype.name not in _DTYPES:
            arr = arr.astype(np.float32) if arr.dtype.kind == "f" else arr.astype(np.int64)
        arrays.append(arr)
        return {"__nd__": len(arrays) - 1}
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            raise TypeError("only str dict keys can be sent")
        return {"__map__": {k: _pack(v, arrays) for k, v in value.items()}}
    if isinstance(value, tuple):   # ONNX runners take a tuple for several inputs
        return {"__tuple__": [_pack(v, arrays) for v in value]}
    if isinstance(value, list):
        return [_pack(v, arrays) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"cannot send a {type(value).__name__}")


def encode(message: dict) -> bytes:
    """One framed message; ``message`` holds JSON values, numpy arrays, dicts and lists."""
    arrays: list[np.ndarray] = []
    body = _pack(message, arrays)
    header = {"body": body, "arrays": [[a.dtype.name, list(a.shape)] for a in arrays]}
    head = json.dumps(header, separators=(",", ":")).encode()
    payload = b"".join([_U32.pack(len(head)), head, *(a.tobytes() for a in arrays)])
    if len(payload) > MAX_MESSAGE:
        raise ValueError(f"message of {len(payload)} bytes exceeds {MAX_MESSAGE}")
    return _U32.pack(len(payload)) + payload


def _unpack(value: Any, arrays: list[np.ndarray]) -> Any:
    if isinstance(value, dict):
        if set(value) == {"__nd__"}:
            i = value["__nd__"]
            if not isinstance(i, int) or not 0 <= i < len(arrays):
                raise ValueError("bad array reference")
            return arrays[i]
        if set(value) == {"__map__"} and isinstance(value["__map__"], dict):
            return {k: _unpack(v, arrays) for k, v in value["__map__"].items()}
        if set(value) == {"__tuple__"} and isinstance(value["__tuple__"], list):
            return tuple(_unpack(v, arrays) for v in value["__tuple__"])
        raise ValueError("bad object in message")
    if isinstance(value, list):
        return [_unpack(v, arrays) for v in value]
    return value


def decode(payload: bytes) -> dict:
    """Inverse of :func:`encode` (without the outer length); malformed input raises ``ValueError``."""
    try:
        (n,) = _U32.unpack_from(payload, 0)
        header = json.loads(payload[4:4 + n])
        specs = header["arrays"]
        arrays, off = [], 4 + n
        for dtype, shape in specs:
            if dtype not in _DTYPES or not all(isinstance(d, int) and d >= 0 for d in shape):
                raise ValueError(f"bad array spec {dtype} {shape}")
            size = int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize
            if off + size > len(payload):
                raise ValueError("message shorter than its arrays")
            arrays.append(np.frombuffer(payload, dtype=dtype, count=size // np.dtype(dtype).itemsize,
                                        offset=off).reshape(shape).copy())
            off += size
        if off != len(payload):
            raise ValueError("trailing bytes after the arrays")
        message = _unpack(header["body"], arrays)
    except (KeyError, TypeError, struct.error, json.JSONDecodeError) as e:
        raise ValueError(f"malformed message: {e}") from None
    if not isinstance(message, dict):
        raise ValueError("message is not an object")  # noqa: TRY004 - malformed input is ValueError
    return message


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("connection closed")
        buf += chunk
    return bytes(buf)


def recv(sock: socket.socket) -> dict:
    (n,) = _U32.unpack(_recv_exact(sock, 4))
    if n > MAX_MESSAGE:
        raise ValueError(f"message of {n} bytes exceeds {MAX_MESSAGE}")
    return decode(_recv_exact(sock, n))


def parse_endpoint(endpoint: str) -> tuple[str, int]:
    """``tcp://host:port`` (or ``host:port``) -> ``(host, port)``."""
    rest = endpoint.removeprefix("tcp://")
    host, sep, port = rest.rpartition(":")
    if not sep or not host or not port.isdigit():
        raise ValueError(f"endpoint {endpoint!r} must be tcp://<host>:<port>")
    return host.strip("[]"), int(port)


def send(sock: socket.socket, message: dict) -> None:
    sock.sendall(encode(message))


# ----- which callables are "the model" ------------------------------------------
def model_fns(controller) -> dict[str, Callable]:
    """The controller's model callables by name, the same on robot and server.

    ``Controller._model_fns()`` (attributes ending in ``_fn``) plus a SONIC
    planner's ``planner_fn`` as ``"planner"``.
    """
    fns = {k: v for k, v in vars(controller).items() if k.endswith("_fn") and callable(v)}
    planner = getattr(controller, "planner", None)
    if planner is not None and callable(getattr(planner, "_fn", None)):
        fns["planner"] = planner._fn
    return fns


def _set_fn(controller, name: str, fn: Callable) -> None:
    if name == "planner":
        controller.planner._fn = fn
    else:
        setattr(controller, name, fn)


# ----- client -----------------------------------------------------------------
class RemoteFn:
    """A model callable that runs on ``hiwbc.serve``: ``fn(*args)`` -> the server's ``fn(*args)``.

    Each call has ``timeout_s`` for the round trip, connecting included. A call
    that times out or finds the server gone raises :class:`ForwardTimeout` and
    drops the connection; the next call reconnects. An error the server reports
    raises :class:`RemoteError`.
    """

    def __init__(self, endpoint: str, *, name: str, profile: str, variant: str,
                 timeout_s: float, connect_timeout_s: float = 2.0) -> None:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        self.endpoint = endpoint
        self.address = parse_endpoint(endpoint)
        self.name, self.profile, self.variant = name, profile, variant
        self.timeout_s = float(timeout_s)
        self.connect_timeout_s = float(connect_timeout_s)
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()
        self.calls = 0
        self.reconnects = 0

    def connect(self, timeout_s: float | None = None) -> None:
        """Open the connection and check that the server runs the same profile and variant."""
        budget = self.connect_timeout_s if timeout_s is None else timeout_s
        deadline = time.monotonic() + budget
        sock = socket.create_connection(self.address, timeout=budget)
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(max(deadline - time.monotonic(), 1e-3))
            send(sock, {"op": "hello", "protocol": PROTOCOL, "profile": self.profile,
                        "variant": self.variant, "fn": self.name})
            reply = recv(sock)
        except BaseException:
            sock.close()
            raise
        if not reply.get("ok"):
            sock.close()
            raise RemoteError(f"{self.endpoint} refused {self.profile}:{self.variant} "
                              f"{self.name}: {reply.get('error')}")
        self._sock = sock

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def __call__(self, *args):
        with self._lock:
            start = time.monotonic()
            try:
                if self._sock is None:
                    self.reconnects += 1
                    self.connect(self.timeout_s)
                self._sock.settimeout(max(self.timeout_s - (time.monotonic() - start), 1e-4))
                send(self._sock, {"op": "call", "args": list(args)})
                reply = recv(self._sock)
            except RemoteError:
                self.close()
                raise
            except (OSError, ConnectionError, ValueError) as e:
                # socket.timeout is an OSError. A late reply would arrive on this
                # socket next call, so drop it and reconnect.
                self.close()
                raise ForwardTimeout(f"{self.name} @ {self.endpoint}: {type(e).__name__}: {e}") from e
            self.calls += 1
        if not reply.get("ok"):
            raise RemoteError(f"{self.name} @ {self.endpoint}: {reply.get('error')}")
        return reply["result"]


def make_remote(controller, endpoint: str, *, timeout_s: float = 0.015,
                planner_timeout_s: float = 0.5, connect: bool = True) -> dict[str, RemoteFn]:
    """Swap ``controller``'s model callables for :class:`RemoteFn` s to ``endpoint``.

    ``controller`` must come from ``hiwbc.load`` (its ``profile`` / ``variant``
    are checked by the server). ``timeout_s`` bounds each per-tick call;
    ``planner_timeout_s`` the SONIC planner (it runs at 10 Hz and takes ~35 ms on
    a CPU). Recurrent models (a TorchScript export with buffers, e.g.
    ``unitree_loco`` ``default``) are refused: the server keeps no state.
    """
    profile, variant = getattr(controller, "profile", None), getattr(controller, "variant", None)
    if profile is None:
        raise ValueError("make_remote needs a controller built by hiwbc.load (no profile set)")
    fns = model_fns(controller)
    if not fns:
        raise ValueError(f"{profile} has no model callables to run remotely")
    stateful = [k for k, f in fns.items() if getattr(f, "stateful", False)]
    if stateful:
        raise ValueError(f"{profile}:{variant} has a recurrent model ({stateful}); the remote "
                         "service is stateless, so run it locally")
    remote = {}
    for name in fns:
        t = planner_timeout_s if name == "planner" else timeout_s
        r = RemoteFn(endpoint, name=name, profile=profile, variant=variant, timeout_s=t)
        if connect:
            r.connect()
        remote[name] = r
    for name, r in remote.items():
        _set_fn(controller, name, r)
    controller.remote_fns = remote
    return remote

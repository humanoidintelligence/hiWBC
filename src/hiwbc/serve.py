"""Model forward service for ``runtime="remote"``.

    python -m hiwbc.serve --profile sonic/unitree_g1 [--variant default] \\
        [--host 127.0.0.1] [--port 5555] [--option providers=CUDAExecutionProvider]

Loads the profile with :func:`hiwbc.load` and runs only its model callables
(:func:`hiwbc.remote.model_fns`); the robot builds observations and applies the
actions. One thread per connection; each model callable runs under its own lock.
The service keeps no state between calls, so a robot can reconnect at any time.

There is no authentication: bind to localhost or a trusted wired network.
"""

from __future__ import annotations

import argparse
import ast
import logging
import socket
import threading
from collections.abc import Callable, Sequence

from . import remote
from .registry import load

logger = logging.getLogger("hiwbc.serve")


class ModelServer:
    """Serve ``controller``'s model callables on ``host:port``.

    Args:
        controller: Built by :func:`hiwbc.load` (``profile`` / ``variant`` set).
        host / port: Where to listen; port 0 picks a free one (see :attr:`port`).
    """

    def __init__(self, controller, host: str = "127.0.0.1", port: int = 5555) -> None:
        self.profile = controller.profile
        self.variant = controller.variant
        self.fns: dict[str, Callable] = remote.model_fns(controller)
        stateful = [k for k, f in self.fns.items() if getattr(f, "stateful", False)]
        if stateful:
            raise ValueError(f"{self.profile}:{self.variant} has a recurrent model ({stateful}); "
                             "the service is stateless and cannot serve it")
        self._locks = {k: threading.Lock() for k in self.fns}
        self._sock = socket.create_server((host, port))
        self.host, self.port = self._sock.getsockname()[:2]
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._conns: set[socket.socket] = set()
        self._conns_lock = threading.Lock()

    def start(self) -> ModelServer:
        """Accept connections on a background thread."""
        self._thread = threading.Thread(target=self.serve_forever, name="hiwbc-serve", daemon=True)
        self._thread.start()
        return self

    def serve_forever(self) -> None:
        logger.info("serving %s:%s %s on %s:%d", self.profile, self.variant,
                    sorted(self.fns), self.host, self.port)
        while not self._stop.is_set():
            try:
                conn, addr = self._sock.accept()
            except OSError:
                break   # closed by stop()
            if self._stop.is_set():
                conn.close()
                break
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with self._conns_lock:
                self._conns.add(conn)
            threading.Thread(target=self._handle, args=(conn, addr), daemon=True).start()

    def stop(self) -> None:
        """Stop accepting and close every open connection."""
        self._stop.set()
        try:
            # close() alone does not wake a thread blocked in accept() on Linux
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._sock.close()
        with self._conns_lock:
            conns, self._conns = list(self._conns), set()
        for c in conns:
            try:
                c.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            c.close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _hello(self, msg: dict) -> tuple[str | None, str | None]:
        """``(fn name, None)`` if ``msg`` is an acceptable hello, else ``(None, reason)``."""
        if msg.get("op") != "hello":
            return None, "the first message must be hello"
        if msg.get("protocol") != remote.PROTOCOL:
            return None, f"protocol {msg.get('protocol')} != {remote.PROTOCOL}"
        if (msg.get("profile"), msg.get("variant")) != (self.profile, self.variant):
            return None, (f"this server runs {self.profile}:{self.variant}, "
                          f"not {msg.get('profile')}:{msg.get('variant')}")
        name = msg.get("fn")
        if name not in self.fns:
            return None, f"no model {name!r}; this server has {sorted(self.fns)}"
        return name, None

    def _handle(self, conn: socket.socket, addr) -> None:
        try:
            name, err = self._hello(remote.recv(conn))
            remote.send(conn, {"ok": err is None, "error": err})
            if name is None:
                logger.warning("refused %s: %s", addr, err)
                return
            fn, lock = self.fns[name], self._locks[name]
            while True:
                msg = remote.recv(conn)
                if msg.get("op") != "call" or not isinstance(msg.get("args"), list):
                    remote.send(conn, {"ok": False, "error": "expected a call"})
                    return
                try:
                    with lock:
                        result = fn(*msg["args"])
                    reply = {"ok": True, "result": result}
                except Exception as e:  # report to the robot, keep serving
                    logger.exception("%s failed", name)
                    reply = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                remote.send(conn, reply)
        except (ConnectionError, OSError):
            pass   # the robot hung up or timed out; it reconnects on its next call
        except ValueError as e:
            logger.warning("bad message from %s: %s", addr, e)
        finally:
            with self._conns_lock:
                self._conns.discard(conn)
            conn.close()


def _option(text: str) -> tuple[str, object]:
    key, sep, value = text.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError(f"--option {text!r} must be key=value")
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        parsed = value   # a bare string
    if key == "providers" and isinstance(parsed, str):
        parsed = [parsed]
    return key, parsed


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m hiwbc.serve", description=__doc__.split("\n\n")[0])
    ap.add_argument("--profile", required=True, help="e.g. sonic/unitree_g1")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to accept from the robot")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--option", type=_option, action="append", default=[],
                    help="profile option key=value (repeatable), e.g. checkpoint_dir=/data/sonic")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ctrl = load(args.profile, variant=args.variant, **dict(args.option))
    server = ModelServer(ctrl, args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()

"""SONIC's locomotion planner: base-velocity commands to a reference motion SONIC tracks.

A port of the GR00T C++ deployment's planner path (gear_sonic_deploy,
``localmotion_kplanner.hpp`` and the planner parts of ``g1_deploy_onnx_ref.cpp``).
``planner_sonic.onnx`` takes the last 4 frames of the current motion (30 Hz MuJoCo
qpos) and a movement command (mode, speed, movement / facing direction, height),
and predicts up to 64 frames of new motion. The deployment then:

- resamples the prediction to 50 Hz (``ResampleGeneratedSequence50Hz``);
- blends it into the motion being played, starting ``look_ahead`` frames after
  the frame the context was taken at, linearly over 8 frames
  (``CurrentFrameAdvancement``);
- holds the last frame when the motion runs out, and in IDLE pulls the lower-body
  reference toward the measured joints when they drift apart (idle readaptation);
- replans when the command changes, and every 0.1-1 s while moving.

Quaternions here are [w, x, y, z] (the C++ convention); :meth:`SonicPlanner.frames`
converts to hiWBC's [x, y, z, w]. Joint vectors are in the controller's IsaacLab
order; ``qpos_index`` maps them into the planner's MuJoCo-order qpos.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from enum import IntEnum

import numpy as np

logger = logging.getLogger(__name__)


class LocomotionMode(IntEnum):
    """Planner modes, as ``LocomotionMode`` in ``localmotion_kplanner.hpp`` (V2: 27 modes)."""

    IDLE = 0
    SLOW_WALK = 1     # 0.1-0.8 m/s
    WALK = 2          # 0.8-1.5 m/s on the gamepad
    RUN = 3           # 1.5-3.0 m/s on the gamepad
    IDLE_SQUAT = 4
    IDLE_KNEEL_TWO_LEGS = 5
    IDLE_KNEEL = 6
    IDLE_LYING_FACE_DOWN = 7
    CRAWLING = 8
    IDLE_BOXING = 9
    WALK_BOXING = 10
    LEFT_PUNCH = 11
    RIGHT_PUNCH = 12
    RANDOM_PUNCH = 13
    ELBOW_CRAWLING = 14
    LEFT_HOOK = 15
    RIGHT_HOOK = 16
    FORWARD_JUMP = 17
    STEALTH_WALK = 18
    INJURED_WALK = 19
    LEDGE_WALKING = 20
    OBJECT_CARRYING = 21
    STEALTH_WALK_2 = 22
    HAPPY_DANCE_WALK = 23
    ZOMBIE_WALK = 24
    GUN_WALK = 25
    SCARE_WALK = 26


#: Modes the deployment treats as standing in place (``is_static_motion_mode``).
STATIC_MODES = frozenset({LocomotionMode.IDLE, LocomotionMode.IDLE_SQUAT,
                          LocomotionMode.IDLE_KNEEL_TWO_LEGS, LocomotionMode.IDLE_KNEEL,
                          LocomotionMode.IDLE_LYING_FACE_DOWN, LocomotionMode.IDLE_BOXING})
_BOXING = frozenset({LocomotionMode.LEFT_PUNCH, LocomotionMode.RIGHT_PUNCH,
                     LocomotionMode.RANDOM_PUNCH, LocomotionMode.LEFT_HOOK,
                     LocomotionMode.RIGHT_HOOK})

_NUM_QPOS = 36            # 7 root (xyz + wxyz) + 29 joints
_CONTEXT_FRAMES = 4
_MAX_PRED_FRAMES = 64
_BLEND_FRAMES = 8         # "about the size of the context we use for conditioning"
# Idle readaptation thresholds (rad, mean abs lower-body error).
_ADAPT_TRIGGER, _ADAPT_STOP, _RECOVER_TRIGGER = 0.10, 0.05, 0.045
_REPLAN_PERIOD = 0.1      # the deployment's planner thread runs at 10 Hz


# ----- quaternion helpers, [w, x, y, z] as in the C++ ------------------------------
def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(np.asarray(a, np.float64), -1, 0)
    bw, bx, by, bz = np.moveaxis(np.asarray(b, np.float64), -1, 0)
    return np.stack([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw], axis=-1)


def _heading(q: np.ndarray) -> np.ndarray:
    """``calc_heading_quat_d``: keep w and z, renormalize."""
    h = np.zeros_like(np.asarray(q, np.float64))
    h[..., 0], h[..., 3] = q[..., 0], q[..., 3]
    return h / np.maximum(np.linalg.norm(h, axis=-1, keepdims=True), 1e-9)


def _conj(q: np.ndarray) -> np.ndarray:
    return np.asarray(q, np.float64) * np.array([1.0, -1.0, -1.0, -1.0])


def _slerp(q0: np.ndarray, q1: np.ndarray, t) -> np.ndarray:
    """``quat_slerp_d``, row-wise: shorter path, normalized lerp when nearly equal."""
    q0 = np.atleast_2d(np.asarray(q0, np.float64))
    q1 = np.atleast_2d(np.asarray(q1, np.float64)).copy()
    t = np.broadcast_to(np.asarray(t, np.float64), (len(q0),))[:, None]
    dot = np.sum(q0 * q1, axis=1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.abs(dot)
    lerp = q0 + t * (q1 - q0)
    lerp /= np.maximum(np.linalg.norm(lerp, axis=1, keepdims=True), 1e-12)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    s = np.sin(theta)
    s = np.where(s == 0.0, 1.0, s)
    sl = (np.sin((1.0 - t) * theta) / s) * q0 + (np.sin(t * theta) / s) * q1
    return np.where(dot > 0.9995, lerp, sl)


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def _yaw(q: np.ndarray) -> float:
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


class Motion:
    """A 50 Hz planner motion: root position, root quaternion [w, x, y, z], joints."""

    def __init__(self, root_pos, root_quat, joint_pos, joint_vel) -> None:
        self.root_pos = np.asarray(root_pos, np.float64)
        self.root_quat = np.asarray(root_quat, np.float64)
        self.joint_pos = np.asarray(joint_pos, np.float64)
        self.joint_vel = np.asarray(joint_vel, np.float64)

    def __len__(self) -> int:
        return len(self.joint_pos)


class SonicPlanner:
    """The deployment's planner state machine, one 50 Hz tick at a time.

    Args:
        planner_fn: Runs ``planner_sonic.onnx``: a dict of input arrays to
            ``(mujoco_qpos (1, 64, 36), num_pred_frames (1,))``.
        qpos_joint_index: For each controller joint, its index among the
            planner's 29 MuJoCo-order qpos joints.
        lower_body_index: Controller indices of the leg joints (idle readaptation).
        default_height: Root height of the initial context.
        control_dt: Tick length (0.02 s).
        look_ahead: Frames after the current one where a new plan starts.
        seed: The planner's random seed (the deployment never changes it).
        background: Run replans on a worker thread, as the deployment's planner
            thread does, so a 35 ms CPU inference does not stall the 50 Hz tick.
            The plan is blended in at the first tick after it finishes. Off,
            a replan runs inside :meth:`advance` (deterministic, for simulation).
    """

    def __init__(self, planner_fn: Callable[[dict], Sequence[np.ndarray]], *,
                 qpos_joint_index: Sequence[int], lower_body_index: Sequence[int],
                 default_height: float = 0.788740, control_dt: float = 0.02,
                 look_ahead: int = 2, seed: int = 1234, background: bool = False) -> None:
        self._fn = planner_fn
        self._executor = ThreadPoolExecutor(1, thread_name_prefix="sonic-planner") if background else None
        self._future: Future | None = None
        self.qpos_joint_index = np.asarray(qpos_joint_index, dtype=int)
        n = len(self.qpos_joint_index)
        if n != _NUM_QPOS - 7 or sorted(self.qpos_joint_index) != list(range(n)):
            raise ValueError("qpos_joint_index must be a permutation of the planner's 29 joints")
        self.lower_body_index = np.asarray(lower_body_index, dtype=int)
        self.default_height = float(default_height)
        self.dt = float(control_dt)
        self.look_ahead = int(look_ahead)
        self.seed = int(seed)
        self._check_every = max(1, round(_REPLAN_PERIOD / self.dt))
        self.reset()

    def reset(self) -> None:
        """Forget the motion; the next :meth:`start` re-initializes from the robot."""
        if getattr(self, "_future", None) is not None:
            self._future.cancel()
            try:
                self._future.result()   # an inference already running finishes first
            except Exception:
                logger.warning("discarded SONIC replan had failed", exc_info=True)
        self._future = None
        self.motion: Motion | None = None
        self.frame = 0
        self.command = None           # the command the current motion was planned for
        self._pending = None          # the command to plan for at the next check
        self._ticks = 0
        self._replan_timer = 0.0
        self._readapt_stored = False
        self._readapt_state = "idle"
        self._readapt_targets = None
        self.anchor = np.array([1.0, 0.0, 0.0, 0.0])   # planner frame -> world (yaw only)
        self.replans = 0
        self.facing_yaw = 0.0         # commanded facing, planner frame
        self._reanchor = False

    # ----- the model ------------------------------------------------------------
    def _feeds(self, context: np.ndarray, cmd: tuple) -> dict:
        mode, speed, mdir, face, height = cmd
        return {
            "context_mujoco_qpos": context.astype(np.float32)[None],
            "target_vel": np.array([speed], np.float32),
            "mode": np.array([mode], np.int64),
            "movement_direction": np.array([mdir], np.float32),
            "facing_direction": np.array([face], np.float32),
            "random_seed": np.array([self.seed], np.int64),
            "has_specific_target": np.zeros((1, 1), np.int64),
            "specific_target_positions": np.zeros((1, 4, 3), np.float32),
            "specific_target_headings": np.zeros((1, 4), np.float32),
            # LocalMotionPlannerTensorRT (what the deployment runs): 9-11 tokens allowed.
            "allowed_pred_num_tokens": np.array([[0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]], np.int64),
            "height": np.array([height], np.float32),
        }

    def _plan(self, context: np.ndarray, cmd: tuple) -> Motion:
        """Run the model and resample its 30 Hz prediction to 50 Hz (``ResampleGeneratedSequence50Hz``)."""
        qpos, num = self._fn(self._feeds(context, cmd))
        n = int(np.asarray(num).reshape(-1)[0])
        if not 2 <= n <= _MAX_PRED_FRAMES:
            raise RuntimeError(f"the SONIC planner predicted {n} frames")
        q = np.asarray(qpos, np.float64).reshape(-1, _NUM_QPOS)[:n]
        if not np.isfinite(q).all():
            raise RuntimeError("the SONIC planner predicted NaN")
        steps = int(np.floor(n / 30.0 * 50))
        f30 = np.arange(steps) / 50.0 * 30
        f0 = np.floor(f30).astype(int)
        f1 = np.minimum(f0 + 1, n - 1)
        a = (f30 - f0)[:, None]
        pos = (1 - a) * q[f0, :3] + a * q[f1, :3]
        quat = _slerp(q[f0, 3:7], q[f1, 3:7], a[:, 0])
        joints = ((1 - a) * q[f0, 7:] + a * q[f1, 7:])[:, self.qpos_joint_index]
        vel = np.empty_like(joints)
        vel[:-1] = (joints[1:] - joints[:-1]) / self.dt
        vel[-1] = vel[-2]
        return Motion(pos, quat, joints, vel)

    def _context(self, gen_frame: int) -> np.ndarray:
        """Four 30 Hz frames of the current motion from ``gen_frame`` on (``UpdateContextFromMotion``)."""
        m = self.motion
        t = gen_frame * self.dt + np.arange(_CONTEXT_FRAMES) / 30.0
        f = t / self.dt
        f0 = np.minimum(np.floor(f).astype(int), len(m) - 1)
        f1 = np.minimum(f0 + 1, len(m) - 1)
        a = (f - f0)[:, None]
        ctx = np.zeros((_CONTEXT_FRAMES, _NUM_QPOS))
        ctx[:, :3] = (1 - a) * m.root_pos[f0] + a * m.root_pos[f1]
        ctx[:, 3:7] = _slerp(m.root_quat[f0], m.root_quat[f1], a[:, 0])
        ctx[:, 7 + self.qpos_joint_index] = (1 - a) * m.joint_pos[f0] + a * m.joint_pos[f1]
        return ctx

    # ----- lifecycle ------------------------------------------------------------
    @property
    def started(self) -> bool:
        return self.motion is not None

    def start(self, joint_pos: np.ndarray) -> None:
        """First plan, from the measured joints standing at the origin (``Initialize``).

        The context is the robot's joints at ``default_height`` with zero yaw, so
        the planner frame's +x is the robot's heading; :meth:`window` re-anchors
        it to the robot at the next call.
        """
        self.reset()
        ctx = np.zeros((_CONTEXT_FRAMES, _NUM_QPOS))
        ctx[:, 2] = self.default_height
        ctx[:, 3] = 1.0
        ctx[:, 7 + self.qpos_joint_index] = np.asarray(joint_pos, np.float64)
        self.command = self.idle_command()
        self.motion = self._plan(ctx, self.command)
        self.frame = 0
        self._reanchor = True

    # ----- commands -------------------------------------------------------------
    @staticmethod
    def idle_command(facing_yaw: float = 0.0) -> tuple:
        """Stand in place facing ``facing_yaw`` (planner frame), as the gamepad's dead zone sends."""
        return (int(LocomotionMode.IDLE), -1.0, (0.0, 0.0, 0.0),
                (float(np.cos(facing_yaw)), float(np.sin(facing_yaw)), 0.0), -1.0)

    def set_command(self, mode: int, speed: float, movement_yaw: float | None,
                    facing_yaw: float, height: float = -1.0) -> None:
        """Request a plan: ``MovementState`` (mode, speed, movement / facing direction, height).

        Directions are yaw angles in the planner frame; ``movement_yaw=None`` is no
        movement. Takes effect at the next 10 Hz check, as in the deployment.
        """
        mdir = (0.0, 0.0, 0.0) if movement_yaw is None else (
            float(np.cos(movement_yaw)), float(np.sin(movement_yaw)), 0.0)
        face = (float(np.cos(facing_yaw)), float(np.sin(facing_yaw)), 0.0)
        self.facing_yaw = float(facing_yaw)
        self._pending = (int(mode), float(speed), mdir, face, float(height))

    def _replan_interval(self, mode: int) -> float:
        if mode == LocomotionMode.RUN:
            return 0.1
        if mode == LocomotionMode.CRAWLING:
            return 0.2
        return 1.0   # boxing and everything else (replan_interval_boxing_ / replan_interval_)

    def _need_replan(self, cmd: tuple) -> bool:
        """The deployment's ``Planner()`` decision, checked every 0.1 s."""
        last = self.command
        self._replan_timer += _REPLAN_PERIOD
        timed = self._replan_timer >= self._replan_interval(cmd[0]) - 1e-9
        if timed:
            self._replan_timer = 0.0
        if cmd[0] != last[0] or cmd[3] != last[3] or cmd[4] != last[4]:
            return True
        static = LocomotionMode(cmd[0]) in STATIC_MODES
        return not static and (cmd[1] != last[1] or cmd[2] != last[2] or (timed and cmd[1] != 0))

    # ----- the 50 Hz tick -------------------------------------------------------
    def advance(self, joint_pos: np.ndarray) -> None:
        """One control tick after the policy ran (``CurrentFrameAdvancement`` + the 10 Hz check).

        ``joint_pos`` is the measured joints (controller order), for idle readaptation.
        """
        if self.motion is None:
            raise RuntimeError("SonicPlanner.start() was not called")
        self._ticks += 1
        if self._future is not None and self._future.done():
            future, self._future = self._future, None
            try:
                new, gen = future.result()   # a failed inference raises here
            except TimeoutError as e:        # a remote planner (hiwbc.remote.ForwardTimeout)
                # Keep playing the current motion and replan at the next check.
                logger.warning("SONIC replan timed out, retrying: %s", e)
                self.command, self.replans = self._submitted_from, self.replans - 1
            else:
                self._blend(new, gen)
        if (self._future is None and self._ticks % self._check_every == 0
                and self._pending is not None and self._need_replan(self._pending)):
            self._replan(self._pending)
        # play: next frame, clamp at the end
        m = self.motion
        f = self.frame + 1
        if f >= len(m):
            f = len(m) - 1
            if self.command[0] == LocomotionMode.IDLE:
                self._readapt(f, np.asarray(joint_pos, np.float64))
        self.frame = f

    def _replan(self, cmd: tuple) -> None:
        """``UpdatePlanning`` from the current frame, then blend the result in."""
        gen = self.frame + self.look_ahead
        context = self._context(gen)
        if self._executor is None:
            # Record the command only once the plan exists: a plan that fails (a
            # remote timeout) is retried at the next check.
            new = self._plan(context, cmd)
            self.command = cmd
            self.replans += 1
            self._blend(new, gen)
        else:
            self._submitted_from = self.command
            self.command = cmd
            self.replans += 1
            self._future = self._executor.submit(lambda: (self._plan(context, cmd), gen))

    def _blend(self, new: Motion, fgen: int) -> None:
        """Rebase the motion to the current frame and fade ``new`` in over 8 frames from ``fgen``."""
        old, cur = self.motion, self.frame
        length = fgen - cur + len(new)
        if length <= 0:
            return
        f = np.arange(length)
        f_old = np.clip(f + cur, 0, len(old) - 1)
        f_new = np.clip(f + cur - fgen, 0, len(new) - 1)
        w = np.clip((f - max(0, fgen - cur)) / _BLEND_FRAMES, 0.0, 1.0)[:, None]
        self.motion = Motion(
            (1 - w) * old.root_pos[f_old] + w * new.root_pos[f_new],
            _slerp(old.root_quat[f_old], new.root_quat[f_new], w[:, 0]),
            (1 - w) * old.joint_pos[f_old] + w * new.joint_pos[f_new],
            (1 - w) * old.joint_vel[f_old] + w * new.joint_vel[f_new])
        self.frame = 0
        self._readapt_stored = False

    def _readapt(self, f: int, joint_pos: np.ndarray) -> None:
        """Idle readaptation at the held last frame (double-threshold state machine)."""
        idx = self.lower_body_index
        q = self.motion.joint_pos[f]
        if not self._readapt_stored:
            self._readapt_targets = q[idx].copy()
            self._readapt_stored = True
            self._readapt_state = "idle"
        err = float(np.mean(np.abs(q[idx] - joint_pos[idx])))
        s = self._readapt_state
        if s == "idle":
            s = "adapting" if err > _ADAPT_TRIGGER else "recovering" if err < _RECOVER_TRIGGER else s
        elif s == "adapting" and err < _ADAPT_STOP:
            s = "idle"
        elif s == "recovering" and err > _ADAPT_TRIGGER:
            s = "adapting"
        self._readapt_state = s
        if s == "adapting":
            q[idx] = 0.98 * q[idx] + 0.02 * joint_pos[idx]
        elif s == "recovering":
            q[idx] = 0.98 * q[idx] + 0.02 * self._readapt_targets

    # ----- reading the motion ---------------------------------------------------
    def window(self, length: int, base_quat_wxyz: np.ndarray) -> dict[str, np.ndarray]:
        """``length`` frames from the current one (clamped at the end), in the world frame.

        The first call after :meth:`start` fixes the planner-to-world yaw so the
        motion's current heading matches the robot's (``reinitialize_heading_``:
        ``heading(base) * heading(motion)^-1``). Returns ``joint_pos``,
        ``joint_vel`` (controller order), ``root_quat`` [x, y, z, w], ``root_z``.
        """
        m = self.motion
        if self._reanchor:
            self.anchor = _qmul(_heading(np.asarray(base_quat_wxyz, np.float64)),
                                _conj(_heading(m.root_quat[self.frame])))
            self._reanchor = False
        idx = np.minimum(self.frame + np.arange(length), len(m) - 1)
        quat = _qmul(self.anchor, m.root_quat[idx])
        return {
            "joint_pos": m.joint_pos[idx].astype(np.float32),
            "joint_vel": m.joint_vel[idx].astype(np.float32),
            "root_quat": quat[:, [1, 2, 3, 0]].astype(np.float32),
            "root_z": m.root_pos[idx, 2].astype(np.float32),
        }

    def robot_yaw(self, base_quat_wxyz: np.ndarray) -> float:
        """The robot's yaw in the planner frame."""
        return _yaw(_qmul(_conj(self.anchor), np.asarray(base_quat_wxyz, np.float64)))


#: Speed bands of the walking modes (``LocomotionMode`` comments; RUN as the gamepad's 1.5 m/s).
_SLOW_WALK_MAX, _WALK_MAX, _MAX_SPEED = 0.8, 1.5, 3.0


def velocity_command(planner: SonicPlanner, velocity: np.ndarray, base_quat_wxyz: np.ndarray, *,
                     dt: float, stand_speed: float = 0.05, max_lead: float = np.pi / 2) -> None:
    """Turn ``[vx, vy, wz]`` (body frame) into a planner command, as the gamepad does.

    ``wz`` turns the commanded facing (the gamepad integrates its right stick the
    same way), held at most ``max_lead`` ahead of the robot's actual heading so a
    robot that lags behind does not wind up a large turn. ``[vx, vy]`` is taken
    relative to the commanded facing; its norm picks the mode: below
    ``stand_speed`` IDLE (turning in place), then SLOW_WALK, WALK, RUN.
    """
    vx, vy, wz = (float(v) for v in np.asarray(velocity, np.float64).reshape(3))
    face = planner.facing_yaw + wz * dt
    actual = planner.robot_yaw(base_quat_wxyz)
    lead = (face - actual + np.pi) % (2 * np.pi) - np.pi
    if abs(lead) > max_lead:
        face = actual + np.sign(lead) * max_lead
    speed = min(float(np.hypot(vx, vy)), _MAX_SPEED)
    if speed < stand_speed:
        planner.set_command(LocomotionMode.IDLE, -1.0, None, face)
        return
    mode = (LocomotionMode.SLOW_WALK if speed <= _SLOW_WALK_MAX else
            LocomotionMode.WALK if speed <= _WALK_MAX else LocomotionMode.RUN)
    planner.set_command(mode, speed, face + float(np.arctan2(vy, vx)), face)


def load_planner_runner(path: str, *, providers: Sequence[str] | None = None) -> Callable:
    """``planner_fn`` for :class:`SonicPlanner` from ``planner_sonic.onnx`` (onnxruntime).

    The model takes several typed inputs, so it does not fit the single-array
    ``runtime.load_onnx_runner``.
    """
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(path), providers=list(providers) if providers else ["CPUExecutionProvider"])
    inputs = {i.name for i in session.get_inputs()}
    outputs = [o.name for o in session.get_outputs()]
    expected = {"context_mujoco_qpos", "target_vel", "mode", "movement_direction",
                "facing_direction", "random_seed", "has_specific_target",
                "specific_target_positions", "specific_target_headings",
                "allowed_pred_num_tokens", "height"}
    if inputs != expected or outputs != ["mujoco_qpos", "num_pred_frames"]:
        raise ValueError(f"{path} is not the V2 SONIC planner: inputs {sorted(inputs)}, "
                         f"outputs {outputs}")

    def run(feeds: dict) -> list:
        return session.run(outputs, feeds)

    return run

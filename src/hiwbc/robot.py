"""``WbcRobot``: the hiRobot layer that runs a hiWBC controller.

``make_robot`` builds it from a ``layers: [{type: wbc, profile: ...}]`` entry as
``WbcRobot(inner, **entry)``. To the caller it is a plain ``Robot`` that takes
base velocities, motion latents and motion clips; underneath, every control tick
it reads ``inner``, runs the controller, and sends one PD action for all joints.

- **Start.** ``connect()`` / ``reset()`` ramp the controller's joints from the
  measured pose to ``ctrl.default_pos`` over ``start_seconds`` (GR00T
  ``InitControl``), then hand over to the controller at once.
- **Time.** A simulated ``inner`` (``inner.simulated``) advances only when the
  caller sends: each ``send_action`` / ``_idle_step`` runs the ticks of one outer
  period (``outer_hz``). A real one gets a thread at ``1 / control_dt``.
- **Reference per tick:** a running behavior, else a playing clip, else the last
  ``BASE_VELOCITY`` / ``MOTION_LATENT`` / ``CARTESIAN`` command (``timeout_s`` old
  at most), else a ``BASE_POSE`` goal being walked to, else standing still. A controller without a standing reference holds
  ``default_pos`` under PD instead.
- **Faults.** An exception in the tick (controller, safety gate underneath,
  driver) enters the fault state and e-stops the stack (``Layer.estop``: the
  robot at the bottom first, which damps a Unitree or MuJoCo body):
  ``send_action`` raises ``RobotFaultError`` until ``reset()``.
- **Layer skills.** ``stand`` and the controller's behaviors (BeyondMimic
  dances, ...) are layer skills: ``run_layer_skill`` holds the command stream
  until the behavior ends or ``stop_layer_skill``.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np
from hirobot import (
    Action,
    CommandProgress,
    ControlMode,
    DofMap,
    Layer,
    Observation,
    Robot,
    RobotFaultError,
    RobotSkillSpec,
    RobotStatus,
)

from .controller import Controller
from .references.clip import ClipPlayer
from .registry import load
from .remote import ForwardTimeout
from .types import ControllerState, Reference, ReferenceKind

logger = logging.getLogger(__name__)

STAND_SKILL = RobotSkillSpec(
    name="stand",
    description="Stop the current motion and stand still in place, keeping balance.",
    parameters={"type": "object", "properties": {}, "additionalProperties": False},
)


def _rotate_inv(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate world-frame ``v`` into the frame of unit quaternion ``q`` [x, y, z, w]."""
    u, w = -np.asarray(q[:3], float), float(q[3])
    t = 2.0 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


#: ``BASE_POSE`` controller: body-frame P control on the position
#: and yaw error, speed-limited; inside both tolerances the robot stands still.
GOTO_DEFAULTS = {
    "kp_xy": 1.0,         # 1/s: m/s per m of error
    "kp_yaw": 1.0,        # 1/s: rad/s per rad
    "max_speed": 0.6,     # m/s
    "max_yaw_rate": 0.6,  # rad/s
    "pos_tol": 0.10,      # m
    "yaw_tol": 0.15,      # rad
}


def _wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def _yaw_xyzw(q) -> float:
    x, y, z, w = (float(v) for v in q)
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _step_pose(cur: np.ndarray, goal: np.ndarray, max_d: float,
               max_a: float) -> tuple[np.ndarray, bool]:
    """``cur`` moved toward ``goal`` by at most ``max_d`` m and ``max_a`` rad.

    Poses are ``[x, y, z, qx, qy, qz, qw]`` with unit quaternions; the rotation
    follows the shorter slerp path. Returns the new pose and whether it is ``goal``.
    """
    out = np.empty(7)
    d = goal[:3] - cur[:3]
    dist = float(np.linalg.norm(d))
    eps = 1e-9   # steps of max_d accumulate rounding; land on the goal, not 1e-12 short
    out[:3] = goal[:3] if dist <= max_d + eps else cur[:3] + d * (max_d / dist)
    q0, q1 = cur[3:], goal[3:]
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1, dot = -q1, -dot
    angle = 2.0 * float(np.arccos(min(dot, 1.0)))
    reached = dist <= max_d + eps and angle <= max_a + eps
    if angle <= max_a + eps:
        out[3:] = q1
    else:
        t = max_a / angle
        half = angle / 2.0
        q = (np.sin((1 - t) * half) * q0 + np.sin(t * half) * q1) / np.sin(half)
        out[3:] = q / np.linalg.norm(q)
    return out, reached


class WbcRobot(Layer):
    """A hiRobot :class:`~hirobot.Layer` that balances the robot by running a hiWBC controller.

    Everything it does not change (cameras, audio, ``now()``, the vendor's onboard
    skills, status) is forwarded to ``inner`` by ``Layer``.

    Args:
        inner: The robot underneath: the assembled robot (body + hands), or one
            part of an assembly (``layers: [{type: wbc, part: ...}]``).
        profile: ``hiwbc.load`` profile name, e.g. ``"sonic/unitree_g1"``.
        controller: A built controller instead of ``profile`` (tests).
        variant: Profile variant.
        timeout_s: Stand still when no new ``BASE_VELOCITY`` / ``MOTION_LATENT``
            arrives for this long (simulated time in simulation).
        outer_hz: Simulation only: how often the caller sends. Each
            ``send_action`` runs ``1 / outer_hz`` seconds of ticks (rounded on
            the running total). Default ``1 / control_dt``: one tick per send.
        start_seconds: Ramp from the measured pose to ``default_pos`` at start.
        direct_kp / direct_kd: Gains for joints the controller does not drive
            (scalar or one per :attr:`direct_joints`). Needed when the
            controller sends its own gains and some joints are not its own;
            otherwise the driver's gains apply to every joint.
        forward_timeout_s / max_missed_ticks: A remote forward that times out
            repeats the last target for that tick; on hardware an in-process
            ``step`` over ``forward_timeout_s`` still sends its target and
            counts as a miss. More than ``max_missed_ticks`` misses in a row
            damps the robot.
        goto: ``BASE_POSE`` gains and limits, merged into
            :data:`GOTO_DEFAULTS`.
        cartesian_max_speed / cartesian_max_rate: ``CARTESIAN`` rate limit,
            m/s and rad/s. The target the controller tracks moves toward the
            commanded one at most this fast, starting from the controller's
            :meth:`~hiwbc.Controller.cartesian_rest` pose, and returns to it the
            same way when the command ends. A target jump in one tick makes
            SONIC command joint targets past the safety limit (15-20 cm faulted
            the MuJoCo G1).
        stale_s: Fault (and e-stop) when the body's joint state is older than
            this at a tick: ``inner.now() - obs.stamps["body"].t``, on the
            robot's clock (simulated time in simulation). ``None`` turns the
            check off. A body without a ``body`` stamp is not checked.
        **options: The profile's options (``checkpoint_dir``, ``device`` ...).
    """

    def __init__(self, inner: Robot, profile: str | None = None, *,
                 controller: Controller | None = None, variant: str | None = None,
                 timeout_s: float = 0.5, outer_hz: float | None = None,
                 start_seconds: float = 3.0, direct_kp=None, direct_kd=None,
                 forward_timeout_s: float = 0.015, max_missed_ticks: int = 2,
                 goto: dict | None = None, cartesian_max_speed: float = 0.5,
                 cartesian_max_rate: float = 1.5, stale_s: float | None = 0.1,
                 **options: Any) -> None:
        if (profile is None) == (controller is None):
            raise ValueError("WbcRobot needs exactly one of profile / controller")
        if controller is None:
            if variant is not None:
                options["variant"] = variant
            controller = load(profile, **options)
        elif options or variant is not None:
            raise TypeError(f"profile options {sorted(options)} given with a built controller")
        ctrl = controller
        if ctrl.joint_names is None:
            raise ValueError("the controller has no joint_names; build it from a profile")
        super().__init__(inner)
        self.controller = ctrl
        emb = inner.embodiment
        names = list(emb.joint_names)
        try:
            self._act_map = DofMap(names, ctrl.joint_names)
            self._obs_map = DofMap(names, ctrl.obs_joint_names)
        except ValueError as e:
            raise ValueError(f"controller joints not in the {emb.name} embodiment: {e}") from None
        own = set(ctrl.joint_names)
        self._direct = tuple(n for n in names if n not in own)
        self._direct_idx = np.array([names.index(n) for n in self._direct], dtype=int)
        self.dt = float(ctrl.control_dt)
        period = getattr(inner, "control_period", None)
        if period is not None and abs(float(period) - self.dt) > 1e-9:
            raise ValueError(f"{type(inner).__name__} runs at {period} s per send, the controller "
                             f"at {self.dt} s; set the embodiment's control_hz to {1 / self.dt:g}")
        if timeout_s <= 0 or start_seconds < 0:
            raise ValueError("timeout_s must be positive and start_seconds non-negative")
        self.timeout_s = float(timeout_s)
        self.outer_dt = self.dt if outer_hz is None else 1.0 / float(outer_hz)
        self.start_seconds = float(start_seconds)
        self.forward_timeout_s = float(forward_timeout_s)
        self.max_missed_ticks = int(max_missed_ticks)
        if stale_s is not None and stale_s <= 0:
            raise ValueError("stale_s must be positive (or None to turn the check off)")
        self.stale_s = None if stale_s is None else float(stale_s)
        self._direct_kp = self._direct_gains(direct_kp, "direct_kp")
        self._direct_kd = self._direct_gains(direct_kd, "direct_kd")
        if (ctrl.kp is not None or ctrl.kd is not None) and self._direct and (
                (ctrl.kp is not None and self._direct_kp is None)
                or (ctrl.kd is not None and self._direct_kd is None)):
            raise ValueError(
                f"{getattr(ctrl, 'profile', type(ctrl).__name__)} sends its own gains, so the "
                f"{len(self._direct)} joints it does not drive need direct_kp / direct_kd")

        modes = []
        if ReferenceKind.VELOCITY in ctrl.accepts:
            modes.append(ControlMode.BASE_VELOCITY)
        if ReferenceKind.LATENT in ctrl.accepts:
            modes.append(ControlMode.MOTION_LATENT)
        if ctrl.motion_spaces:
            modes.append(ControlMode.MOTION_CLIP)
        if ctrl.cartesian_targets and ReferenceKind.VELOCITY in ctrl.accepts:
            modes.append(ControlMode.CARTESIAN)
        self._modes = tuple(modes)
        self._has_base_pos = False   # BASE_POSE needs Observation.base_position (set at connect)
        unknown = sorted(set(goto or {}) - set(GOTO_DEFAULTS))
        if unknown:
            raise TypeError(f"unknown goto option(s) {unknown}; known: {sorted(GOTO_DEFAULTS)}")
        self.goto = {**GOTO_DEFAULTS, **(goto or {})}
        if cartesian_max_speed <= 0 or cartesian_max_rate <= 0:
            raise ValueError("cartesian_max_speed and cartesian_max_rate must be positive")
        self.cartesian_max_speed = float(cartesian_max_speed)
        self.cartesian_max_rate = float(cartesian_max_rate)
        self._ee_rest = {k: np.asarray(v, float) for k, v in ctrl.cartesian_rest().items()}
        self._motion_spaces = tuple(ctrl.motion_spaces)

        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._connected = False
        self._clear_run_state()

    # ------------------------------------------------------------------ helpers
    def _direct_gains(self, value, name: str) -> np.ndarray | None:
        if value is None:
            return None
        v = np.asarray(value, dtype=float)
        if v.ndim == 0:
            return np.full(len(self._direct), float(v))
        if v.shape != (len(self._direct),):
            raise ValueError(f"{name} has {v.size} values; the controller leaves "
                             f"{len(self._direct)} joints free: {list(self._direct)}")
        return v

    def _clear_run_state(self) -> None:
        self._mode = "idle"            # idle / ramp / stand / command / clip / behavior / goto
        self._command: Reference | None = None
        self._command_time = 0.0
        self._player: ClipPlayer | None = None
        self._clip = None              # the MotionClip _player was built from
        self._source = None
        self._behavior: str | None = None
        self._stand_ref: Any = None    # None = not built yet, False = PD hold
        self._goal: np.ndarray | None = None   # BASE_POSE [x, y, yaw]
        self._arrived = False
        # The caller's last high-level command, for RobotStatus.command: its mode and,
        # for BASE_POSE, its goal (kept after arrival; _goal is cleared on the way).
        self._cmd_mode: ControlMode | None = None
        self._cmd_goal: list[float] | None = None
        self._ee_pose: dict[str, np.ndarray] = {}   # rate-limited CARTESIAN targets
        self._ramp: dict | None = None
        self._last_target: np.ndarray | None = None   # controller order
        self._last_kp = self._last_kd = None
        self._direct_target = np.zeros(len(self._direct))
        self._direct_cmd_kp: np.ndarray | None = None
        self._direct_cmd_kd: np.ndarray | None = None
        self._last_clip: ClipPlayer | None = None
        self._ticks = 0
        self._outer_ticks = 0
        self._sim_time_target = 0.0
        self._outer_ticks = 0
        self._last_clip: ClipPlayer | None = None
        self._missed = 0
        self._missed_total = 0
        self._late_ticks = 0
        self._step_ms_max = 0.0

    def _now(self) -> float:
        return self._ticks * self.dt if self.simulated else time.monotonic()

    # ------------------------------------------------------------ Robot surface
    # Layer forwards embodiment, simulated, fault, now(), cameras / audio,
    # get_observation and the vendor's onboard skills to inner.
    @property
    def direct_joints(self):  # type: ignore[override]
        return self._direct

    @property
    def action_modes(self):  # type: ignore[override]
        if self._has_base_pos and ControlMode.BASE_VELOCITY in self._modes:
            return (*self._modes, ControlMode.BASE_POSE)
        return self._modes

    @property
    def cartesian_targets(self):  # type: ignore[override]
        if ControlMode.CARTESIAN not in self._modes:
            return ()
        return tuple(self.controller.cartesian_targets)

    @property
    def motion_spaces(self):  # type: ignore[override]
        return self._motion_spaces

    @property
    def control_period(self) -> float:
        """Seconds each ``send_action`` covers: the outer period (one tick by default)."""
        return self.outer_dt

    def set_gains(self, kp, kd) -> None:
        """The driver's fallback gains (joints the controller sends no gains for)."""
        with self._lock:
            self.inner.set_gains(kp, kd)

    def connect(self) -> None:
        if self.inner.has_onboard_skills:
            raise RuntimeError(
                f"{type(self.inner).__name__} offers its built-in skills, so the vendor's motion "
                "service owns the motors; drop builtin_skills from the robot config to run WBC")
        self.inner.connect()
        self._connected = True
        self._start()

    def reset(self, seed: int | None = None) -> None:
        """Stop, reset ``inner``, and run the start ramp again.

        ``Robot`` clears this layer's fault when ``reset`` returns.
        """
        self._stop_loop()
        self.inner.reset(seed)
        self._clear_fault()   # before _start: a simulated start ramp ticks through send_action
        self._start()

    def disconnect(self) -> None:
        """Stop the controller, damp ``inner`` when it can, then disconnect it.

        A humanoid left at its last PD target with nobody balancing it falls,
        so the joints go limp first (``damp`` is public on the Unitree and MuJoCo
        bodies). Unlike a fault this does not e-stop: ``connect()`` works again.
        """
        self._stop_loop()
        try:
            damp = getattr(self.inner, "damp", None)
            if callable(damp):
                damp()
        finally:
            self._connected = False
            self.inner.disconnect()

    def _estop(self) -> None:
        """Stop the controller thread (``inner`` is already stopped).

        Takes no lock: it may run while the thread holds ``_lock`` (a failing
        tick e-stops the stack from inside it). The fault state refuses new
        commands; ``reset()`` clears the targets.
        """
        self._stop.set()

    def hold(self) -> None:
        """Stand still: drop the clip, behavior and command (direct-joint targets stay)."""
        with self._lock:
            if self._mode != "ramp":
                self._cmd_mode = self._cmd_goal = None   # the caller ended its own command
                self._enter_stand()

    def _idle_step(self) -> None:
        if self.simulated and self._connected and self.fault is None:
            self._advance_outer()

    # ------------------------------------------------------------ layer skills
    # ``stand`` and the controller's behaviors are layer skills (Layer runs
    # run_layer_skill / stop_layer_skill and the lease); onboard skills are the
    # vendor's, forwarded to inner by Layer. hiRobot's list_robot_skills /
    # run_robot_skill cover both.
    def _layer_skills(self) -> list[RobotSkillSpec]:
        return [STAND_SKILL] + [
            RobotSkillSpec(name=b.name, description=b.description, parameters=dict(b.parameters))
            for b in self.controller.behaviors]

    def _start_layer_skill(self, spec: RobotSkillSpec, args) -> None:
        if spec.name == STAND_SKILL.name:
            self.hold()
            self._layer_skill_finished()   # standing is immediate; give the stream back
            return
        behavior = next(b for b in self.controller.behaviors if b.name == spec.name)
        with self._lock:
            source = self.controller.start_behavior(spec.name, **args)
            self._cmd_mode = self._cmd_goal = None   # the skill took the caller's stream
            self._player = None
            self._command = None
            self._behavior = spec.name
            if behavior.start_pos is not None and self.start_seconds > 0:
                self._begin_ramp(np.asarray(behavior.start_pos, np.float32), then=source)
            else:
                self._source, self._mode = source, "behavior"

    def _stop_layer_skill(self) -> None:
        """Stop the behavior and stand; ``Layer`` releases the lease afterwards."""
        with self._lock:
            self._behavior = None   # _enter_stand must not release the lease too
        self.hold()

    # ---------------------------------------------------------------- commands
    def _send_action(self, action: Action) -> None:
        mode = action.mode
        if mode not in self.action_modes:
            raise ValueError(
                f"WbcRobot takes {[m.value for m in self.action_modes]}, got {mode.value}; the "
                "whole-body controller owns the joints (joint_names + joint_target beside one of "
                f"those modes commands the free ones: {list(self._direct)})")
        with self._lock:
            if action.joint_names is not None:
                self._set_direct(action)
            if mode == ControlMode.BASE_VELOCITY:
                if action.base_velocity is None:
                    raise ValueError("BASE_VELOCITY needs base_velocity [vx, vy, wz]")
                self._set_command(Reference(ReferenceKind.VELOCITY, value=np.asarray(
                    action.base_velocity, np.float32).reshape(-1)))
            elif mode == ControlMode.MOTION_LATENT:
                if action.latent is None:
                    raise ValueError("MOTION_LATENT needs latent")
                self._set_command(Reference(ReferenceKind.LATENT, value=np.asarray(
                    action.latent, np.float32).reshape(-1)))
            elif mode == ControlMode.CARTESIAN:
                self._set_cartesian(action)
            elif mode == ControlMode.BASE_POSE:
                self._set_goal(action)
            elif mode == ControlMode.MOTION_CLIP:
                self._set_clip(action)
            self._cmd_mode = mode
            self._cmd_goal = self._goal.tolist() if mode == ControlMode.BASE_POSE else None
        if self.simulated:
            self._advance_outer()

    def _set_direct(self, action: Action) -> None:
        names = list(action.joint_names)
        if action.joint_target is None or np.asarray(action.joint_target).shape != (len(names),):
            raise ValueError("joint_names needs a joint_target of the same length")
        idx = [self._direct.index(n) for n in names]
        self._direct_target[idx] = np.asarray(action.joint_target, float)
        for gain, name in ((action.kp, "_direct_cmd_kp"), (action.kd, "_direct_cmd_kd")):
            if gain is None:
                continue
            base = self._direct_kp if name == "_direct_cmd_kp" else self._direct_kd
            cur = getattr(self, name)
            if cur is None:
                cur = np.zeros(len(self._direct)) if base is None else base.copy()
            cur[idx] = np.broadcast_to(np.asarray(gain, float), (len(names),))
            setattr(self, name, cur)

    def _set_command(self, ref: Reference) -> None:
        self._command, self._command_time = ref, self._now()
        if self._mode != "ramp":
            self._player = self._source = self._behavior = None
            self._mode = "command"
        else:
            self._ramp["then"] = None   # a command replaces a queued clip / behavior
            self._player = self._behavior = None

    def _set_cartesian(self, action: Action) -> None:
        names = self.controller.cartesian_targets
        if not action.ee_targets:
            raise ValueError(f"CARTESIAN needs ee_targets for some of {list(names)}")
        unknown = sorted(set(action.ee_targets) - set(names))
        if unknown:
            raise ValueError(f"{getattr(self.controller, 'profile', type(self.controller).__name__)} "
                             f"tracks {list(names)}, not {unknown}")
        targets = {}
        for ee, pose in action.ee_targets.items():
            pose = np.asarray(pose, float).reshape(-1)
            if pose.shape != (7,) or not np.isfinite(pose).all():
                raise ValueError(f"CARTESIAN target {ee!r} must be 7 finite numbers "
                                 "[x, y, z, qx, qy, qz, qw] in the base frame")
            norm = float(np.linalg.norm(pose[3:]))
            if norm < 1e-6:
                raise ValueError(f"CARTESIAN target {ee!r} has a zero quaternion")
            targets[names[ee]] = np.concatenate([pose[:3], pose[3:] / norm])
        vel = np.zeros(3) if action.base_velocity is None else action.base_velocity
        self._set_command(Reference(ReferenceKind.VELOCITY, value=np.asarray(
            vel, np.float32).reshape(-1), extra={"ee_targets": targets}))

    def _set_goal(self, action: Action) -> None:
        if action.base_target is None:
            raise ValueError("BASE_POSE needs base_target [x, y, yaw]")
        goal = np.asarray(action.base_target, float).reshape(-1)
        if goal.shape != (3,) or not np.isfinite(goal).all():
            raise ValueError("base_target must be 3 finite numbers [x, y, yaw]")
        self._goal, self._arrived = goal, False
        self._command = None
        if self._mode != "ramp":
            self._player = self._source = self._behavior = None
            self._mode = "goto"
        else:
            self._ramp["then"] = "goto"
            self._player = self._behavior = None

    def _goto_velocity(self, state: ControllerState) -> np.ndarray | None:
        """Body-frame ``[vx, vy, wz]`` toward the goal; ``None`` once there."""
        pos = state.extra.get("base_pos")
        if pos is None:
            raise RuntimeError("BASE_POSE needs Observation.base_position, which stopped arriving")
        g = self.goto
        yaw = _yaw_xyzw(state.base_quat)
        dx, dy = self._goal[0] - float(pos[0]), self._goal[1] - float(pos[1])
        dist = float(np.hypot(dx, dy))
        dyaw = _wrap(float(self._goal[2]) - yaw)
        if dist < g["pos_tol"] and abs(dyaw) < g["yaw_tol"]:
            return None
        c, s = np.cos(yaw), np.sin(yaw)
        v = g["kp_xy"] * np.array([c * dx + s * dy, -s * dx + c * dy])
        if dist < g["pos_tol"]:
            v[:] = 0.0
        speed = float(np.hypot(*v))
        if speed > g["max_speed"]:
            v *= g["max_speed"] / speed
        wz = float(np.clip(g["kp_yaw"] * dyaw, -g["max_yaw_rate"], g["max_yaw_rate"]))
        return np.array([v[0], v[1], wz], np.float32)

    def _set_clip(self, action: Action) -> None:
        clip = action.motion
        if clip is None:
            raise ValueError("MOTION_CLIP needs motion")
        if clip.space not in self.motion_spaces:
            raise ValueError(f"{type(self.controller).__name__} tracks {list(self.motion_spaces)} "
                             f"clips, not {clip.space!r}")
        if action.append and self._mode == "clip" and not self._player.done:
            self._player.append(clip)
            return
        if self._player is not None and not self._player.done and clip is self._clip:
            return   # the same clip sent again (to advance a simulation): keep playing it
        player = ClipPlayer(clip, self.dt, **self.controller.clip_player_options)
        self.controller.check_clip(player)
        self._player, self._command, self._source, self._behavior = player, None, None, None
        self._clip = clip
        if self._mode != "ramp":
            self._mode = "clip"
        else:
            self._ramp["then"] = "clip"

    # -------------------------------------------------------------- the loop
    def _start(self) -> None:
        self.controller.reset()
        with self._lock:
            self._clear_run_state()
            obs = self.inner.get_observation()
            self._has_base_pos = obs.base_position is not None
            self._direct_target = np.asarray(obs.joint_pos, float)[self._direct_idx].copy()
            if self.start_seconds > 0:
                self._begin_ramp(self.controller.default_pos.copy(), then=None)
            else:
                self._enter_stand()
        if self.simulated:
            with self._lock:
                while self._mode == "ramp":
                    self._tick_or_fail()
        else:
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="hiwbc-50hz", daemon=True)
            self._thread.start()

    def _begin_ramp(self, goal: np.ndarray, then) -> None:
        obs = self.inner.get_observation()
        start = self._act_map.remap(np.asarray(obs.joint_pos, float)).astype(np.float32)
        n = max(1, round(self.start_seconds / self.dt))
        self._ramp = {"start": start, "goal": goal, "n": n, "i": 0, "then": then}
        self._mode = "ramp"

    def _enter_stand(self) -> None:
        if self._behavior is not None:
            # A behavior ended on its own (or a command replaced it): give the
            # command stream back. Not for stop_layer_skill / a preemption,
            # which clear _behavior first and release the lease themselves.
            self._layer_skill_finished()
        self._player = self._source = self._behavior = self._command = None
        if not self._arrived:
            self._goal = None
        self._stand_ref = None
        self._mode = "stand"

    def _state(self, obs) -> ControllerState:
        quat = (np.array([0.0, 0.0, 0.0, 1.0]) if obs.base_orientation is None
                else np.asarray(obs.base_orientation, float))
        ang = np.zeros(3) if obs.base_angular_velocity is None else obs.base_angular_velocity
        lin = (None if obs.base_linear_velocity is None
               else _rotate_inv(quat, np.asarray(obs.base_linear_velocity, float)))
        extra = dict(obs.extra)
        if obs.base_position is not None:
            extra.setdefault("base_pos", np.asarray(obs.base_position, float))
        return ControllerState(
            dof_pos=self._obs_map.remap(np.asarray(obs.joint_pos)).astype(np.float32),
            dof_vel=self._obs_map.remap(np.asarray(obs.joint_vel)).astype(np.float32),
            base_quat=quat.astype(np.float32), base_ang_vel=np.asarray(ang, np.float32),
            base_lin_vel=None if lin is None else lin.astype(np.float32), extra=extra)

    def _reference(self, state: ControllerState) -> Reference | None:
        """This tick's reference; ``None`` = hold ``default_pos`` under PD."""
        if self._mode == "behavior":
            if not self._source.done:
                return self._source.next(state)
            self._enter_stand()
        if self._mode == "clip":
            if not self._player.done:
                return self.controller.clip_reference(self._player, state)
            self._last_clip = self._player
            self._enter_stand()
        if self._mode == "command":
            if self._now() - self._command_time <= self.timeout_s:
                return self._command
            self._enter_stand()
        if self._mode == "goto":
            v = self._goto_velocity(state)
            if v is not None:
                return Reference(ReferenceKind.VELOCITY, value=v)
            self._arrived = True
            self._enter_stand()
        if self._stand_ref is None:
            try:
                self._stand_ref = self.controller.stand_reference(state)
            except NotImplementedError:
                self._stand_ref = False
        return self._stand_ref or None

    def _limit_cartesian(self, ref: Reference | None) -> Reference | None:
        """Rate-limit ``ee_targets`` on a VELOCITY reference (see ``cartesian_max_speed``).

        A target starts from its rest pose (or the first command when the
        controller has none) and moves toward the command; a target no longer
        commanded moves back to rest and is dropped there. Any other reference
        kind drops them at once (a clip or token does not track VR points).
        """
        if ref is None or ref.kind != ReferenceKind.VELOCITY:
            self._ee_pose = {}
            return ref
        goals = ref.extra.get("ee_targets") or {}
        if not goals and not self._ee_pose:
            return ref
        max_d = self.cartesian_max_speed * self.dt
        max_a = self.cartesian_max_rate * self.dt
        out = {}
        for name in (*goals, *(n for n in self._ee_pose if n not in goals)):
            goal = goals.get(name, self._ee_rest.get(name))
            if goal is None:
                continue                  # no rest pose to return to: drop now
            cur = self._ee_pose.get(name)
            if cur is None:
                cur = self._ee_rest.get(name, goal)
            pose, reached = _step_pose(cur, goal, max_d, max_a)
            if reached and name not in goals:
                continue                  # back at rest
            out[name] = pose
        self._ee_pose = out
        extra = {k: v for k, v in ref.extra.items() if k != "ee_targets"}
        if out:
            extra["ee_targets"] = {k: v.copy() for k, v in out.items()}
        return Reference(ReferenceKind.VELOCITY, value=ref.value, extra=extra)

    def _check_fresh(self, obs: Observation) -> None:
        """Raise when the body's joint state is older than ``stale_s`` (the tick then e-stops)."""
        stamp = obs.stamps.get("body") if self.stale_s is not None else None
        if stamp is None:
            return
        age = self.inner.now() - stamp.t
        if age > self.stale_s:
            raise RuntimeError(f"body state is {age * 1e3:.0f} ms old (stale_s = "
                               f"{self.stale_s * 1e3:.0f} ms): no new joint state from the robot")

    def _tick(self) -> None:
        obs = self.inner.get_observation()
        self._check_fresh(obs)
        ctrl = self.controller
        kp, kd = ctrl.kp, ctrl.kd
        if self._mode == "ramp":
            r = self._ramp
            r["i"] += 1
            a = min(1.0, r["i"] / r["n"])
            position = (1 - a) * r["start"] + a * r["goal"]
            if r["i"] >= r["n"]:
                then = r["then"]
                if then is None and self._command is not None:
                    self._mode = "command"   # a command sent during the start ramp
                elif then is None:
                    self._enter_stand()
                elif then == "clip":
                    self._mode = "clip"
                elif then == "goto":
                    self._mode = "goto"
                else:
                    self._source, self._mode = then, "behavior"
        else:
            state = self._state(obs)
            ref = self._limit_cartesian(self._reference(state))
            if ref is None:
                position = ctrl.default_pos
            else:
                t0 = time.perf_counter()
                try:
                    target = ctrl.step(state, ref)
                except ForwardTimeout as e:
                    self._count_miss(e)          # raises after max_missed_ticks in a row
                    self._send_pd(self._last_target, self._last_kp, self._last_kd)
                    self._ticks += 1
                    return
                ms = (time.perf_counter() - t0) * 1e3
                self._step_ms_max = max(self._step_ms_max, ms)
                self._count_timing(ms)
                position, kp, kd = target.position, target.kp, target.kd
                if self._mode == "clip":
                    self._player.advance()
        self._last_target = np.asarray(position, float)
        self._last_kp, self._last_kd = kp, kd
        self._send_pd(self._last_target, kp, kd)
        self._ticks += 1

    def _count_miss(self, exc: ForwardTimeout) -> None:
        """A remote forward timed out: hold the last target, damp after too many in a row."""
        self._missed += 1
        self._missed_total += 1
        if self._last_target is None or self._missed > self.max_missed_ticks:
            raise RuntimeError(f"remote forward timed out {self._missed} tick(s) in a row "
                               f"(max_missed_ticks={self.max_missed_ticks}): {exc}") from exc
        logger.warning("remote forward timed out, repeating the last target: %s", exc)

    def _count_timing(self, ms: float) -> None:
        if self.simulated:
            self._missed = 0   # a step that returned ends a run of remote timeouts
            return             # simulated time: wall-clock lateness does not count
        if ms > self.forward_timeout_s * 1e3:
            self._missed += 1
            self._missed_total += 1
            # In-process inference cannot be cut short, so a late step's target is
            # still sent; repeated lateness damps.
            if self._missed > self.max_missed_ticks:
                raise RuntimeError(f"controller step over {self.forward_timeout_s * 1e3:.0f} ms "
                                   f"for {self._missed} ticks in a row")
        else:
            self._missed = 0

    def _send_pd(self, position, kp, kd) -> None:
        full = np.zeros(len(self.embodiment.joint_names))
        full[self._direct_idx] = self._direct_target
        q = self._act_map.scatter(np.asarray(position, float), default=full)

        def gains(ctrl_g, direct_cmd, direct_cfg):
            if ctrl_g is None:
                return None
            d = np.zeros(len(full))
            src = direct_cmd if direct_cmd is not None else direct_cfg
            if src is not None:
                d[self._direct_idx] = src
            return self._act_map.scatter(np.asarray(ctrl_g, float), default=d)

        self.inner.send_action(Action(
            mode=ControlMode.PD, joint_target=q,
            kp=gains(kp, self._direct_cmd_kp, self._direct_kp),
            kd=gains(kd, self._direct_cmd_kd, self._direct_kd)))

    def _tick_or_fail(self) -> None:
        try:
            self._tick()
        except Exception as e:
            self._fail(e)
            raise RobotFaultError(f"WbcRobot damped: {self._fault}") from e

    def _advance_outer(self) -> None:
        with self._lock:
            self._sim_time_target += self.outer_dt
            n = round(self._sim_time_target / self.dt) - self._outer_ticks
            for _ in range(max(n, 0)):
                self._tick_or_fail()
                self._outer_ticks += 1

    def _loop(self) -> None:
        next_t = time.monotonic()
        while not self._stop.is_set():
            with self._lock:
                try:
                    self._tick()
                except Exception as e:
                    if self._stop.is_set():
                        return   # e-stopped meanwhile: the robot below refused this tick
                    logger.exception("WBC tick failed; e-stopping")
                    self._fail(e, from_thread=True)
                    return
            next_t += self.dt
            delay = next_t - time.monotonic()
            if delay > 0:
                self._stop.wait(delay)
            else:
                self._late_ticks += 1
                if delay < -self.dt:
                    next_t = time.monotonic()

    def _stop_loop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout=1.0)
        self._thread = None

    def _fail(self, exc: BaseException, from_thread: bool = False) -> None:
        """Enter the fault state and e-stop the stack (the hardware first, see ``Layer.estop``)."""
        self._set_fault(f"{type(exc).__name__}: {exc}")
        if not from_thread:
            self._stop_loop()
        else:
            self._stop.set()
        try:
            self.estop()
        except Exception:
            logger.exception("e-stop after a WBC failure failed")

    # ------------------------------------------------------------------ status
    def get_status(self) -> RobotStatus:
        status = self.inner.get_status()
        status.fault = self.fault
        with self._lock:
            player = self._player if self._player is not None else self._last_clip
            ref = None
            if self._mode == "command" and self._command is not None:
                ref = self._command.kind.value
            status.extra["wbc"] = {
                "profile": getattr(self.controller, "profile", type(self.controller).__name__),
                "variant": getattr(self.controller, "variant", None),
                "mode": self._mode,
                "reference": ref,
                "behavior": self._behavior,
                "clip_time": None if player is None else player.time,
                "clip_duration": None if player is None else player.duration,
                "clip_done": None if player is None else player.done,
                "goal": None if self._goal is None else self._goal.tolist(),
                "ee_targets": {k: v.tolist() for k, v in self._ee_pose.items()},
                "arrived": self._arrived,
                "ticks": self._ticks,
                "missed_ticks": self._missed_total,
                "late_ticks": self._late_ticks,
                "step_ms_max": self._step_ms_max,
            }
            status.command = self._command_progress(player)
        return status

    def _command_progress(self, player: ClipPlayer | None) -> CommandProgress | None:
        """``RobotStatus.command``: progress of the caller's last high-level command.

        ``None`` before the first command, after ``hold()`` and while a layer skill
        runs (the skill holds the stream; see ``layer_skill_running``). A command
        that ended (goal reached, clip played out, velocity timed out to a stand)
        stays until the next one. ``MOTION_CLIP`` progress covers the whole
        ``append`` queue. Modes with no end (``BASE_VELOCITY``, ``MOTION_LATENT``,
        ``CARTESIAN``) report ``done=None``.
        """
        mode = self._cmd_mode
        if mode is None:
            return None
        if mode == ControlMode.BASE_POSE:
            return CommandProgress(mode=mode, done=self._arrived, goal=self._cmd_goal)
        if mode == ControlMode.MOTION_CLIP and player is not None:
            return CommandProgress(mode=mode, done=player.done, clip_time=player.time,
                                   clip_duration=player.duration)
        return CommandProgress(mode=mode)

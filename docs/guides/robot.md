# Running a controller as a hiRobot robot

English | [简体中文](./robot.zh-CN.md)

The other guides call `Controller.step` from your own loop. This guide covers the
other way to run a controller: as a [hiRobot](https://github.com/humanoidintelligence/hiRobot)
**layer**. Installing hiWBC registers a layer named `wbc` with hiRobot. A robot
config that lists it gets back a `Robot` that balances itself. The layer:

- runs the controller every control tick,
- takes high-level commands (`BASE_VELOCITY`, `BASE_POSE`, `CARTESIAN`,
  `MOTION_LATENT`, `MOTION_CLIP`) instead of joint targets,
- stands still when commands stop,
- damps the robot when anything in the tick fails.

hiRobot itself holds no WBC code; it finds the layer through the
`hirobot.drivers` entry point. Read [Getting started](getting_started.md) first.

## 1. A robot config with a `wbc` layer

```yaml
# g1_sonic.yaml: SONIC on the MuJoCo G1
defaults: [robots/unitree_g1_mujoco_standing]   # the G1 free on the floor

assembly:
  parts:
    body:
      overrides:
        model:
          # Start in SONIC's standing pose (G1 SDK order). SONIC must take over from tick 0.
          home: [-0.312, 0, 0, 0.669, -0.363, 0, -0.312, 0, 0, 0.669, -0.363, 0,
                 0, 0, 0, 0.2, 0.2, 0, 0.6, 0, 0, 0, 0.2, -0.2, 0, 0.6, 0, 0, 0]

sim:
  realtime: false          # step as fast as the caller sends
  ramp_seconds: 0.0

layers:
  - type: wbc
    profile: sonic/unitree_g1    # any hiwbc profile
    start_seconds: 0.0           # ramp to the controller's default pose first (3 s default)
```

Every key in the `wbc` entry except `type` goes to `WbcRobot`, and the keys it
does not know go to `hiwbc.load` as profile options (`variant`,
`checkpoint_dir`, `device`, `providers`, ...). See [Layer options](#6-layer-options).

```python
import numpy as np
from hirobot import Action, ControlMode, make_robot

robot = make_robot("g1_sonic.yaml")
robot.action_modes        # (BASE_VELOCITY, MOTION_LATENT, MOTION_CLIP, CARTESIAN, BASE_POSE)
robot.connect()

for _ in range(250):      # 5 s at 50 Hz
    robot.send_action(Action(mode=ControlMode.BASE_VELOCITY,
                             base_velocity=np.array([0.4, 0.0, 0.0])))

goal = Action(mode=ControlMode.BASE_POSE, base_target=np.array([0.0, 0.0, 0.0]))
while not robot.get_status().command.done:
    robot.send_action(goal)   # in simulation each send advances one outer period

robot.hold()              # stand still
robot.disconnect()
```

In the MuJoCo G1 this walks about 1.5 m forward and comes back to within about
10 cm of the start.

## 2. Time: simulation and real robot

- **Simulation** (the inner robot is simulated): time advances only when you
  call `send_action`. Each call runs one outer period (`outer_hz`, default: one
  control tick); with `outer_hz: 10` each call runs five 50 Hz ticks. To let a
  `BASE_POSE` or `MOTION_CLIP` run to completion, send the same `Action` object
  again each step until `get_status().command.done`. A repeated `BASE_POSE` goal
  just keeps walking toward it. A clip keeps playing only when it is the same
  `MotionClip` object that is playing; a new clip, even an identical one, starts
  from its first frame, and so does any clip sent after `done`. Stop sending once
  `done` is true.
- **Real robot**: `connect()` starts a thread that runs the controller at
  `1 / control_dt`. `send_action` only updates the command; the thread keeps
  balancing when you send nothing.

At `connect()` and `reset()` the layer ramps the controller's joints from the
measured pose to `default_pos` over `start_seconds`, then hands over to the
controller at once. The ramp is plain PD with no balance:

- **Real robot**: keep it on a gantry or supported during the ramp.
- **Free-floating simulation** (`unitree_g1_mujoco_standing`): the robot falls
  during the ramp (in MuJoCo the G1 was on the floor after the default 3 s, and
  the first SONIC tick then faulted). Start the simulator in the controller's
  default pose (the body's `model: {home: ...}`) and set `start_seconds: 0`, as the config
  above does. `unitree_g1_mujoco` welds the pelvis like a gantry, so the ramp is
  safe there.

### The simulated G1

hiRobot's MuJoCo G1 is the real G1's config with a MuJoCo overlay, so joint
names, order and cameras match the real robot, and the same `layers:` entry runs
on both. `unitree_g1_mujoco_standing` stands it free on the floor;
`unitree_g1_mujoco` welds the pelvis. Both need the G1 MJCF from Unitree's
`unitree_ros`: `hirobot assets fetch unitree_g1` downloads it (hiRobot's
[setup guide](https://github.com/humanoidintelligence/hiRobot/blob/main/docs/setup/unitree.md)).
The `kungfubot_tracker` and `protomotions_tracker` profiles read the same file for body FK.

- `robot.get_observation().images["head"]` is the head camera (1920x1080),
  rendered each call (about 1 ms). The layer does not use it; it is there for a
  policy above. Without a GL context the robot runs without images.
- In simulation `make_robot` returns the `WbcRobot` itself (the real camera and
  audio drivers are skipped). On the real G1 it returns hiRobot's
  `RobotAssembly` with the cameras and audio around the layer; `send_action`,
  `get_status` and the rest reach the layer the same way.
- `hirobot sim <config> --viewer` connects a simulated robot and holds its start
  pose, which is a quick check that the model and gains load. It sends plain PD
  joint targets, so it does not run the `wbc` layer, and a free-standing G1
  falls.

## 3. Commands

`robot.action_modes` lists the modes the loaded controller supports:

| Mode | Fields | Supported by | What happens |
|---|---|---|---|
| `BASE_VELOCITY` | `base_velocity` `[vx, vy, wz]`, body frame | velocity controllers; `sonic` (through its planner) | Walk at this velocity |
| `BASE_POSE` | `base_target` `[x, y, yaw]`, world frame | every controller with `BASE_VELOCITY`, when the robot reports `base_position` | Walk to the target, then stand. Progress in `status.command` ([§5](#5-status)) |
| `CARTESIAN` | `ee_targets` `{name: [x, y, z, qx, qy, qz, qw]}` in the base frame, optional `base_velocity` | `sonic` (`left_hand`, `right_hand`) | Track hand poses with SONIC's VR 3-point teleop encoder, while walking at `base_velocity` |
| `MOTION_LATENT` | `latent` | `sonic` (64-d token) | Use this motion token for the tick |
| `MOTION_CLIP` | `motion` (`MotionClip`), `append` | controllers with `motion_spaces` | Play the clip, then stand. `append=True` queues it after the current one |

Rules that apply to every mode:

- **Timeout.** A `BASE_VELOCITY`, `CARTESIAN` or `MOTION_LATENT` command older than
  `timeout_s` (0.5 s) is dropped and the robot stands still. A caller has to keep
  sending. `BASE_POSE` and `MOTION_CLIP` run until they finish.
- **`hold()`** stands still in place. An idle caller calls it once, then lets
  time pass without sending (in simulation through hiRobot's idle step); it does
  not send a zero velocity, which is not standing for every controller
  (`smooth_loco` stands at `[-0.5, 0, -0.5]`).
- **Joints the controller does not drive** (fingers, or the arms of a legs-only
  policy) are in `robot.direct_joints`. Command them in the same `Action` with
  `joint_names` + `joint_target`. Leave out the mode's own field (`base_velocity`,
  `latent`, `ee_targets`) to move only them: the reference and its timeout stay.
- **Speed range.** `robot.base_velocity_limits` is the per-axis `(min, max)` of
  `base_velocity` the profile was tested at, in its units (`None` if not given),
  for scaling a joystick. The layer does not clamp to it.
- **Layer skills.** `stand` and the controller's behaviors (for example
  BeyondMimic's `dance`) are listed by `robot.list_layer_skills()` and started
  with `robot.run_layer_skill(name)`; `robot.stop_layer_skill()` ends one. While
  a behavior runs the layer holds the command stream, so a direct `send_action`
  raises `RobotBusyError` until it ends. The vendor robot's own skills are
  `robot.list_onboard_skills()` (none on the simulated G1).
  `robot.list_robot_skills()`, `robot.run_robot_skill(name)` and
  `robot.stop_robot_skill()` cover both kinds in one list.

`CARTESIAN` targets are rate-limited: the pose SONIC tracks moves toward the
commanded one at most `cartesian_max_speed` (0.5 m/s) and `cartesian_max_rate`
(1.5 rad/s). It starts from SONIC's default VR pose and returns to it when the
command ends. A 15–20 cm jump in one tick made SONIC command joint targets past
the ±3 rad safety limit in MuJoCo, so do not raise these limits much. In MuJoCo,
hand targets 16–29 cm from rest were reached to within 5–12 cm; SONIC is a
learned tracker, not IK.

## 4. Faults

An exception in a tick (the controller, the safety gate underneath, the driver)
enters the fault state and e-stops the stack: hiRobot stops the robot at the
bottom first (a Unitree or MuJoCo body is damped, kp 0), then the layer stops
its thread. After that:

- `send_action` raises `RobotFaultError`,
- `robot.get_status().fault` holds the reason,
- `robot.reset()` clears it and ramps to `default_pos` again.

Keep hiRobot's safety gate enabled. It is what turns a bad joint target into a
damped robot instead of a sent command.

## 5. Status

`robot.get_status().command` is hiRobot's generic `CommandProgress` for the
caller's last high-level command; read it rather than `extra["wbc"]` to tell
whether the robot arrived or a clip ended:

| Field | `BASE_POSE` | `MOTION_CLIP` | `BASE_VELOCITY`, `MOTION_LATENT`, `CARTESIAN` |
|---|---|---|---|
| `mode` | `BASE_POSE` | `MOTION_CLIP` | that mode |
| `done` | `True` once the goal is reached | `True` once the clip has played out | `None` (no end) |
| `goal` | `[x, y, yaw]`, in the frame of `base_position` | `None` | `None` |
| `clip_time`, `clip_duration` | `None` | Seconds played and total, over the whole `append` queue | `None` |

An ended command (arrived, played out, or a velocity command that timed out to a
stand) stays until the next one. `command` is `None` before the first command,
after `hold()`, and while a layer skill runs: the skill holds the command stream,
so its progress is `layer_skill_running`, not `command`.

`robot.get_status().extra["wbc"]` holds the layer's diagnostics:

| Key | Meaning |
|---|---|
| `profile`, `variant` | The loaded controller |
| `mode` | `idle`, `ramp`, `stand`, `command`, `goto`, `clip` or `behavior` |
| `clip_time`, `clip_duration`, `clip_done` | Clip progress, or `None` |
| `goal`, `arrived` | `BASE_POSE` target and whether it was reached |
| `ee_targets` | Hand poses SONIC is tracking right now (after the rate limit) |
| `ticks` | Control ticks run |
| `missed_ticks` | Ticks whose forward pass timed out (remote inference) or ran late (real time) |
| `late_ticks`, `step_ms_max` | Step timing |

## 6. Layer options

| Option | Default | Description |
|---|---|---|
| `profile` | required | `hiwbc.load` profile, e.g. `sonic/unitree_g1` |
| `variant` | profile default | Profile variant |
| `timeout_s` | 0.5 | Stand still when no command arrives for this long |
| `outer_hz` | control rate | Simulation: how often the caller sends |
| `start_seconds` | 3.0 | Ramp to `default_pos` at start |
| `direct_kp`, `direct_kd` | driver gains | Gains of the joints the controller does not drive (scalar or one per joint). Required when the controller sends its own gains (SONIC) and some joints are not its own |
| `forward_timeout_s`, `max_missed_ticks` | 0.015, 2 | A step longer than `forward_timeout_s` counts as missed; more than `max_missed_ticks` in a row damps. Wall-clock lateness is counted on a real robot only; remote timeouts always |
| `stale_s` | 0.1 | Fault when the body's joint state (`obs.stamps["body"]`) is older than this at a tick, on the robot's clock. `None` turns it off |
| `goto` | see below | `BASE_POSE` gains and limits |
| `cartesian_max_speed`, `cartesian_max_rate` | 0.5, 1.5 | `CARTESIAN` rate limit, m/s and rad/s |
| `runtime`, `endpoint`, `remote_timeout_s` | `local` | [Remote inference](#8-remote-inference) |
| other keys | | Profile options for `hiwbc.load` |

`goto` keys: `kp_xy` 1.0 (m/s per m), `kp_yaw` 1.0 (rad/s per rad),
`max_speed` 0.6 m/s, `max_yaw_rate` 0.6 rad/s, `pos_tol` 0.10 m,
`yaw_tol` 0.15 rad. The error is turned into a body-frame velocity command
each tick; inside both tolerances the robot stands.

## 7. SONIC's locomotion planner

SONIC by itself only tracks reference motions. To walk on a velocity command,
the `sonic/unitree_g1` profile adds NVIDIA's kinematic planner
(`planner_sonic.onnx`), as the C++ deployment does: the planner turns the
command into future lower-body frames, and SONIC tracks them. Details in the
[SONIC guide](sonic.md#7-walking-velocity-commands-and-the-planner).

## 8. Remote inference

The 50 Hz loop, the observation, the safety checks and the faults stay on the
robot; only the model forward runs on a workstation. If the network or the
workstation fails, the robot notices the timeout and damps itself.

On the workstation:

```bash
python -m hiwbc.serve --profile sonic/unitree_g1 --host 0.0.0.0 --port 5555
#   [--variant low_latency] [--option checkpoint_dir=/data/sonic] [--option providers=CUDAExecutionProvider]
```

On the robot, add to the `wbc` layer:

```yaml
layers:
  - type: wbc
    profile: sonic/unitree_g1
    runtime: remote
    endpoint: tcp://192.168.123.10:5555
    remote_timeout_s: 0.015
```

or in Python: `hiwbc.load("sonic/unitree_g1", runtime="remote", endpoint="tcp://...")`.

- The robot still needs the checkpoints: it builds the controller from them,
  then replaces each model call with a call to the server. The server refuses a
  client whose profile or variant differs from its own.
- A call that does not return within `remote_timeout_s` (or finds the server
  gone) counts as a missed tick: the layer sends the previous joint target
  again. With the default `max_missed_ticks` of 2, the third missed tick in a row
  damps the robot. The client reconnects on the next call.
- The server keeps no state, so the robot can reconnect at any time. Recurrent
  models (the `unitree_loco` `default` LSTM) are refused on both sides; run them
  locally.
- **The service has no authentication.** It listens on 127.0.0.1 unless you
  pass `--host`. Use it only on a trusted wired network; Wi-Fi latency is too
  irregular for a 15 ms budget.

In MuJoCo, remote and local inference gave identical joint targets every tick
over 8 s of walking. Over localhost a call adds about 0.1 ms.

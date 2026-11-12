# SONIC

English | [简体中文](./sonic.zh-CN.md)

`sonic` runs NVIDIA's [GEAR-SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl)
behavior foundation model on the 29-DoF Unitree G1. SONIC is two models:

- the **encoder** maps a reference (G1 motion frames, a VR 3-point teleop target,
  or SMPL human joints) to a 64-d motion token;
- the **decoder** maps the token plus 10 ticks of proprioception history to 29
  joint actions.

A VLA can emit the token directly (`ReferenceKind.LATENT`), which skips the
encoder. Read [Getting started](getting_started.md) first.

hiWBC's `sonic` follows NVIDIA's C++ deployment (`gear_sonic_deploy`). Checked
against a transliteration of that C++ in MuJoCo, every encoder input, token,
decoder input and PD target agreed within 1e-5 over 600 ticks.

## 1. Get the checkpoints

hiWBC ships no SONIC weights. Download them from
[nvidia/GEAR-SONIC](https://huggingface.co/nvidia/GEAR-SONIC) (NVIDIA Open Model
License):

```bash
pip install -e '.[onnx]' huggingface_hub
huggingface-cli download nvidia/GEAR-SONIC --local-dir ~/.cache/hiwbc/sonic
```

| Variant | Files | Encoder input | Decoder input |
|---|---|---|---|
| `default` | `model_encoder.onnx`, `model_decoder.onnx` | 1762 | 994 |
| `low_latency` | `low_latency/` | 1247 | 994 |
| `sonic_v1_1` | `sonic_v1_1/` | 1751 | 994 |

Each folder has an `observation_config.yaml`; `SONIC_VARIANTS` in
`hiwbc.configs.sonic.unitree_g1` mirrors those term orders. The same download
includes the locomotion planner, `planner_sonic.onnx` (740 MB, shared by all
variants; see [section 7](#7-walking-velocity-commands-and-the-planner)).

## 2. Build the controller

The profile loads a variant from the checkpoint folder:

```python
import hiwbc

ctrl = hiwbc.load("sonic/unitree_g1", variant="default")   # checkpoint_dir: ~/.cache/hiwbc/sonic
# encoder=False for LATENT-only use; encoder_mode="joint" | "teleop" | "smpl"
# planner=False to skip the locomotion planner; planner_thread=True to replan off the control thread
```

Or build it directly:

```python
from hiwbc.configs.sonic.unitree_g1 import SONIC_G1, SONIC_VARIANTS
from hiwbc.controllers.sonic import SonicController

ckpt = "~/.cache/hiwbc/sonic"
ctrl = SonicController.from_onnx(
    f"{ckpt}/model_decoder.onnx",
    f"{ckpt}/model_encoder.onnx",   # omit for LATENT-only use
    robot=SONIC_G1,                 # joint order, standing pose, gains of the 29-DoF G1
    obs=SONIC_VARIANTS["default"],
    encoder_mode="joint",
)
ctrl.reset()
print(ctrl.encoder_dim, ctrl.decoder_dim)   # 1762 994
```

Each `step` returns `JointTarget(position, kp, kd)`: SONIC brings its own
per-joint gains (`SONIC_G1.kp`, `SONIC_G1.kd`), so send them with the target.

## 3. Joint order

SONIC runs in **IsaacLab order** (`G1_JOINT_NAMES`). The robot and MuJoCo use
SDK order (`G1_JOINT_NAMES_MUJOCO`). Both use hiRobot's joint names (no `_joint`
suffix), so hiRobot's `DofMap` can do the reordering by name;
`ISAACLAB_TO_MUJOCO_INDEX` is the same permutation as an index array:

```python
import numpy as np
from hiwbc import ControllerState
from hiwbc.configs.sonic.unitree_g1 import ISAACLAB_TO_MUJOCO_INDEX as I2M

# robot (SDK order) -> SONIC (IsaacLab order)
state = ControllerState(
    dof_pos=q_sdk[I2M], dof_vel=dq_sdk[I2M],
    base_quat=quat_wxyz[[1, 2, 3, 0]],      # SDK / MuJoCo give [w, x, y, z]
    base_ang_vel=gyro,
)
target = ctrl.step(state, ref)

# SONIC -> robot
q_cmd = np.empty(29); q_cmd[I2M] = target.position
kp = np.empty(29); kp[I2M] = target.kp
kd = np.empty(29); kd[I2M] = target.kd
```

## 4. Track a reference motion (`joint` mode)

With a hiRobot `MotionClip` (space `joint` or `smpl`, with `root_quat`), a
`ClipPlayer` builds the encoder window each tick:

```python
from hiwbc.references import ClipPlayer

player = ClipPlayer(clip, ctrl.control_dt)
ctrl.check_clip(player)
while not player.done:
    state = read_state()
    send(ctrl.step(state, ctrl.clip_reference(player, state)))
    player.advance()
```

The player resamples the clip to 50 Hz, aligns its heading and xy start to the
robot at the first tick (the deployment's `apply_delta_heading`), passes the
window the encoder reads (46 frames for `default`), and sets `encoder_mode` to
the clip's space. Checked against the deployment-style player in
on two GR00T clips (497 and 1375 ticks), all
three variants, with the robot heading starting at 1.3 rad and drifting: the
encoder inputs agree within 4.8e-7.

To build the frames yourself:

The `joint` encoder (the deployment's `g1` mode) takes future G1 frames: `joint_pos` and `joint_vel` (T, 29) in
IsaacLab order, `root_quat` (T, 4), `[x, y, z, w]`, and `root_z` (T,), the
reference root height, which the `default` variant reads. Pass a window starting
at the current frame; the controller samples it at the stride of each term
(every 5th frame for `default`) and repeats the last frame when the window is
short. The `default` config looks 45 frames ahead, so pass 46.

```python
from hiwbc import Reference, ReferenceKind

# motion: arrays at 50 Hz from a clip, e.g. the deployment's reference CSVs
# joint_pos.csv / joint_vel.csv (IsaacLab order), body_quat.csv (root = body 0, wxyz)
T = len(motion["joint_pos"])
for t in range(T):
    win = slice(t, min(t + 46, T))
    ref = Reference(ReferenceKind.MOTION_FRAMES, frames={
        "joint_pos": motion["joint_pos"][win],
        "joint_vel": motion["joint_vel"][win],
        "root_quat": aligned_root_xyzw[win],      # see below
        "root_z": motion["body_pos"][win, 0, 2],  # root height, m
    })
    target = ctrl.step(read_state(), ref)
    send(target)
```

**Heading alignment** is yours when you build the frames. A clip recorded
facing one way must be rotated, by yaw only, so its first frame faces the way
the robot faces at the start:

```python
from hiwbc.math_utils import quat_mul, quat_conjugate

def yaw_only(q):                       # [x, y, z, w]
    q = np.array([0.0, 0.0, q[2], q[3]])
    return q / np.linalg.norm(q)

align = quat_mul(yaw_only(robot_quat_at_start), quat_conjugate(yaw_only(clip_root_xyzw[0])))
aligned_root_xyzw = np.array([quat_mul(align, q) for q in clip_root_xyzw])
```

Pass `extra={"play": False}` to pause: the controller holds frame 0 and zeroes
the velocities, as the deployment does.

## 5. Drive SONIC with a token (`LATENT`)

```python
ref = Reference(ReferenceKind.LATENT, value=token)     # (64,), e.g. a VLA's motion token
target = ctrl.step(state, ref)
```

The encoder is not run. `target.extra["motion_token"]` (and `ctrl.token`) hold
the token used, so a run in `MOTION_FRAMES` mode can record tokens for later
replay through `LATENT`. Replaying the encoder's own tokens this way gave the
same tracking as running the encoder.

## 6. Teleop and SMPL modes

| Mode | `frames` keys |
|---|---|
| `teleop` | `joint_pos`, `joint_vel` (the lower body is selected inside), `root_quat`, `vr_3point_pos` (3, 3) and `vr_3point_quat` (3, 4, `[x, y, z, w]`) for left wrist, right wrist, head, in the reference root frame |
| `smpl` | `smpl_joints` (T, 24, 3) root-local SMPL joints, `root_quat`, `joint_pos` (the wrist joints are selected inside) |

Switch per tick with `extra={"encoder_mode": "teleop"}`. A key the mode needs
but `frames` lacks raises `KeyError`; the C++ deployment falls back to another
mode silently instead.

## 7. Walking: velocity commands and the planner

SONIC has no velocity input. NVIDIA's C++ deployment walks it with a separate
kinematic planner (`planner_sonic.onnx`) that predicts the next 64 frames of
G1 motion from the last 4 frames and a movement command. SONIC then tracks
those frames in `joint` mode. The `sonic/unitree_g1` profile loads the planner
by default (`planner=True`, needs `encoder=True`), and the controller then
also accepts `VELOCITY`:

```python
ref = Reference(ReferenceKind.VELOCITY, value=np.array([0.5, 0.0, 0.0]))   # [vx, vy, wz], body frame
target = ctrl.step(state, ref)
```

`[0, 0, 0]` stands still (the planner's idle mode). `hiwbc.controllers.sonic_planner`
follows the deployment's planner path:

- `[vx, vy, wz]` becomes a planner command the way the gamepad does: `wz` turns
  the commanded facing, at most 90° ahead of the robot's actual heading, and the
  speed picks the mode: idle below 0.05 m/s, slow walk to 0.8, walk to 1.5, run
  to 3.0 m/s.
- The prediction (30 Hz) is resampled to 50 Hz and blended into the motion being
  played, starting 2 frames after the frame the planner's context was taken at,
  linearly over 8 frames.
- The planner is checked at 10 Hz and replans when the command changes, and
  every 0.1–1 s while moving. Standing, it pulls the lower-body reference toward
  the measured joints when they drift apart.
- Switching away from `VELOCITY` (a clip, a token) and back restarts the planner
  from the measured joints.

One planner call takes about 35 ms on a desktop CPU, longer than a 20 ms tick.
By default (`planner_thread=False`) it runs inside `step` on the ticks that
replan; a real robot should use `planner_thread=True`, which runs it on a
worker thread and blends the result in when it is ready, as the deployment
does.

**Hand targets while walking.** `extra["ee_targets"]` on a `VELOCITY` reference
switches the encoder to `teleop`: the planner still drives the lower body, and
the VR 3-point term comes from the given wrist poses (`{"left_wrist": [x, y, z,
qx, qy, qz, qw], ...}` in the root frame; missing points keep the deployment's
default pose). Rate-limit the targets: a 15–20 cm jump in one tick made SONIC
command elbow targets near −4 rad. hiRobot's `CARTESIAN` mode through the `wbc`
layer does the limiting for you ([Running as a hiRobot robot](robot.md#3-commands)).

Results through the `wbc` layer in MuJoCo (2 ms physics step, 50 Hz control):

| Command | Result |
|---|---|
| stand, 10 s | moved 5 cm in the first 2 s, then still |
| vx 0.5 m/s, 8 s | 3.44 m (0.43 m/s), 0.27 m sideways drift |
| wz 0.5 rad/s, 6 s | 151° (0.44 rad/s) |
| `BASE_POSE` to (1.5 m, 1.0 m, 90°) | arrived in 8.7 s, within 9.7 cm / 8° |
| hand targets 16–29 cm from rest | reached to within 5–12 cm; the hands moved about 1.3× the requested distance |
| walk at 0.4 m/s with the left hand raised | 1.61 m in 5 s, mean wrist error 10.9 cm |

The robot never fell. Walking runs slower than commanded and hand tracking
overshoots; neither has been compared with the C++ deployment yet.

## 8. Simulation results

Run in MuJoCo with GR00T's G1 scene, 50 Hz control:

| Motion | Variant | Ticks | Mean joint error (rad) | Fell |
|---|---|---|---|---|
| macarena | default | 600 | 0.087 | no |
| macarena | low_latency | 600 | 0.103 | no |
| macarena | sonic_v1_1 | 600 | 0.104 | no |
| macarena | default, token via `LATENT` | 600 | 0.087 | no |
| neutral_kick | default | 165 (clip end) | 0.097 | no |
| forward_lunge | default | 399 (clip end) | 0.104 | no |
| dance_in_da_party | default | 497 (clip end) | 0.130 | no |

**Physics step.** These runs used a 2 ms physics step. At 5 ms, the step in
GR00T's sim config, the robot fell after about 1.4 s. The C++
transliteration fell the same way, so hiWBC's observation and action code is
not the cause; the actual cause was not found, and GR00T's own simulator loop
may do something this test harness does not. Use 2.5 ms or finer in MuJoCo.

## Pitfalls

- **Joint order.** SONIC is IsaacLab order; the robot is SDK order. Map both
  the state and the target, including `kp` / `kd`.
- **Gains.** Send `target.kp` / `kd`. The robot's default gains are not
  SONIC's.
- **Heading.** An unaligned `root_quat` makes the robot turn to the clip's
  recorded heading as soon as it starts. `ClipPlayer` aligns it; frames you
  build yourself must be aligned first.
- **Planner latency.** On a real robot set `planner_thread=True`; a synchronous
  replan takes longer than one tick.
- **Window too short.** Pass enough future frames (46 for `default`); fewer
  repeat the last frame and the robot anticipates less.

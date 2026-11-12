# Velocity-command locomotion

English | [简体中文](./locomotion.zh-CN.md)

Controllers that walk the robot from a joystick-style command:
`ReferenceKind.VELOCITY`, `value = [vx, vy, wyaw]` (AMO: plus height and
torso orientation). Read [Getting started](getting_started.md) first for the
control loop, joint mapping and first power-on.

| Controller | Policy | G1 checkpoint (RoboJuDo `assets/models/g1/`) | Joints driven |
|---|---|---|---|
| `unitree_loco` | [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) | `unitree/policy_lstm_1.pt` | 12 legs |
| `unitree_loco` (`with_gait=False`) | [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab) | `unitree/policy_wo_gait.pt` | 29 (IsaacLab order) |
| `smooth_loco` | [Smooth Humanoid Locomotion](https://github.com/zixuan417/smooth-humanoid-locomotion) | `smooth/for_test-2000-jit.pt` | 21 |
| `amo_loco` | [AMO](https://github.com/OpenTeleVision/AMO) | `amo/amo_jit.pt` + `adapter_jit.pt` + `adapter_norm_stats.pt` | 15 driven, 23 observed |

All run at 50 Hz. Joint names below drop the `_joint` suffix.

## Joint orders

**12-DoF legs** (`unitree_loco`, and the first 12 of every order below except
IsaacLab):

```
left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee, left_ankle_pitch, left_ankle_roll,
right_hip_pitch, right_hip_roll, right_hip_yaw, right_knee, right_ankle_pitch, right_ankle_roll
```

**21-DoF** (`smooth_loco`): the 12 legs, then
`waist_yaw, left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw, left_elbow,
right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw, right_elbow`.

**23-DoF** (`amo_loco` observation): the 12 legs, then
`waist_yaw, waist_roll, waist_pitch`, then the 8 arm joints (shoulder pitch / roll /
yaw, elbow; left then right). AMO drives the first 15 (legs + waist).

**29-DoF IsaacLab order** (`unitree_loco` with `with_gait=False`): joints
interleaved left / right / waist:

```
left_hip_pitch, right_hip_pitch, waist_yaw, left_hip_roll, right_hip_roll, waist_roll,
left_hip_yaw, right_hip_yaw, waist_pitch, left_knee, right_knee,
left_shoulder_pitch, right_shoulder_pitch, left_ankle_pitch, right_ankle_pitch,
left_shoulder_roll, right_shoulder_roll, left_ankle_roll, right_ankle_roll,
left_shoulder_yaw, right_shoulder_yaw, left_elbow, right_elbow,
left_wrist_roll, right_wrist_roll, left_wrist_pitch, right_wrist_pitch,
left_wrist_yaw, right_wrist_yaw
```

## `unitree_loco`: unitree_rl_gym (gait)

```python
import numpy as np
import torch
from hiwbc import ControllerState, Reference, ReferenceKind, make_controller

LEG_DEFAULT = [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0] * 2
KP = [100, 100, 100, 150, 40, 40] * 2          # set on the robot side
KD = [2, 2, 2, 4, 2, 2] * 2

model = torch.jit.load("assets/models/g1/unitree/policy_lstm_1.pt", map_location="cpu")

@torch.no_grad()
def forward_fn(obs):
    return model(torch.from_numpy(obs)[None]).numpy().squeeze()

ctrl = make_controller(
    "unitree_loco",
    num_actions=12,
    default_pos=LEG_DEFAULT,
    forward_fn=forward_fn,
    # defaults match RoboJuDo's G1 config: cycle_time=0.8, max_cmd=(0.8, 0.5, 1.57),
    # obs scales ang_vel 0.25 / dof_vel 0.05 / command (2, 2, 0.25),
    # action_scale=0.25, action_beta=0.8
)
ctrl.reset()

ref = Reference(ReferenceKind.VELOCITY, value=np.array([0.5, 0.0, 0.0]))
target = ctrl.step(state, ref)   # state: 12 leg joints, see Getting started
```

**The command is normalized.** `value` is in `[-1, 1]` per axis and the
controller multiplies it by `max_cmd`, so `[0.5, 0, 0]` asks for
`0.5 * 0.8 = 0.4 m/s` forward. Positive is forward, left and counter-clockwise.
To command in physical units, divide by `max_cmd` first.

The gait clock (`phase = t / (50 * 0.8)`) runs from `reset()` whether or not
the robot is moving. `policy_lstm_1.pt` is recurrent: its hidden state lives in
the model object, so load one model per controller. `ctrl.reset()` restores it
to its as-loaded state (`load_torchscript_runner` snapshots the buffers).

The arms and waist are not driven: hold them at a fixed pose with your own PD
gains.

## `unitree_loco`: unitree_rl_lab (`with_gait=False`)

The unitree_rl_lab policy drives all 29 joints in IsaacLab order, has no gait
clock, and takes a 5-tick history.

```python
from hiwbc.controllers.unitree import ObsScales

G1_29_DEFAULT = [  # IsaacLab order
    -0.1, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.3, 0.3, -0.2, -0.2,
    0.25, -0.25, 0.0, 0.0, 0.0, 0.0, 0.97, 0.97, 0.15, -0.15, 0.0, 0.0, 0.0, 0.0,
]

ctrl = make_controller(
    "unitree_loco",
    num_actions=29,
    default_pos=G1_29_DEFAULT,
    forward_fn=forward_fn,               # wraps policy_wo_gait.pt
    with_gait=False,
    history_length=5,
    gravity_scale=1.0,
    obs_scales=ObsScales(ang_vel=0.2, dof_vel=0.05, command=(1.0, 1.0, 1.0)),
    action_beta=1.0,                     # this policy is trained without EMA
)
```

The observation is the 6 terms (ang vel, gravity, command, joint pos, joint
vel, last action), each kept for 5 ticks: `5 * (9 + 3 * 29) = 480` values. The
history is term-major and includes the current tick, so the first tick after
`reset()` sees four zero frames, then the current one. The command is still
normalized and multiplied by `max_cmd`.

## `smooth_loco`

```python
from hiwbc.controllers.smooth import SmoothObsScales

SMOOTH_DEFAULT = [-0.4, 0.0, 0.0, 0.8, -0.35, 0.0] * 2 + [0.0] * 9   # 21-DoF order

ctrl = make_controller(
    "smooth_loco",
    num_actions=21,
    default_pos=SMOOTH_DEFAULT,
    forward_fn=forward_fn,               # wraps for_test-2000-jit.pt
    cycle_time=0.64,
    n_priv_latent=50,                    # 4 + 1 + 2 * 21 + 3, zero-filled
    history_length=10,
    obs_scales=SmoothObsScales(ang_vel=0.25, dof_vel=0.05),
    action_scale=0.5,
    action_clip=10.0,
    action_beta=0.8,
)
```

The observation is the current proprio block (`10 + 3 * 21 = 73`), the 50-zero
privileged slot, and the previous 10 blocks, oldest first:
`73 + 50 + 730 = 853` values.

The command enters unscaled. RoboJuDo's G1 config maps the joystick to these
ranges (stick minimum / neutral / maximum); reproduce them to command what the
policy was trained on:

| `value` index | Stick | Range |
|---|---|---|
| 0 | left Y | -1.5 / **-0.5** / 0.5 |
| 1 | left X | 0.25 / 0.0 / -0.25 |
| 2 | right X | 0.5 / **-0.5** / -1.5 |

The neutral command is `[-0.5, 0.0, -0.5]`, not zero. In the lockstep check this
checkpoint (`for_test-2000`) fell after about 7 s of scripted walking, the same
way under RoboJuDo alone; treat it as a test checkpoint.

## `amo_loco`

AMO commands the lower body by velocity and the upper body by height and torso
orientation. An adapter MLP turns that command plus the arm joint angles into
part of the observation, and the policy takes a second input, the last 25
proprio frames.

```python
from hiwbc import make_controller
from hiwbc.controllers.amo import load_amo_runners

AMO_OBS_DEFAULT = (
    [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0] * 2         # legs
    + [0.0, 0.0, 0.0]                            # waist yaw / roll / pitch
    + [0.5, 0.0, 0.2, 0.3, 0.5, 0.0, -0.2, 0.3]  # arms
)

ctrl = make_controller(
    "amo_loco",
    num_actions=15,
    default_pos=AMO_OBS_DEFAULT[:15],
    obs_default_pos=AMO_OBS_DEFAULT,
    default_command=[0.0, 0.0, 0.0, 0.75, 0.0, 0.0, 0.0],
    **load_amo_runners(
        "assets/models/g1/amo/amo_jit.pt",
        "assets/models/g1/amo/adapter_jit.pt",
        "assets/models/g1/amo/adapter_norm_stats.pt",
    ),
)
ctrl.reset()
```

`load_amo_runners` needs the `torch` extra. It returns the `forward_fn`,
`adapter_fn` and adapter normalization stats, and patches the CUDA device
constants baked into the released TorchScript file when running on CPU.

`state.dof_pos` / `dof_vel` carry all 23 observed joints. `target.position`
covers the 15 driven joints; the arms are not driven by AMO. Hold them at a pose
of your choice: their measured angles go into the observation, so the policy
balances for them.

The command has 7 entries (or 3, with the last 4 taken from
`default_command`):

| Index | Meaning | RoboJuDo joystick range (min / neutral / max) |
|---|---|---|
| 0 | linear velocity, from left stick Y | -1.0 / 0.0 / 1.0 |
| 1 | yaw command; the yaw term is `dyaw = -value[1]` | 0.2 / 0.0 / -0.2 (right stick X) |
| 2 | linear velocity, from left stick X | 0.8 / 0.0 / -0.8 |
| 3 | base height, m | 0.75 at rest; RoboJuDo allows 0.3 to 0.9 |
| 4, 5, 6 | torso yaw, pitch, roll, rad | 0 at rest |

RoboJuDo's config labels index 0 `vel_y` and index 2 `vel_x`; confirm the axis
directions in simulation before using them on hardware.

```python
# walk, then crouch to 0.6 m while pitching the torso forward
walk   = Reference(ReferenceKind.VELOCITY, value=np.array([0.0, 0.0, 0.4]))
crouch = Reference(ReferenceKind.VELOCITY,
                   value=np.array([0.0, 0.0, 0.0, 0.6, 0.0, 0.3, 0.0]))
```

When all three velocity entries are below 0.1 the gait clock locks both feet in
phase (standing); above that it switches them to antiphase. The lockstep check
covered standing, walking, turning, sidestepping, a crouch to 0.6 m with torso
pitch, and torso yaw / roll, for 24 s without a fall.

Two behaviours are copied from RoboJuDo as-is, because they are what the
checkpoint runs with: the joint velocities in the observation are not masked
(RoboJuDo computes an ankle / waist mask but does not apply it), and the height
entry in the demo observation is `0.75 + value[3]`.

## Gains

From RoboJuDo's G1 configs (the order is each controller's joint order):

| Controller | kp | kd |
|---|---|---|
| `unitree_loco` (gait) | legs `[100, 100, 100, 150, 40, 40] * 2` | `[2, 2, 2, 4, 2, 2] * 2` |
| `amo_loco` | legs `[150, 150, 150, 300, 80, 20] * 2`, waist `[400] * 3`, arms `[80, 80, 40, 60] * 2` | legs `[2, 2, 2, 4, 2, 1] * 2`, waist `[15] * 3`, arms `[2, 2, 1, 1] * 2` |
| `smooth_loco` | legs `[200, 150, 150, 200, 20, 20] * 2`, waist `200`, arms `[40] * 8` | legs `[5, 5, 5, 5, 4, 4] * 2`, waist `5`, arms `[10] * 8` |

`unitree_loco` with `with_gait=False`, in IsaacLab order:

```python
KP = [100, 100, 200, 100, 100, 40, 100, 100, 40, 150, 150] + [40] * 18
KD = [2, 2, 5, 2, 2, 5, 2, 2, 5, 4, 4, 10, 10, 2, 2, 10, 10, 2, 2] + [10] * 10
```

## Pitfalls

- **Command in m/s passed to `unitree_loco`.** The controller multiplies by
  `max_cmd`; `value=[0.8, 0, 0]` asks for 0.64 m/s, not 0.8.
- **`smooth_loco` neutral command.** Zero is not "stand"; use `[-0.5, 0, -0.5]`
  as RoboJuDo does.
- **AMO joint subset.** `default_pos` must equal `obs_default_pos[:15]`;
  the constructor raises otherwise. Pass all 23 joints in `state`.
- **LSTM state across episodes.** `ctrl.reset()` resets the hidden state only for
  a runner with a `reset()` (`load_torchscript_runner` has one); a hand-written
  `forward_fn` around a recurrent model must provide its own.

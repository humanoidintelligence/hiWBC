# Reference-motion tracking

English | [简体中文](./motion_tracking.zh-CN.md)

Controllers that make the robot follow a reference motion. They accept
`ReferenceKind.MOTION_FRAMES` and read the reference terms for the current tick
from `reference.frames`; some build those terms themselves from a motion clip or
from the motion in their checkpoint. Read [Getting started](getting_started.md)
first.

| Controller | Policy | G1 checkpoint | Joints | `frames` |
|---|---|---|---|---|
| `asap_tracker` | [ASAP](https://github.com/LeCAR-Lab/ASAP), KungfuBot v1 | `asap/mimic/<motion>/*.onnx` | 23 | none (motion baked in, a behavior) |
| `twist_tracker` | [TWIST](https://github.com/YanjieZe/TWIST) | `twist/twist_general_motion_tracker.pt` | 23 | `mimic_target`, or a motion clip |
| `beyondmimic_tracker` | [BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking) | `beyondmimic/{Jump_wose,Dance_wose,Violin,Waltz}.onnx` | 29 (IsaacLab) | `command`, `motion_anchor_ori_b`, (+ `motion_anchor_pos_b`, `base_lin_vel`); or the motion in the checkpoint, as a behavior |
| `kungfubot_tracker` | [KungfuBot2 / PBHC](https://github.com/TeleHuman/PBHC) | `kungfubot2/horse_test_43000.onnx` | 23 | anchor rotation, next-step and future-motion terms |
| `protomotions_tracker` | [Human2Humanoid](https://github.com/LeCAR-Lab/human2humanoid) student (ProtoMotions format) | `h2h/h2h.pt` | 21 | `task_obs` |
| `supermimic` | SuperMimic, unofficial ([supermimic_deploy](https://github.com/KyrieYoung926/supermimic_deploy)) | the bundle's `policy.onnx` | 29 (SDK order) | `goal`, or a motion clip |
| `blinkmimic` | [BlinkMimic](https://github.com/fuyichen0725/blinkmimic) | one you train: `blinkmimic/policy.pt`, or a single-motion `.onnx` export | 29 (SDK order) | `goal`, `ref_root_quat`, or a motion clip; an `.onnx` export plays its own motion, as a behavior |
| `blinkmimic` (`xmimic` variant) | [XMimic](https://github.com/nico-bohlinger/xmimic) | one you train: `xmimic/policy.npz` | 29 (SDK order) | as `blinkmimic` |

## Where the reference comes from

Each tracker needs reference terms for the current tick. There are three ways
to get them:

| Source | Controllers | What you do |
|---|---|---|
| A motion clip | `twist_tracker`, `supermimic`, `blinkmimic`, `kungfubot_tracker`, `protomotions_tracker` (H2H profile), `sonic` | Wrap a hiRobot `MotionClip` in a `ClipPlayer`; `ctrl.clip_reference(player, state)` builds the terms each tick |
| The checkpoint's own motion | `asap_tracker`, `beyondmimic_tracker`, `blinkmimic` (`onnx` variant) | `ctrl.start_behavior(name)` returns a source that builds the terms each tick |
| Your own code | all | Compute the terms and put them in `frames` under the names below |

The quickest start is a profile, which loads the checkpoint with the values it
was trained with (`checkpoint_dir` defaults to `$HIWBC_ROBOJUDO_MODELS`):

```python
import hiwbc
from hiwbc.references import ClipPlayer

ctrl = hiwbc.load("twist_tracker/unitree_g1", checkpoint_dir="RoboJuDo/assets/models/g1")
player = ClipPlayer(clip, ctrl.control_dt, **ctrl.clip_player_options)   # clip: hiRobot MotionClip, space "joint"
ctrl.check_clip(player)                      # ValueError if root_pos / root_quat / joints are missing
ctrl.reset()
while not player.done:
    state = read_state()
    send(ctrl.step(state, ctrl.clip_reference(player, state)))
    player.advance()
```

The player resamples the clip to 50 Hz, turns it to the robot's heading at the
first tick, computes velocities the clip does not have, and matches joints by
name ([API: Motion clips](../api.md#motion-clips)).

For your own code, these are the steps RoboJuDo's motion controllers
(`robojudo/controller/motion_*_ctrl.py`) do, and the reference for each
checkpoint:

1. **Advance the motion clock** (50 Hz, one frame per tick unless you change
   the play speed) and sample the reference frame(s) the policy needs.
2. **Align the motion to the robot**, usually by yaw (heading) and xy position
   at the start, so a clip recorded facing +x plays in the direction the robot
   faces.
3. **Compute the policy's reference terms**: joint targets, root height and
   velocity, and any term that needs body poses (an anchor orientation in the
   robot frame, key-body positions). The robot's own body poses come from FK on
   the measured joint angles.
4. Put the terms in `frames` under the names below.

The `kungfubot_tracker` key-body terms and the `protomotions_tracker` `task_obs`
need body FK. Their profiles use `hirobot.geometry.kinematics` on the G1 29-DoF MJCF
(option `mjcf`), imported only when a clip is tracked. Like PHC's motion library,
the player runs FK at the clip's own frame rate and smooths body and root
velocities (Gaussian, sigma = 2 source frames), then resamples to 50 Hz.

**Start pose.** Trackers expect the robot to start near the motion's first
frame. Ramp from the measured pose to that frame before the first `step`
(RoboJuDo: the policy's `get_init_dof_pos()`; a behavior: `BehaviorSpec.start_pos`),
then call `ctrl.reset()` (`start_behavior` resets for you).

None of these policies use action EMA except H2H (`action_beta=0.8`).

## Joint orders

**23-DoF** (`asap_tracker`, `twist_tracker`, `kungfubot_tracker`): legs (12),
`waist_yaw, waist_roll, waist_pitch`, then shoulder pitch / roll / yaw and elbow,
left then right. The 6 wrist joints are not driven.

**21-DoF** (`protomotions_tracker` for H2H): legs (12), `waist_yaw`, then the 8
arm joints.

**29-DoF** (`beyondmimic_tracker`): IsaacLab order, listed in the
[locomotion guide](locomotion.md#joint-orders). The ONNX file's metadata
(`joint_names`, `default_joint_pos`, `joint_stiffness`, `joint_damping`,
`action_scale`) is the source of truth:

```python
import onnxruntime as ort
sess = ort.InferenceSession("Jump_wose.onnx", providers=["CPUExecutionProvider"])
meta = sess.get_modelmeta().custom_metadata_map
floats = lambda k: [float(x) for x in meta[k].split(",")]
joint_names = meta["joint_names"].split(",")
default_pos, kp, kd = floats("default_joint_pos"), floats("joint_stiffness"), floats("joint_damping")
action_scale = floats("action_scale")
print(meta["observation_names"])   # e.g. command,motion_anchor_ori_b,base_ang_vel,...
```

**29-DoF SDK order** (`supermimic`, `blinkmimic`): `SUPERMIMIC_JOINT_NAMES` in
`hiwbc.controllers.supermimic`, the same as the Unitree SDK / MuJoCo order.

## `asap_tracker`

ASAP checkpoints have the motion baked in. The controller only keeps a phase,
`t / (50 * motion_length_s)`, so `frames` stays empty.

```python
import numpy as np
import onnxruntime as ort
from hiwbc import Reference, ReferenceKind, make_controller
from hiwbc.controllers.asap import AsapObsScales

G1_23_DEFAULT = [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0] * 2 + [0.0] * 11

sess = ort.InferenceSession("asap/mimic/CR7_level1/model_191500.onnx",
                            providers=["CPUExecutionProvider"])

def forward_fn(obs):
    return sess.run(["action"], {"actor_obs": obs[None]})[0].squeeze()

ctrl = make_controller(
    "asap_tracker",
    num_actions=23,
    default_pos=G1_23_DEFAULT,
    forward_fn=forward_fn,
    motion_length_s=3.967,          # CR7_level1
    use_history=True,
    history_length=4,
    obs_scales=AsapObsScales(base_ang_vel=0.25, dof_vel=0.05),
    action_scale=0.25,
    action_clip=100.0,
)
ctrl.reset()

ref = Reference(ReferenceKind.MOTION_FRAMES)
while not ctrl.motion_done:
    target = ctrl.step(read_state(), ref)
    send(target)
```

The observation is 380 values: last action, ang vel, joint pos, joint vel, the
4-tick history, gravity, phase. The history is **term-major, newest first**:
all 4 past last-actions, then all 4 past ang vels, and so on.

With the profile, the motion is a behavior (`cr7_level1`, `horse_squat` or
`dance_0605`, by variant) that takes `play_speed` in `(0, 2]`:

```python
ctrl = hiwbc.load("asap_tracker/unitree_g1", variant="default")   # CR7_level1
(spec,) = ctrl.behaviors
ramp_to(spec.start_pos)
source = ctrl.start_behavior(spec.name, play_speed=1.0)
while not source.done:
    state = read_state()
    send(ctrl.step(state, source.next(state)))
```

The phase advances by `reference.extra.get("play_speed", 1.0)` per tick; pass
`extra={"play_speed": 0.0}` to pause. Once the phase reaches 1 it stops and
`ctrl.motion_done` is `True`; switch to a standing policy then.

| Checkpoint | `motion_length_s` | Upper-body start pose (waist, arms) |
|---|---|---|
| `CR7_level1` | 3.967 | `0.200, 0.077, -0.288, 0.217, 0.156, -0.517, 0.578, 0.257, -0.250, 0.225, 0.513` |
| `kongfu/horse_squat` (KungfuBot v1) | 6.67 | `-0.0005, 0.0007, 0.0029, 0.206, 0.196, 0.005, 0.890, 0.208, -0.197, -0.002, 0.888` |
| `robomimic/dance_0605` | 18.0 | `0, 0, 0, 0.35, 0.18, 0, 0.87, 0.35, -0.18, 0, 0.87` |

The robot must start at the motion's first pose: the legs at default and the
upper body at the pose listed. Starting all joints at the default pose makes
CR7 fall. KungfuBot v1 uses default arms `0.2, 0.2, 0.0, 0.9` (left)
and `0.2, -0.2, 0.0, 0.9` (right) in its `default_pos`. Gains for the CR7
config:

```python
KP = [100, 100, 100, 200, 20, 20] * 2 + [400] * 3 + [90, 60, 20, 60] * 2
KD = [2.5, 2.5, 2.5, 5.0, 0.2, 0.1] * 2 + [5.0] * 3 + [2.0, 1.0, 0.4, 1.0] * 2
```

## `twist_tracker`

TWIST tracks a per-tick "mimic target" from a motion player or a teleop
retargeter.

```python
import torch

TWIST_DEFAULT = ([-0.2, 0.0, 0.0, 0.4, -0.2, 0.0] * 2 + [0.0] * 3
                 + [0.0, 0.2, 0.0, 1.2, 0.0, -0.2, 0.0, 1.2])
ankle_mask = np.zeros(23); ankle_mask[[4, 5, 10, 11]] = 1.0

model = torch.jit.load("twist/twist_general_motion_tracker.pt", map_location="cpu")

@torch.no_grad()
def forward_fn(obs):
    return model(torch.from_numpy(obs)[None]).numpy().squeeze()

ctrl = make_controller(
    "twist_tracker",
    num_actions=23,
    default_pos=TWIST_DEFAULT,
    forward_fn=forward_fn,
    history_length=10,
    ang_vel_scale=0.25, dof_pos_scale=1.0, dof_vel_scale=0.05,
    zero_dof_vel_mask=ankle_mask,   # TWIST zeroes ankle velocities in the obs
    action_scale=0.5,
    action_clip=10.0,
)
```

`ctrl.clip_reference(player, state)` builds `mimic_target` from a `joint`
clip with `root_pos` and `root_quat` (see
[Where the reference comes from](#where-the-reference-comes-from)). Checked
against RoboJuDo's `MotionTwistCtrl` on the Horse-stance clip for 150 ticks:
root height, roll / pitch / yaw and joints match exactly; the body-frame root
velocity differs by up to 0.014 m/s, because RoboJuDo smooths velocities with a
Gaussian filter (σ = 2 frames) and the player uses plain finite differences.
Pass a clip with smoothed velocities if you need RoboJuDo's.

`mimic_target` has 31 values, built from the reference frame (RoboJuDo
`MotionTwistCtrl`):

```python
# root_rot: [x, y, z, w]; root_vel, root_ang_vel: world frame; dof_pos: 23 joints
rpy = euler_xyz(root_rot)                               # roll, pitch, yaw
v_b = quat_rotate_inverse(root_rot, root_vel)
w_b = quat_rotate_inverse(root_rot, root_ang_vel)
dof_25 = np.insert(dof_pos, [19, 23], 0.0)              # zero wrist slots at 19 and 24
mimic = np.concatenate([[root_z], rpy, v_b, [w_b[2]], dof_25])   # 33 values
mimic_target = np.delete(mimic, [27, 32])               # drop the wrist slots -> 31

ref = Reference(ReferenceKind.MOTION_FRAMES, frames={"mimic_target": mimic_target})
```

The observation is the mimic target, the current proprio, and the previous 10
`[mimic_target, proprio]` blocks, oldest first: `11 * (31 + 74) = 1155` values.

```python
KP = [100, 100, 100, 150, 40, 40] * 2 + [150] * 3 + [40] * 8
KD = [2, 2, 2, 4, 2, 2] * 2 + [4] * 3 + [5] * 8
```

## `beyondmimic_tracker`

Each released checkpoint is trained on one motion, and its ONNX file outputs
that motion's reference (`joint_pos`, `joint_vel`, body poses) for a given
`time_step`. The profile exposes it as a behavior:

| Variant | Behavior | Frames (50 Hz) | State estimator |
|---|---|---|---|
| `dance_wose` (default) | `dance` | 6574 (131.5 s) | no |
| `jump_wose` | `jump` | 165 (3.3 s) | no |
| `violin` | `violin` | 612 (12.2 s) | yes: needs `state.base_lin_vel` |
| `waltz` | `waltz` | 945 (18.9 s) | yes: needs `state.base_lin_vel` |

The frame counts are measured: the first `time_step` from which all reference
outputs stop changing, plus one. RoboJuDo's own config caps Violin at 500 and
Waltz at 850 ticks.

```python
import hiwbc

ctrl = hiwbc.load("beyondmimic_tracker/unitree_g1", variant="jump_wose",
                  checkpoint_dir="RoboJuDo/assets/models/g1")
(spec,) = ctrl.behaviors                 # "jump"
ramp_to(spec.start_pos)                  # the motion's first pose, IsaacLab order
source = ctrl.start_behavior("jump")     # reset + RoboJuDo's warm-up forward
while not source.done:
    state = read_state()                 # 29 joints; base_lin_vel for violin / waltz
    send(ctrl.step(state, source.next(state)))
```

This follows RoboJuDo's `use_motion_from_model` mode with one forward per tick:
the observation reads the reference the previous forward returned, the forward
runs at the current frame, and the reference anchor heading is turned onto the
robot's at the first tick. The robot's torso orientation is
`state.extra["torso_quat"]` if you pass it, else the base orientation turned by
the three waist joints. The anchor position term is zero, as RoboJuDo's
`override_robot_anchor_pos` default makes it.

Checked in RoboJuDo's MuJoCo scene, hiWBC driving the physics, against
RoboJuDo computing the same tick: all four motions to their end plus 60 ticks
(Dance: 1000 ticks), no fall, max observation difference 3.3e-5 (Waltz; the
others 1e-5 or less) and max PD target difference 1.5e-5. The remaining
difference is the torso orientation: RoboJuDo takes it from MuJoCo's FK, hiWBC
from the waist joints.

To feed your own reference instead, build the controller directly:

```python
sess = ort.InferenceSession("beyondmimic/Jump_wose.onnx", providers=["CPUExecutionProvider"])
meta = sess.get_modelmeta().custom_metadata_map   # joint order, defaults, gains (above)
clock = {"t": 0}

def forward_fn(obs):
    return sess.run(["actions"], {"obs": obs[None],
                                  "time_step": np.array([[clock["t"]]], np.float32)})[0].squeeze()

ctrl = make_controller(
    "beyondmimic_tracker",
    num_actions=29,
    default_pos=default_pos,                 # from meta, IsaacLab order
    forward_fn=forward_fn,
    without_state_estimator=True,            # *_wose checkpoints; False for Violin / Waltz
    action_scale=np.asarray(action_scale),   # per joint, from meta
)
ctrl.reset()
```

Each checkpoint was trained on one motion, so a different reference will likely
make it fall. Reference terms per tick (RoboJuDo `BeyondMimicPolicy._get_command`):

| Term | How to get it |
|---|---|
| `command` (58) | `concat(joint_pos, joint_vel)` of the reference at `time_step`. Read it from the model's `joint_pos` / `joint_vel` outputs |
| `motion_anchor_ori_b` (6) | Reference anchor (`torso_link`) orientation relative to the robot's torso orientation, as the first two columns of the rotation matrix: `R(q_robot_torso⁻¹ * q_ref_anchor)[:, :2].reshape(-1)`. Align the reference yaw to the robot's start heading first |
| `motion_anchor_pos_b` (3) | Reference anchor position in the robot torso frame. Only when `without_state_estimator=False` |
| `base_lin_vel` (3) | Body-frame linear velocity from a state estimator. Only when `without_state_estimator=False` |

```python
q_rel = quat_mul(quat_conjugate(robot_torso_quat), ref_anchor_quat)   # [x, y, z, w]
anchor_ori_b = matrix_from_quat(q_rel)[:, :2].reshape(-1)
ref = Reference(ReferenceKind.MOTION_FRAMES,
                frames={"command": command, "motion_anchor_ori_b": anchor_ori_b})
target = ctrl.step(state, ref)
clock["t"] += 1
```

`quat_mul`, `quat_conjugate` and `matrix_from_quat` are in `hiwbc.math_utils`.

**Warm-up.** RoboJuDo runs one forward on a zero observation at reset, which
leaves a non-zero "last action" and starts the model's motion output.
`start_behavior` does this. With your own reference, run that forward yourself
and copy the action into `ctrl._last_action` after `ctrl.reset()` to match it
exactly; without it the first ticks differ slightly.

Observation sizes: 154 for `*_wose` (no state estimator), 160 otherwise. The
Waltz checkpoint showed a 3.7e-4 offset against RoboJuDo that did not grow over
1000 ticks; the others matched within 5e-6.

## `kungfubot_tracker`

KungfuBot2 takes three inputs, so `observe` returns a tuple.

```python
sess = ort.InferenceSession("kungfubot2/horse_test_43000.onnx",
                            providers=["CPUExecutionProvider"])

def forward_fn(obs):
    actor_obs, future, history = obs
    return sess.run(["action"], {"actor_obs": actor_obs[None],
                                 "future_motion_targets": future[None],
                                 "prop_history": history[None]})[0].squeeze()

ctrl = make_controller(
    "kungfubot_tracker",
    num_actions=23,
    default_pos=kungfubot_default,           # RoboJuDo g1_kungfubot_policy_cfg.py
    forward_fn=forward_fn,
    history_length=10,
    ang_vel_scale=0.25, dof_pos_scale=1.0, dof_vel_scale=0.05,
    roll_pitch_scale=1.0, actions_scale=1.0, history_scale=1.0,
    action_scale=np.asarray(kungfubot_action_scales),   # per joint, same file
    action_clip=100.0,
)
```

Input sizes: `actor_obs` 877, `future_motion_targets` 600, `prop_history` 740.

Reference terms come from the future reference frames (RoboJuDo
`MotionKungfuBotCtrl`): `future_motion_root_height`, `future_motion_roll_pitch`,
`future_motion_base_lin_vel` and `future_motion_base_yaw_vel` (both in the
reference root frame), `future_motion_dof_pos`, `next_step_ref_motion`, and
`anchor_ref_rot`, the reference root orientation relative to the robot base as
two rotation-matrix columns (6 values). Multiply each by its `obs_scales` entry
(all 1.0 for `horse_test_43000`).

```python
q_rel = quat_mul(quat_conjugate(state.base_quat), ref_root_quat)    # [x, y, z, w]
q_rel = q_rel[[1, 2, 3, 0]]   # horse_test_43000 only: trained with this reorder
anchor_ref_rot = matrix_from_quat(q_rel)[:, :2].reshape(-1)
```

The reorder reproduces a bug in the PBHC training code
([PBHC#68](https://github.com/TeleHuman/PBHC/issues/68)) that
`horse_test_43000` was trained with; RoboJuDo enables it with
`compatibility_old_version=True`. Drop it for checkpoints trained after the fix.

From a clip, `clip_reference` builds all seven terms. The future window is
frames `1, 5, 10, ..., 95` (`linspace(1, 95, 20)`, truncated). The key points in
`next_step_ref_motion` are the knees, ankles, and elbows, plus hand and head
points offset from the elbows and torso (PHC `unitree_g1_23dof_KungfuBot.yaml`).
Against RoboJuDo's `MotionKungfuBotCtrl` on its horse-stance clip, the max
difference was 6.7e-3 (joint angles, at the clip's end, where RoboJuDo's
interpolation clamps differently) and below 1.1e-3 for everything else. hiWBC's
window starts at the current frame; RoboJuDo's lags one frame, because it
computes tick t's window before advancing.

The history is term-major, newest first: all 10 past actions, then all 10 ang
vels, and so on. In the lockstep check this checkpoint fell at tick 404; it
falls at tick 405 under RoboJuDo alone.

## `protomotions_tracker` (H2H)

```python
model = torch.jit.load("h2h/h2h.pt", map_location="cpu")

ctrl = make_controller(
    "protomotions_tracker",
    num_actions=21,
    default_pos=[-0.1, 0.0, 0.0, 0.3, -0.2, 0.0] * 2 + [0.0] * 9,
    forward_fn=forward_fn,                   # wraps h2h.pt
    history_length=25,
    use_dof_pos_offset=False,                # H2H feeds absolute joint angles
    action_scale=0.25,
    action_clip=10.0,
    action_beta=0.8,
)
```

`task_obs` is the teleop imitation observation: reference key-body positions
(and velocities) relative to the robot's key bodies, in the robot's heading
frame. It needs FK of the robot's measured pose. RoboJuDo computes it with
`compute_imitation_observations_teleop_max_np` (`robojudo/policy/utils/utils_h2h.py`)
and clamps the head target to within 1 m of the robot horizontally.

From a clip, `clip_reference` builds `task_obs` (27 values) for the hands and
head (PHC `unitree_g1_29dof.yaml` extend_config). It needs the robot's world
position in `state.extra["base_pos"]`, and the clip starts under it. The robot's
own points are FK of `state.dof_pos` (29-DoF model, wrists at 0). The clip must
carry 29-DoF joint angles. Checked against RoboJuDo: reference points 1.5e-5,
velocities 1.3e-5 (default clip, playback speed 1), and `task_obs` 2.3e-7 for
200 random robot poses. RoboJuDo plays H2H clips at half speed after a fade-in
(`speed_target = 1/2`); hiWBC plays at 1x.

```python
KP = [100, 100, 100, 150, 40, 40] * 2 + [200] + [40] * 8
KD = [5, 5, 5, 5, 2, 2] * 2 + [6] + [2] * 8
```

## `supermimic`

```python
from hiwbc.controllers.supermimic import SuperMimicController, goal_features

ctrl = SuperMimicController.from_bundle("supermimic_deploy/assets")   # config.json + policy.onnx
ctrl.reset()

goal = goal_features(joint_pos, joint_vel, root_quat, root_lin_vel_w, root_ang_vel_w,
                     root_z, default_pos=ctrl.default_pos)            # (T, 68), whole clip
for t in range(len(goal)):
    ref = Reference(ReferenceKind.MOTION_FRAMES, frames={"goal": goal[t:t + 10]})
    target = ctrl.step(read_state(), ref)    # target.kp / kd set from the bundle
    send(target)
```

The bundle is the `assets/` directory of a
[supermimic_deploy](https://github.com/KyrieYoung926/supermimic_deploy)
checkout, with `policy.onnx` downloaded as its README describes (section 2). hiWBC
ships no SuperMimic weights. supermimic_deploy is MIT-licensed and its policy was
trained with [SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced)
(MIT); check those projects for the terms of the weights.

Or `hiwbc.load("supermimic/unitree_g1", checkpoint_dir="supermimic_deploy/assets")`,
and with a `joint` clip, `ctrl.clip_reference(player, state)` builds the 10 goal
frames with `goal_features`.

`from_bundle` checks the joint order, history length and the ONNX file's
sha256 against `config.json`. `goal_features` takes the reference clip at 50 Hz
(`root_quat` `[x, y, z, w]`, world-frame velocities) and returns one 68-d goal
per frame; pass frames `t .. t+9` each tick, and the controller repeats the last
one near the end of the clip. With `bound_torque=True` (default), the target is
re-expressed so the robot's PD torque stays within the bundle's effort limits.

## `blinkmimic`

[BlinkMimic](https://github.com/fuyichen0725/blinkmimic) (PyTorch, mjlab)
trains a G1 tracking policy through a learned world model, the method SuperMimic
introduced. Its observation and goal differ from `supermimic`'s: a BlinkMimic
policy does not run on `supermimic`, and the reverse. The same controller also
runs XMimic policies, which share BlinkMimic's interface ([XMimic](#xmimic)).

```python
import hiwbc
from hiwbc.references import ClipPlayer

ctrl = hiwbc.load("blinkmimic/unitree_g1",
                  checkpoint="model_2700.pt")                # the policy you trained
player = ClipPlayer(clip, ctrl.control_dt, **ctrl.clip_player_options)   # space "joint"
ctrl.check_clip(player)                      # needs root_pos, root_quat and joints
ctrl.reset()
while not player.done:
    state = read_state()
    send(ctrl.step(state, ctrl.clip_reference(player, state)))   # target.kp / kd set
    player.advance()
```

Through the `wbc` layer, the same profile plays a hiRobot `MotionClip` sent as
`MOTION_CLIP` ([robot guide](robot.md#3-commands)):

```yaml
# g1_blinkmimic.yaml
defaults: [robots/unitree_g1_mujoco_standing]
layers:
  - type: wbc
    profile: blinkmimic/unitree_g1   # checkpoint defaults to ~/.cache/hiwbc/blinkmimic/policy.pt
```

```python
from hirobot import Action, ControlMode, make_robot

robot = make_robot("g1_blinkmimic.yaml")
robot.connect()
play = Action(mode=ControlMode.MOTION_CLIP, motion=clip)
robot.send_action(play)
while not robot.get_status().command.done:
    robot.send_action(play)   # the same object: the clip keeps playing
robot.hold()
```

As with every tracker, the robot should start near the clip's first frame
(see **Start pose** above).

| Variant | Checkpoint (default `checkpoint`) | Needs | Notes |
|---|---|---|---|
| `default` | BlinkMimic `.pt` (`~/.cache/hiwbc/blinkmimic/policy.pt`) | `torch` extra | torch only reads the file; the network runs in numpy, fp32 |
| `xmimic` | XMimic `ckpt_*.npz` (`~/.cache/hiwbc/xmimic/policy.npz`) | numpy | See [XMimic](#xmimic) |
| `onnx` | a BlinkMimic `tools/export_onnx.py` export, one motion baked in (`~/.cache/hiwbc/blinkmimic/dance1_subject2.onnx`; pass your export's path) | `onnx` extra | Plays that motion as a behavior (`ctrl.behaviors`, `start_behavior`); cannot track other clips |

Profile options: `checkpoint` (path), `precision` (`xmimic` only, default
`"bf16"`), `providers` (`onnx` only, onnxruntime execution providers, default
CPU), `bound_torque` (default `True`: the target is re-expressed so the PD
torque stays within each joint's effort limit, as the training actuator clips
it). Default pose, action scale, gains and effort limits are BlinkMimic's G1
actuator model (`G1_*` in `hiwbc.controllers.blinkmimic`).

**Getting a checkpoint.** BlinkMimic publishes no trained weights and hiWBC
ships none. Train one with the upstream repository and pass its path as
`checkpoint=`: BlinkMimic reaches its training target in about 3 minutes on one
RTX 4090 (its README).

**Layout** (G1, 29 joints, SDK order, 50 Hz). The model takes two inputs:

- `obs_history` (990): 10 ticks, frame-major, oldest first, current tick
  included; after `reset()` all 10 slots hold the current entry. Each tick is
  99 values: `q - q0` (29), `dq` (29), base angular velocity in the body frame
  (3), projected gravity (3), previous action (29), and the reference root
  orientation relative to the robot root (6: the first two columns of
  `R^T R_ref`).
- `goal` (730): the next 10 reference frames, `f+1 .. f+10`, the last one
  repeated at the clip end. Each frame is 73 values: root displacement (3) and
  rotation (6D, 6) in the current reference frame's heading frame, root linear
  and angular velocity in that future frame's root frame (3 + 3), joint angles
  (29), joint velocities × 0.1 (29).

`frames` carries `goal` `(10, 73)` and `ref_root_quat` `(4,)`, the current
reference root orientation, `[x, y, z, w]`. `clip_reference` builds both; the
goal velocities are central differences of the clip resampled to 50 Hz, as the
trainers compute them. The model adds the next frame's reference joint angles to
its output (a residual stored in the checkpoint), and the target is
`q0 + action_scale * action`, with no EMA and no clip. Gains follow the
trainers' formula: `kp = armature * (2π · 10 Hz)²`,
`kd = 2 · 2.0 · armature · (2π · 10 Hz)` (damping ratio 2.0), `action_scale = 0.25 · effort / kp`.
XMimic uses the same layout.

The orientation term in `obs_history` compares the robot's world orientation
with the reference's, so the clip must face the way the robot faces. The clip
player turns the clip to the robot's heading at the first tick. On hardware,
IMU yaw drift goes straight into that term.

**The `onnx` variant.** BlinkMimic's `tools/export_onnx.py` bakes one motion's
goal into the graph, so the export plays that motion and cannot track other
clips:

```python
ctrl = hiwbc.load("blinkmimic/unitree_g1", variant="onnx",
                  checkpoint="dance1_subject2.onnx")   # your export
(spec,) = ctrl.behaviors                 # named after the metadata's motion field
ramp_to(spec.start_pos)                  # the motion's first pose, SDK order
source = ctrl.start_behavior(spec.name)  # no arguments; resets the controller
while not source.done:
    state = read_state()
    send(ctrl.step(state, source.next(state)))
```

- The behavior is named after the export's metadata `motion` field, or the file
  stem if there is none. It takes no arguments and lasts `(frames - 1) × 0.02`
  s.
- Playback follows upstream `tools/play_onnx.py`: one frame per tick, the
  previous action starting at zero. At the first tick the motion is turned, yaw
  only, onto the robot's heading.
- Through the `wbc` layer, `robot.run_layer_skill(spec.name)` starts it, and
  `WbcRobot` first ramps the robot to the motion's first pose
  ([robot guide](robot.md#3-commands)).
- Loading fails if the export's metadata joint names, gains, default pose or
  action scale differ from BlinkMimic's G1 values (relative tolerance 1e-5), or
  its history length, goal frames, control period or observation names differ
  from this controller's.

**On hardware.** `blinkmimic` has run in simulation only, with both BlinkMimic
and XMimic policies. Before a real robot,
note two things:

- The network runs in numpy. On a machine with many cores, BLAS threads can
  make the step time spike; set `OMP_NUM_THREADS=1` for the control process.
- BlinkMimic's G1 actuator model, which the policy was trained with, gives the
  ankles and waist roll / pitch a 50 N·m effort limit. hiRobot's G1 body config
  lists 35 N·m for those joints, so `bound_torque` clips those joints at a
  higher torque than the body config allows.

Checked against BlinkMimic with a checkpoint trained on LAFAN1 G1:

- **Policy inputs.** With identical inputs, the observation entry matches to 3e-7 and the goal to 1e-5. When
  hiWBC builds the goal itself from the 30 fps CSV, it differs by up to 1e-3 late in long clips. The cause is
  BlinkMimic resampling on a float32 time grid, where hiWBC uses float64.
- **Action.** The `.pt` runner matches BlinkMimic's fp32 policy to 2.4e-6. The `onnx` variant replays
  BlinkMimic's recorded 6573-tick rollout of its export to 4.4e-6.
- **Closed loop, training scene.** In BlinkMimic's training scene in MuJoCo, hiWBC stays upright on walk,
  dance and fall-and-get-up clips. Its mean joint error is within 6e-4 rad of BlinkMimic's own evaluation.
- **Closed loop, hiRobot.** Through `WbcRobot` on hiRobot's free-base G1, the policy plays 20 s of walking
  and 30 s of dancing without falling. This holds whether the robot starts at the default pose or at the
  clip's first frame.
- **Clip end.** Like the other trackers, the robot falls within about a second after the clip ends (see
  Pitfalls).

BlinkMimic is MIT-licensed. The checkpoints it trains derive from LAFAN1
(CC BY-NC-ND 4.0), so a trained policy is for non-commercial use only.

## XMimic

[XMimic](https://github.com/nico-bohlinger/xmimic) is a JAX rewrite of
BlinkMimic (paper: Bohlinger and Peters, "XMimic and XSoma: Learning Humanoid
Motion Tracking and Locomotion in Under a Minute", 2026,
[arXiv:2611.20452](https://arxiv.org/abs/2611.20452)). Its policies have
BlinkMimic's interface, so the [`blinkmimic`](#blinkmimic) controller runs them
through the `xmimic` variant; clips, layout, gains and the hardware notes are
the same.

```python
ctrl = hiwbc.load("blinkmimic/unitree_g1", variant="xmimic",
                  checkpoint="ckpt_5800.npz")      # the policy you trained
```

```yaml
# g1_xmimic.yaml
defaults: [robots/unitree_g1_mujoco_standing]
layers:
  - type: wbc
    profile: blinkmimic/unitree_g1
    variant: xmimic          # checkpoint defaults to ~/.cache/hiwbc/xmimic/policy.npz
```

- **Checkpoint.** An XMimic `ckpt_*.npz`. XMimic publishes no weights; its
  README reports 2.5 minutes of training on one H100 and 0.86 minutes on four.
  The file is read with numpy, no extra needed.
- **Precision.** `precision="bf16"` (default) rounds the matmul inputs to bf16
  as XMimic's evaluation does; `"fp32"` does not. `self_check` compares a
  runner with the test actions stored in the `.npz` ([API](../api.md#xmimic)).
- **Parity.** In fp32 the runner matches XMimic's policy to 6.4e-6. The bf16
  path can't be matched tick for tick: XMimic's own bf16 output changes by up
  to 0.023 between running a tick alone and in a batch. hiWBC reproduces the
  checkpoint's stored test actions to 1e-5. In XMimic's training scene hiWBC
  stays upright on walk, dance and fall-and-get-up clips, with mean joint error
  within 3e-4 rad of XMimic's own rollout; through `WbcRobot` it plays 20 s of
  walking and 30 s of dancing without falling.

XMimic is MIT-licensed; policies trained on LAFAN1 are for non-commercial use
only.

## Pitfalls

- **History order.** ASAP and KungfuBot2 histories are term-major and newest
  first. A frame-major history fits the input size but the policy falls within
  about 1 s.
- **Start pose.** Starting a tracker from the default pose instead of the
  motion's first frame can make it fall in the first second.
- **After the clip.** Trackers have no standing reference, so when a clip or
  behavior ends `WbcRobot` holds `default_pos` under plain PD, which does not
  balance. A free-base robot falls within about a second. Send the next clip
  before the current one ends.
- **Motion clock drift.** With your own BeyondMimic reference, `time_step` and
  the reference must refer to the same frame; advance both once per tick. The
  behavior does this for you.
- **Velocity smoothing.** By default `ClipPlayer` differentiates velocities at
  50 Hz without smoothing. `ClipPlayer(..., source_velocity_sigma=2.0)`
  differentiates at the clip's rate and smooths, as PHC does; the KungfuBot
  profile asks for it through `ctrl.clip_player_options` (the 50 Hz difference
  was off by 0.16 m/s on the horse-stance clip). TWIST and SuperMimic use the
  default; their profiles left it off after a 0.014 m/s check.
- **Anchor frame.** `motion_anchor_ori_b` and `anchor_ref_rot` are relative to
  the robot's measured torso / base orientation, recomputed every tick, not a
  fixed world frame.

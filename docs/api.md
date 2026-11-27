# hiWBC API reference

English | [简体中文](./api.zh-CN.md)

For step-by-step setups of each released policy, with the values it was
trained with, see the [guides](guides/README.md).

This page lists the interfaces hiWBC exposes: the contract types, the
`Controller` base class, the controller and profile registries, motion-clip
playback, behaviors, and each built-in controller's constructor and observation
layout. Names under
[Top-level package](#top-level-package) are importable as `from hiwbc import ...`.

hiWBC computes `(ControllerState, Reference) -> JointTarget`. A controller does no
hardware I/O: the caller reads the robot, calls `Controller.step` once per control
tick, and sends the returned target. The caller is either your own loop or
[`WbcRobot`](#wbcrobot-the-hirobot-layer), the hiRobot layer hiWBC registers.

General conventions:

- Arrays are `numpy.ndarray`, converted to `float32` inside the controllers.
  Angles are in rad, angular velocity in rad/s.
- Quaternions are `[x, y, z, w]`.
- Joint vectors (`dof_pos`, `default_pos`, `JointTarget.position`, gains) are in
  the controller's joint order, which is the order its policy was trained with.
  Mapping the robot's joint layout to it, by joint name, is the caller's job.
- The model is never loaded by a controller. Every controller takes an injected
  `forward_fn(obs) -> action`, so a stub can stand in for a checkpoint. A
  [profile](#profiles) loads the checkpoint and builds the controller with the
  values it was trained with.

## Contents

- [Top-level package](#top-level-package)
- [Quick example](#quick-example)
- [Contract types](#contract-types)
- [Controller](#controller)
- [Registry](#registry)
- [Profiles](#profiles)
- [Motion clips](#motion-clips)
- [Behaviors](#behaviors)
- [Forward functions](#forward-functions)
- [Remote inference](#remote-inference)
- [WbcRobot: the hiRobot layer](#wbcrobot-the-hirobot-layer)
- [Joint mapping](#joint-mapping)
- [Velocity-command controllers](#velocity-command-controllers)
- [Motion-tracking controllers](#motion-tracking-controllers)
- [SuperMimic](#supermimic)
- [BlinkMimic](#blinkmimic)
- [XMimic](#xmimic)
- [SONIC](#sonic)
- [echo](#echo)
- [Math helpers](#math-helpers)
- [Writing a controller](#writing-a-controller)

## Top-level package

| Name | Kind | Module | Purpose |
|---|---|---|---|
| `ControllerState` | dataclass | `hiwbc.types` | Proprioception read each tick |
| `Reference`, `ReferenceKind` | dataclass / enum | `hiwbc.types` | The command for this tick and how to read it |
| `JointTarget` | dataclass | `hiwbc.types` | PD position target, optional per-joint gains |
| `BehaviorSpec`, `ReferenceSource` | dataclass / protocol | `hiwbc.types` | A behavior a controller ships with, and the per-tick source it returns |
| `Controller` | abstract class | `hiwbc.controller` | `observe` → `forward` → shared post-processing |
| `register_controller`, `make_controller`, `available_controllers` | functions | `hiwbc.registry` | Name-based controller registry |
| `load`, `get_profile`, `available_profiles`, `register_profile`, `Profile` | functions / dataclass | `hiwbc.registry` | Profiles: a controller with a robot's trained values and checkpoint |
| `load_onnx_runner`, `load_torchscript_runner` | functions | `hiwbc.runtime` | Build a `forward_fn` from a checkpoint file |
| `WbcRobot` | class | `hiwbc.robot` | The hiRobot layer (`type: wbc`); imports `hirobot`, so not re-exported |
| `make_remote`, `RemoteFn`, `ForwardTimeout`, `RemoteError` | function / classes | `hiwbc.remote` | Run a controller's model forward on `hiwbc.serve` |
| `ModelServer` | class | `hiwbc.serve` | The forward service (`python -m hiwbc.serve`) |
| `__version__` | str | `hiwbc` | Package version |

Importing `hiwbc` imports `hiwbc.controllers`, which registers the built-in
controllers. Controller classes live in their modules:

| Module | Classes and helpers |
|---|---|
| `hiwbc.controllers.echo` | `EchoController` |
| `hiwbc.controllers.unitree` | `UnitreeLocoController`, `ObsScales` |
| `hiwbc.controllers.smooth` | `SmoothLocoController`, `SmoothObsScales` |
| `hiwbc.controllers.asap` | `AsapTrackerController`, `AsapObsScales` |
| `hiwbc.controllers.amo` | `AmoLocoController`, `load_amo_runners` |
| `hiwbc.controllers.twist` | `TwistTrackerController` |
| `hiwbc.controllers.beyondmimic` | `BeyondMimicTrackerController` |
| `hiwbc.controllers.protomotions` | `ProtoMotionsTrackerController` |
| `hiwbc.controllers.kungfubot` | `KungfuBotTrackerController` |
| `hiwbc.controllers.supermimic` | `SuperMimicController`, `goal_features`, `load_onnx_runner`, `SUPERMIMIC_JOINT_NAMES` |
| `hiwbc.controllers.blinkmimic` | `BlinkMimicController`, `goal_features`, `load_npz_runner`, `load_pt_runner`, `self_check`, `load_onnx_motion`, `BLINKMIMIC_JOINT_NAMES`, G1 tables (`G1_DEFAULT_POS`, `G1_ACTION_SCALE`, `G1_KP`, `G1_KD`, `G1_EFFORT`) |
| `hiwbc.controllers.sonic` | `SonicController`, `SonicRobotConfig`, `SonicObsConfig`, `ENCODER_MODE_IDS`, `load_onnx_runner`, `vr_3point`, `VR_3POINT_NAMES` |
| `hiwbc.controllers.sonic_planner` | `SonicPlanner`, `LocomotionMode`, `velocity_command`, `load_planner_runner` |
| `hiwbc.configs.blinkmimic.unitree_g1` | `check_onnx_metadata` |
| `hiwbc.configs.sonic.unitree_g1` | `SONIC_G1`, `SONIC_VARIANTS`, G1 tables (`G1_JOINT_NAMES`, `G1_JOINT_NAMES_MUJOCO`, `G1_DEFAULT_POS`, `G1_ACTION_SCALE`, `G1_KP`, `G1_KD`, ...) |
| `hiwbc.configs.<controller>.<robot>` | One profile module each (see [Profiles](#profiles)) |
| `hiwbc.references` | `ClipPlayer`, `require_clip` |
| `hiwbc.math_utils` | Quaternion and projected-gravity helpers |

## Quick example

```python
import numpy as np
from hiwbc import ControllerState, Reference, ReferenceKind, make_controller

n = 12
ctrl = make_controller(
    "unitree_loco",
    num_actions=n,
    default_pos=np.zeros(n),
    forward_fn=lambda obs: np.zeros(n, dtype=np.float32),  # stands in for a checkpoint
)
ctrl.reset()
state = ControllerState(
    dof_pos=np.zeros(n), dof_vel=np.zeros(n),
    base_quat=np.array([0.0, 0.0, 0.0, 1.0]),
    base_ang_vel=np.zeros(3),
)
ref = Reference(kind=ReferenceKind.VELOCITY, value=np.array([0.5, 0.0, 0.0]))
target = ctrl.step(state, ref)     # JointTarget
target.position                    # (12,) absolute PD position target
```

With RoboJuDo's G1 checkpoints on disk, a profile builds the same controller
with its trained values and gains:

```python
import hiwbc

ctrl = hiwbc.load("unitree_loco/unitree_g1", checkpoint_dir="RoboJuDo/assets/models/g1")
ctrl.joint_names, ctrl.kp          # controller joint order and gains
```

## Contract types

`hiwbc.types`.

### `ReferenceKind`

A `str` enum naming what the controller is commanded with.

| Member | Value | Payload |
|---|---|---|
| `VELOCITY` | `"velocity"` | `Reference.value = [vx, vy, wyaw]` |
| `MOTION_FRAMES` | `"motion_frames"` | `Reference.frames = {term: array}`, reference-motion terms for this tick |
| `LATENT` | `"latent"` | `Reference.value` = an opaque motion latent, e.g. a SONIC token |

### `ControllerState`

| Field | Shape | Description |
|---|---|---|
| `dof_pos` | `(num_dofs,)` | Joint positions |
| `dof_vel` | `(num_dofs,)` | Joint velocities |
| `base_quat` | `(4,)` | Base orientation, `[x, y, z, w]`, world frame |
| `base_ang_vel` | `(3,)` | Base angular velocity, body frame |
| `base_lin_vel` | `(3,)` or `None` | Base linear velocity, body frame. Optional; read only by the BeyondMimic behaviors of checkpoints with a state estimator |
| `extra` | `dict` | Free-form. `beyondmimic_tracker` reads `extra["torso_quat"]` if present |

### `Reference`

| Field | Type | Description |
|---|---|---|
| `kind` | `ReferenceKind` | Selects how the other fields are read |
| `value` | array or `None` | `VELOCITY` command or `LATENT` token |
| `frames` | `dict[str, array]` or `None` | `MOTION_FRAMES` payload. Keys depend on the controller |
| `extra` | `dict` | Per-tick options, e.g. `play_speed` (ASAP), `encoder_mode` / `play` (SONIC), `time_step` (BeyondMimic behavior) |

### `JointTarget`

| Field | Shape | Description |
|---|---|---|
| `position` | `(num_actions,)` | Absolute PD position target |
| `kp`, `kd` | `(num_actions,)` or `None` | Per-joint gains, the controller's `kp` / `kd`. `None` means keep the robot's configured gains. Every profile sets them; a controller built with `make_controller` without `kp` / `kd` leaves them `None` |
| `extra` | `dict` | Controller-specific outputs. `sonic` puts the token it used in `extra["motion_token"]` |

## Controller

`hiwbc.controller.Controller`, an abstract base class.

```python
Controller(*, num_actions, default_pos, action_scale=1.0, action_clip=None, action_beta=1.0,
           joint_names=None, obs_joint_names=None, kp=None, kd=None, control_dt=0.02,
           cartesian_targets=None, stand_command=None)
```

| Argument | Description |
|---|---|
| `num_actions` | Length of the action and of `JointTarget.position` |
| `default_pos` | `(num_actions,)` default pose, added to the scaled action |
| `action_scale` | Positive scalar, or a `(num_actions,)` array of per-joint scales |
| `action_clip` | Positive bound for `clip(action, -c, c)`, or `None` for no clip |
| `action_beta` | EMA weight in `(0, 1]`. `1.0` means no smoothing |
| `joint_names` | Driven joints in the controller's order, hiRobot spelling (no `_joint` suffix). `None` in bare test builds; every profile sets it |
| `obs_joint_names` | Joints `ControllerState.dof_pos` carries, if different from `joint_names` (AMO reads the arms it does not drive). Default: `joint_names` |
| `kp`, `kd` | `(num_actions,)` non-negative gains, returned with every target. `None`: the robot keeps its own |
| `control_dt` | Control period, s |
| `cartesian_targets` | End-effector name -> controller target name, for controllers that take Cartesian targets. Only `sonic` with its planner sets it (`left_hand` -> `left_wrist`, `right_hand` -> `right_wrist`) |
| `stand_command` | The `VELOCITY` command that means "stand still" for this checkpoint (not always zero) |

The constructor raises `ValueError` for a wrongly shaped `default_pos`,
`action_scale`, `kp` or `kd`, a non-positive `action_scale`, `action_clip` or
`control_dt`, an `action_beta` outside `(0, 1]`, `joint_names` of the wrong
length or with a repeat, or `obs_joint_names` without `joint_names`.

| Member | Description |
|---|---|
| `accepts` | Class attribute: tuple of `ReferenceKind`s the controller handles. Empty means any |
| `observe(state, reference)` | Abstract. Build the policy input: a flat array, or a tuple of arrays for multi-input models |
| `forward(obs)` | Abstract. Run the model and return the raw action |
| `step(state, reference) -> JointTarget` | One control tick (below) |
| `reset()` | Clear per-episode state. The base clears the last action and calls `reset()` on every injected `*_fn` that has one (a recurrent model's hidden state); subclasses also clear clocks and history |
| `motion_spaces` | `MotionClip.space` values the controller can track (`"joint"`, `"smpl"`); empty means it takes no clips. See [Motion clips](#motion-clips) |
| `check_clip(player)` | Raise `ValueError` if the clip lacks what the controller reads |
| `clip_reference(player, state) -> Reference` | This tick's reference from a `ClipPlayer` |
| `stand_reference(state) -> Reference` | The reference that holds the robot standing: `stand_command` for velocity controllers, the default pose for `sonic` (zero `VELOCITY` with the planner). Raises `NotImplementedError` for the other trackers |
| `cartesian_rest() -> dict` | Target name -> `[x, y, z, qx, qy, qz, qw]` the controller tracks when no Cartesian target is commanded. `{}` by default |
| `behaviors` | `list[BehaviorSpec]` the controller ships with. See [Behaviors](#behaviors) |
| `start_behavior(name, **args) -> ReferenceSource` | Start one; `KeyError` for an unknown name |
| `profile`, `variant` | Set by `hiwbc.load`: the profile and variant that built it |

`step` does, in order:

1. Raises `ValueError` if `accepts` is non-empty and `reference.kind` is not in it.
2. `obs = observe(state, reference)`.
3. `raw = forward(obs)`, flattened. Raises `ValueError` unless it has `num_actions` values.
4. Post-processing, in RoboJuDo's `get_action` order:
   1. `action = (1 - action_beta) * last_action + action_beta * raw`
   2. `last_action = action` (before clip and scale; this is the "last action"
      term the observations read)
   3. `action = clip(action, -action_clip, action_clip)` if `action_clip` is set
   4. `action = action * action_scale`
5. Returns `JointTarget(position=action + default_pos, kp=kp, kd=kd)`.

Call `reset()` at the start of every episode. Controllers with a clock (gait
phase, motion phase) advance it after each `step`, not inside `observe`.

## Registry

`hiwbc.registry`. Controllers by name; [profiles](#profiles) are the usual way
to build one for a robot.

| Function | Description |
|---|---|
| `register_controller(name)` | Decorator for a factory `(**kwargs) -> Controller`. Raises `ValueError` if `name` is taken |
| `make_controller(name, **kwargs) -> Controller` | Calls the factory with `kwargs`. Raises `KeyError` listing the available names |
| `available_controllers() -> list[str]` | Sorted registered names |

Built-in names: `echo`, `unitree_loco`, `smooth_loco`, `amo_loco`,
`asap_tracker`, `twist_tracker`, `beyondmimic_tracker`, `protomotions_tracker`,
`kungfubot_tracker`, `supermimic`, `blinkmimic`, `sonic`. Every factory passes its keyword arguments
straight to the class constructor.

## Profiles

A profile is one controller on one robot: `<controller>/<robot>`. Its module,
`hiwbc/configs/<controller>/<robot>.py`, holds the values the checkpoint was
trained with (joint order, default pose, gains, scales, history lengths) and
loads the checkpoint. `hiwbc.load` builds a ready controller from it.

```python
import hiwbc

hiwbc.available_profiles()      # ['amo_loco/unitree_g1', ..., 'unitree_loco/unitree_g1']
ctrl = hiwbc.load("beyondmimic_tracker/unitree_g1", variant="jump_wose",
                  checkpoint_dir="RoboJuDo/assets/models/g1")
```

| Function | Description |
|---|---|
| `load(profile, **options) -> Controller` | Build the controller. `options` are `variant` plus the profile's `default_options`; another key raises `TypeError`, an unknown variant `ValueError`. The checkpoint is loaded here, so a missing file fails now. `runtime="remote"`, `endpoint`, `remote_timeout_s` run the forward on a server ([Remote inference](#remote-inference)) |
| `get_profile(name) -> Profile` | The `Profile` (`name`, `variants`, `default_variant`, `default_options`, `description`). Imports its module on first use |
| `available_profiles() -> list[str]` | Shipped and registered profiles. Lists module names without importing them, so no model runtime is loaded |
| `register_profile(name, *, build, variants, default_options=None, description="")` | Register a profile from a `configs/` module. `build(variant_data, options) -> Controller` |

`default_variant` is `"default"` if there is one, else the first listed.

Shipped profiles (all for the 29-DoF Unitree G1; the controller may drive a
subset of the joints):

| Profile | Variants | Options besides `variant` | Behaviors | Clip spaces |
|---|---|---|---|---|
| `unitree_loco/unitree_g1` | `default` (gait, 12 legs), `wo_gait` (29, IsaacLab order) | `checkpoint_dir`, `device` | | |
| `smooth_loco/unitree_g1` | `default` | `checkpoint_dir`, `device` | | |
| `amo_loco/unitree_g1` | `default` | `checkpoint_dir`, `device` | | |
| `asap_tracker/unitree_g1` | `default` (CR7), `horse_squat`, `dance_0605` | `checkpoint_dir`, `providers` | the baked-in motion | |
| `twist_tracker/unitree_g1` | `default` | `checkpoint_dir`, `device` | | `joint` |
| `beyondmimic_tracker/unitree_g1` | `dance_wose` (default), `jump_wose`, `violin`, `waltz` | `checkpoint_dir`, `providers` | `dance`, `jump`, `violin`, `waltz` (one per variant) | |
| `kungfubot_tracker/unitree_g1` | `default` | `checkpoint_dir`, `providers` | | `joint` |
| `protomotions_tracker/unitree_g1` | `default` | `checkpoint_dir`, `device` | | `joint` |
| `supermimic/unitree_g1` | `default` | `checkpoint_dir`, `check_hash`, `providers`, `bound_torque` | | `joint` |
| `blinkmimic/unitree_g1` | `default` (BlinkMimic `.pt`), `xmimic` (XMimic `.npz`), `onnx` (one motion baked in) | `checkpoint`, `precision` (`xmimic`, default `"bf16"`), `providers` (`onnx`), `bound_torque` (default `True`) | the baked-in motion (`onnx`) | `joint` (`default`, `xmimic`) |
| `sonic/unitree_g1` | `default`, `low_latency`, `sonic_v1_1` | `checkpoint_dir`, `providers`, `encoder`, `encoder_mode`, `planner` (default `True`), `planner_thread` (default `False`) | | `joint`, `smpl` (with the encoder) |
| `echo/default` | `default` | `default_pos`, `joint_names`, `kp`, `kd` | | |

`checkpoint_dir` defaults: RoboJuDo's G1 models directory,
`$HIWBC_ROBOJUDO_MODELS` or `~/.cache/hiwbc/robojudo/g1`, for the
RoboJuDo-deployed checkpoints; `~/.cache/hiwbc/sonic` for SONIC;
`~/.cache/hiwbc/supermimic/assets` for SuperMimic. BlinkMimic takes a file,
`checkpoint`: `~/.cache/hiwbc/blinkmimic/policy.pt` (`default`),
`~/.cache/hiwbc/xmimic/policy.npz` (`xmimic`, [XMimic](#xmimic)) or
`~/.cache/hiwbc/blinkmimic/dance1_subject2.onnx` (`onnx`). Neither project
publishes weights, so pass `checkpoint=` with the path to a policy or export you
trained ([BlinkMimic](#blinkmimic)).

## Motion clips

`hiwbc.references.ClipPlayer` plays a hiRobot `MotionClip` (or any object with
the same fields: `space`, `fps`, `joint_names`, `joint_pos`, `joint_vel`,
`smpl_joints`, `root_pos`, `root_quat`) for a controller whose `motion_spaces`
contains the clip's `space`.

```python
from hiwbc.references import ClipPlayer

player = ClipPlayer(clip, ctrl.control_dt, **ctrl.clip_player_options)
ctrl.check_clip(player)                      # ValueError if the clip lacks what ctrl reads
while not player.done:
    state = read_state()
    target = ctrl.step(state, ctrl.clip_reference(player, state))
    send(target)
    player.advance()
```

The player:

- resamples the clip from `fps` to `control_dt`: `root_quat` by slerp, the rest
  linearly;
- on the first `clip_reference`, rotates the clip about z so its first root
  heading matches the robot's base heading, and moves its first root position
  to x, y = 0 (SONIC's `apply_delta_heading`), or under the robot when
  `window(..., base_pos=...)` passes its world position (H2H does);
- repeats the last frame for windows that run past the end;
- differentiates `joint_vel` from `joint_pos`, and root linear / angular
  velocities from `root_pos` / `root_quat`, when the clip has none (plain finite
  differences at the control rate, no smoothing; with
  `source_velocity_sigma=s`, root velocities are differentiated at the clip's
  rate and Gaussian-smoothed over `s` source frames first, as PHC does);
- computes terms registered with `player.add_derived(name, fn)` on each clip at
  its own rate, after alignment, and resamples them; `window` returns them
  under `name` (KungfuBot and H2H register body FK this way);
- matches clip joints to the controller's by name (a `_joint` suffix is
  dropped) and fills joints the clip lacks with the controller's default pose
  (zero velocity).

`player.append(clip)` queues a clip with the same space and fields after the
current one, for streaming. `done`, `frame`, `num_frames`, `duration` and `time`
report progress.

| Controller | Spaces | Clip must have | Reference built |
|---|---|---|---|
| `sonic` | `joint`, `smpl` (encoder loaded) | `root_quat`; `joint` also one of the controller's joints, plus `root_pos` for a variant whose `joint` mode reads the root height (none of the released ones) | The encoder window from the current frame, `encoder_mode` set to the clip's space |
| `twist_tracker` | `joint` | `root_pos`, `root_quat`, one of the controller's joints | The 31-value `mimic_target` of the current frame |
| `supermimic` | `joint` | `root_pos`, `root_quat`, one of the controller's joints | 10 goal frames from `goal_features` |
| `blinkmimic` | `joint` (`default`, `xmimic` variants) | `root_pos`, `root_quat`, one of the controller's joints | `goal` (frames `f+1 .. f+10`) and `ref_root_quat` |
| `kungfubot_tracker` | `joint` (with `future_steps`, `kinematics`, `key_bodies`, as in the profile) | `root_pos`, `root_quat`, one of the controller's joints | All seven terms: 20 future frames, the next step with key points, `anchor_ref_rot` |
| `protomotions_tracker` | `joint` (with `kinematics`, `track_points`, as in the H2H profile) | `root_pos`, `root_quat`, one of the controller's joints; `state.extra["base_pos"]` each tick | `task_obs` for the hands and head |

`kungfubot_tracker` and `protomotions_tracker` need `hirobot` installed for FK.
`ClipPlayer` itself does not. The other controllers take no clips.
`asap_tracker`, `beyondmimic_tracker` and a `blinkmimic` ONNX export have their
motion in the checkpoint and play it as a [behavior](#behaviors).

## Behaviors

A behavior is something a controller can do without a reference from above,
such as the motion trained into an ASAP or BeyondMimic checkpoint. `WbcRobot`
lists them as layer skills (`list_layer_skills` / `run_layer_skill`).

```python
(spec,) = ctrl.behaviors                    # BehaviorSpec
ramp_to(spec.start_pos)                     # the motion's first pose, controller order
source = ctrl.start_behavior(spec.name)     # resets the controller
while not source.done:
    state = read_state()
    send(ctrl.step(state, source.next(state)))
```

`BehaviorSpec` fields: `name`, `description`, `parameters` (JSON Schema of the
`start_behavior` keyword arguments), `start_pos` (joint pose the robot must be
in at the start, or `None`). A `ReferenceSource` has `done` and
`next(state) -> Reference`. After `done`, switch to another controller or
reference; the controller keeps feeding the last frame if stepped further.

| Controller | Behavior | Arguments | Ends |
|---|---|---|---|
| `asap_tracker` | `motion_name` (a profile sets it per variant) | `play_speed` in `(0, 2]`, default 1 | When the phase reaches 1 |
| `beyondmimic_tracker` | `motion_name` (`dance`, `jump`, `violin`, `waltz`) | none | After `motion_frames` ticks |
| `blinkmimic` (`onnx` variant) | the export's metadata `motion` field, else the file stem | none | After `frames - 1` ticks, `(frames - 1) × 0.02` s |

## Forward functions

A forward function takes what `observe` returns and returns the raw action, any
array-like with `num_actions` values. Two loaders build one from a checkpoint
file:

```python
from hiwbc import load_onnx_runner, load_torchscript_runner

forward_fn = load_torchscript_runner("policy.pt")          # needs the torch extra
forward_fn = load_onnx_runner("policy.onnx")               # needs the onnx extra
```

`load_torchscript_runner(path, *, device="cpu")` adds a batch dimension, runs
the model under `torch.no_grad()` and returns the output flattened. A tuple
observation is passed as separate positional arguments.

`load_onnx_runner(path, *, input_names=None, output_name=None, extra_inputs=None,
providers=None, num_threads=None)` feeds each array as `(1, -1)` float32:

| Argument | Default | Description |
|---|---|---|
| `input_names` | model inputs not in `extra_inputs`, in model order | Inputs fed from the observation, in the order `observe` returns them |
| `output_name` | first output | Output to return |
| `extra_inputs` | `None` | Constant inputs fed every call, name -> array |
| `providers` | `["CPUExecutionProvider"]` | onnxruntime providers |
| `num_threads` | onnxruntime default | Intra- and inter-op threads; 1 gives the steadiest latency |

For example, `kungfubot_tracker` returns `(actor_obs, future_motion_targets,
history)`, which matches its ONNX input order, so `load_onnx_runner(path)` works
as is. BeyondMimic's ONNX also takes a `time_step` input and has several
outputs; for references built by the caller, feed `time_step` 0 (the released
models' `actions` output does not depend on it):

```python
forward_fn = load_onnx_runner(
    "Dance_wose.onnx", input_names=("obs",), output_name="actions",
    extra_inputs={"time_step": np.zeros((1, 1), np.float32)},
)
```

hiWBC does not pin a torch or onnxruntime version. The `torch` and `onnx`
extras install them; the core needs only numpy, and `import hiwbc` imports
neither.

## Remote inference

`hiwbc.load(profile, runtime="remote", endpoint="tcp://host:port",
remote_timeout_s=0.015, ...)` builds the controller locally, as usual, then
replaces each of its model callables with a `RemoteFn` that calls a server
running the same profile. Building the observation, the history and the
post-processing stay local; only the forward pass is remote. The checkpoints
must be on both machines.

On the server:

```bash
python -m hiwbc.serve --profile sonic/unitree_g1 [--variant default] \
    [--host 127.0.0.1] [--port 5555] [--option key=value ...]
```

`--option` values are parsed as Python literals (`planner=False`,
`checkpoint_dir=/data/sonic`); `providers=CUDAExecutionProvider` becomes a list.

| Name | Description |
|---|---|
| `make_remote(controller, endpoint, *, timeout_s=0.015, planner_timeout_s=0.5, connect=True) -> dict[str, RemoteFn]` | Swap the model callables of a controller built by `hiwbc.load` (`*_fn` attributes, plus SONIC's planner as `"planner"`). Connects each one and checks the server's profile and variant unless `connect=False`. Raises `ValueError` for a recurrent model |
| `RemoteFn(endpoint, *, name, profile, variant, timeout_s)` | `fn(*args)` runs the server's model `name`. Each call, reconnecting included, has `timeout_s`. A timeout, a closed connection or a refused connect raises `ForwardTimeout` and drops the socket; the next call reconnects. `calls`, `reconnects` count |
| `ForwardTimeout` | Subclass of `TimeoutError`. `WbcRobot` treats it as a missed tick |
| `RemoteError` | The server refused the connection (other profile, variant or model) or the model raised on the server |
| `ModelServer(controller, host="127.0.0.1", port=5555)` | The service; `start()` runs it on a thread, `serve_forever()` blocks, `stop()` closes every connection. `port=0` picks a free port (`server.port`) |

- **Stateless.** The server only runs forward passes, so a client can reconnect
  at any time. TorchScript models with buffers (the `unitree_loco` `default`
  LSTM) are refused by `make_remote` and `ModelServer`.
- **Wire format.** TCP, one request and one reply per call. Each message is
  `[u32 length][u32 header length][JSON header][raw array bytes]`, big-endian.
  Arrays travel as raw C-order bytes with a dtype from a fixed list (`float32`,
  `float64`, `int64`, `int32`, `bool`); nothing is unpickled. Messages are
  limited to 16 MB. The first message names the protocol version, profile,
  variant and model; the server refuses a mismatch.
- **No authentication.** Bind to localhost or a trusted wired network.

## WbcRobot: the hiRobot layer

`hiwbc.robot.WbcRobot(inner, profile=None, *, controller=None, variant=None,
**options)` is a hiRobot `Robot` that runs a controller on `inner` (the robot
underneath: its safety gate over the assembly). hiWBC registers it as the `wbc` layer through the
`hirobot.drivers` entry point, so `make_robot` builds it from a
`layers: [{type: wbc, profile: ...}]` config entry; the remaining keys of the
entry are its keyword arguments. The [robot guide](guides/robot.md) walks
through a full setup and lists the options.

| Member | Description |
|---|---|
| `action_modes` | From the controller: `BASE_VELOCITY` (accepts `VELOCITY`), `MOTION_LATENT` (`LATENT`), `MOTION_CLIP` (`motion_spaces`), `CARTESIAN` (`cartesian_targets` and `VELOCITY`), `BASE_POSE` (`BASE_VELOCITY` and the robot reports `base_position`) |
| `direct_joints` | Joints of `inner` the controller does not drive; command them with `Action.joint_names` + `joint_target` beside any mode |
| `cartesian_targets`, `motion_spaces` | From the controller |
| `connect()`, `reset()` | Ramp to `default_pos` over `start_seconds`, then run the controller. A real `inner` gets a thread at `1 / control_dt` |
| `send_action(action)` | Set the command. In simulation, also run one outer period of ticks, so repeating the last command advances simulated time |
| `hold()` | Stand still in place |
| `get_status().command` | hiRobot's `CommandProgress` for the caller's last `BASE_POSE` / `MOTION_CLIP` / `BASE_VELOCITY` / `MOTION_LATENT` / `CARTESIAN` command: `done` (arrived, clip played out; `None` for modes with no end), `goal`, `clip_time` / `clip_duration` (whole `append` queue). Kept after it ends until the next command; `None` before the first, after `hold()`, and while a layer skill runs ([guide](guides/robot.md#5-status)) |
| `get_status().extra["wbc"]` | `profile`, `variant`, `mode`, clip progress, `goal`, `arrived`, `ee_targets`, `ticks`, `missed_ticks`, `late_ticks`, `step_ms_max` |
| `list_layer_skills()`, `run_layer_skill(name, args=None)`, `stop_layer_skill()` | `stand` plus the controller's [behaviors](#behaviors), run by the layer (hiRobot's [layer skills](https://github.com/humanoidintelligence/hiRobot/blob/main/docs/api.md#layers)) |
| `list_robot_skills()`, `run_robot_skill(name, args=None)`, `stop_robot_skill()` | hiRobot's one list: these layer skills, then the vendor's `list_onboard_skills()` |

Per tick the reference is, in order: a running behavior, a playing clip, the
last `BASE_VELOCITY` / `MOTION_LATENT` / `CARTESIAN` command if newer than
`timeout_s`, a `BASE_POSE` goal being walked to, else
`Controller.stand_reference`. A controller with no standing reference holds
`default_pos` under PD.

An exception in a tick puts the layer in the fault state and e-stops the stack
(hiRobot's `Layer.estop`: the robot at the bottom first, which damps a Unitree
or MuJoCo body, kp 0; then the layer stops its thread): `send_action` raises
`RobotFaultError` until `reset()`. A
`ForwardTimeout` (remote inference) is not an exception of that kind: the tick
sends the previous target again, and only more than `max_missed_ticks` in a row
(or a timeout on the first tick) damps. Body joint state older than `stale_s`
(`inner.now() - obs.stamps["body"].t`, default 0.1 s) is an exception of that
kind, checked on every tick including the start ramp.

The layer uses `Controller.stand_reference` for `hold()`, after a clip or
behavior, and on a command timeout, and `Controller.cartesian_rest` as the pose
`CARTESIAN` rate limiting starts from and returns to.

## Joint mapping

Each controller works in its own joint order (e.g. SONIC uses IsaacLab order,
see `hiwbc.configs.sonic.unitree_g1.G1_JOINT_NAMES`). Joint names are hiRobot's
(the G1's have no `_joint` suffix), so hiRobot's `DofMap` reorders vectors by
name:

```python
from hirobot import DofMap
from hiwbc.configs.sonic.unitree_g1 import G1_JOINT_NAMES

to_ctrl = DofMap(robot.embodiment.joint_names, G1_JOINT_NAMES)
dof_pos = to_ctrl.remap(obs.joint_pos)                  # robot order -> controller order
q_cmd = to_ctrl.scatter(target.position, default=hold)  # controller order -> robot order
```

The controller may use a subset of the robot's joints; `scatter` fills the
others from `default`. `DofMap` raises `ValueError` if a controller joint is not
on the robot.

## Velocity-command controllers

All accept `ReferenceKind.VELOCITY`. `unitree_loco` and `smooth_loco` take
`value` of shape `(3,)`; another shape, or `value=None`, raises `ValueError`.
They keep a gait clock, `phase = timestep / (control_hz * cycle_time)`, that
advances by one tick per `step` and restarts on `reset()`, and default to
`action_beta=0.8` and `action_scale=0.25`. [`amo_loco`](#amo_loco) takes a 3-
or 7-value command.

### `unitree_loco`

`UnitreeLocoController`, after RoboJuDo's `UnitreePolicy`
([unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym); set
`with_gait=False` for [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab)).

| Argument | Default | Description |
|---|---|---|
| `num_actions`, `default_pos`, `forward_fn` | required | |
| `control_hz` | `50` | Control rate; sets the clock step |
| `cycle_time` | `0.8` | Gait period, s |
| `with_gait` | `True` | Append the gait sin / cos terms |
| `obs_scales` | `ObsScales()` | `dof_pos=1.0`, `dof_vel=0.05`, `ang_vel=0.25`, `command=(2.0, 2.0, 0.25)` |
| `max_cmd` | `(0.8, 0.5, 1.57)` | Per-axis command multiplier |
| `history_length` | `0` | Ticks kept per term; needs `with_gait=False` (unitree_rl_lab: `5`) |
| `gravity_scale` | `1.0` | Multiplier on projected gravity |
| `action_scale`, `action_clip`, `action_beta` | `0.25`, `None`, `0.8` | See [Controller](#controller) |

Observation, a flat vector of `9 + 3 * num_actions (+ 2)` values:

| # | Term | Width |
|---|---|---|
| 1 | `base_ang_vel * ang_vel` | 3 |
| 2 | projected gravity | 3 |
| 3 | `value * command * max_cmd` | 3 |
| 4 | `(dof_pos - default_pos) * dof_pos` | n |
| 5 | `dof_vel * dof_vel` | n |
| 6 | last action | n |
| 7 | `sin(2π·phase)`, `cos(2π·phase)` (only with `with_gait`) | 2 |

With `history_length > 0` (RoboJuDo's `UnitreeWoGaitPolicy`), terms 1 to 6 are
each kept for `history_length` ticks, current tick included, and laid out
term-major, oldest first: all ang-vel frames, then all gravity frames, and so
on. The buffer is zero-filled at construction and on `reset()`.

### `smooth_loco`

`SmoothLocoController`, after RoboJuDo's `SmoothPolicy`
([Smooth Humanoid Locomotion](https://github.com/zixuan417/smooth-humanoid-locomotion)).

| Argument | Default | Description |
|---|---|---|
| `num_actions`, `default_pos`, `forward_fn` | required | |
| `control_hz`, `cycle_time` | `50`, `0.8` | As for `unitree_loco` |
| `n_priv_latent` | `0` | Width of the zero-filled privileged-latent slot |
| `history_length` | `0` | Number of past proprio blocks appended |
| `obs_scales` | `SmoothObsScales()` | `ang_vel=0.25`, `dof_vel=0.05` |
| `action_scale`, `action_clip`, `action_beta` | `0.25`, `None`, `0.8` | |

The proprio block (`10 + 3 * num_actions` values):

| # | Term | Width |
|---|---|---|
| 1 | `sin(2π·phase)`, `cos(2π·phase)` | 2 |
| 2 | `value` (unscaled) | 3 |
| 3 | `base_ang_vel * ang_vel` | 3 |
| 4 | base roll, pitch | 2 |
| 5 | `dof_pos - default_pos` | n |
| 6 | `dof_vel * dof_vel` | n |
| 7 | last action | n |

Observation = `proprio`, `zeros(n_priv_latent)`, then `history_length` past
proprio blocks, oldest first. The history holds earlier ticks only; it is
zero-filled at construction and on `reset()`.

### `amo_loco`

`AmoLocoController` (`hiwbc.controllers.amo`), after RoboJuDo's `AMOPolicy`
([AMO](https://github.com/OpenTeleVision/AMO)). The policy observes all body
joints but drives only the leading `num_actions` (G1: 23 observed, 15 driven).
An adapter MLP maps the height / torso command and the arm angles to part of
the observation, and the model takes a second input.

| Argument | Default | Description |
|---|---|---|
| `num_actions`, `default_pos` | required | Driven joints and their default pose |
| `obs_default_pos` | required | Default pose of all observed joints. Its first `num_actions` entries must equal `default_pos` |
| `forward_fn` | required | `(obs, extra_history) -> action` |
| `adapter_fn` | required | `(4 + n_upper,) -> (n_adapter_out,)`, normalized in and out |
| `adapter_input_mean`, `adapter_input_std`, `adapter_output_mean`, `adapter_output_std` | required | Adapter normalization stats |
| `control_hz` | `50` | |
| `ang_vel_scale`, `dof_vel_scale` | `0.25`, `0.05` | |
| `action_scale` | `0.25` | No EMA, no clip after scaling |
| `raw_action_clip` | `40.0` | Clip on the raw model output |
| `default_command` | `(0, 0, 0, 0.75, 0, 0, 0)` | Last four entries used with a 3-value command |
| `gait_freq` | `1.3` | Gait clock frequency, Hz |
| `n_priv` | `3` | Zero-filled privileged slot |
| `history_length`, `extra_history_length` | `10`, `25` | Proprio frames in the observation / second input |

`load_amo_runners(policy_file, adapter_file, norm_file, *, device="cpu")`
returns `forward_fn`, `adapter_fn` and the four stats as keyword arguments
(needs the `torch` extra).

`value` has 7 entries, `[c0, c1, c2, height, torso_yaw, torso_pitch,
torso_roll]`, or 3 (the rest from `default_command`). `c1` drives the yaw term
(`dyaw = -c1`); `c0` and `c2` enter the demo term.

The proprio frame (`proprio_size` values): `base_ang_vel * ang_vel_scale` (3),
roll and pitch (2), `sin`, `cos` of `dyaw` (2), `dof_pos - obs_default_pos` (n),
`dof_vel * dof_vel_scale` (n), last action extended to all observed joints (n;
the undriven entries are `offset / action_scale`), gait `sin` (2), adapter
output.

`observe` returns `(obs, extra_history)`:

- `obs`: proprio, demo (`n_upper + 9`: arm angles, `c0`, `c2`, 0, torso
  yaw / pitch / roll, `0.75 + height` three times), `zeros(n_priv)`, then the
  previous `history_length` proprio frames, oldest first.
- `extra_history`: the last `extra_history_length` proprio frames, oldest
  first, current included.

Both buffers are zero-filled at construction and on `reset()`. The gait clock
starts at `[0.25, 0.25]`; when all of `c0..c2` are below 0.1 it locks both feet
in phase, otherwise it moves them to antiphase.

## Motion-tracking controllers

All accept `ReferenceKind.MOTION_FRAMES`. `action_beta` defaults to `1.0`
(no EMA), so the "last action" term is the raw model output of the previous
tick; the H2H student uses `0.8`. `action_scale` (scalar or per joint) defaults
to `1.0` and `action_clip` to `None`.

The trackers read reference terms that are already computed for the current
tick. `twist_tracker`, `supermimic`, `kungfubot_tracker` and
`protomotions_tracker` (and `sonic`) can compute them from a motion clip with
`clip_reference` ([Motion clips](#motion-clips); KungfuBot and H2H need the FK
their profiles set up), and `asap_tracker` and `beyondmimic_tracker` play their
own motion as a [behavior](#behaviors). Otherwise playback and heading
alignment are the caller's. Each `frames` value is flattened
before use. A missing key raises `ValueError` naming it.

The four trackers (`hiwbc.controllers.twist`, `beyondmimic`, `protomotions`, `kungfubot`) share these arguments:

| Argument | Default | Description |
|---|---|---|
| `num_actions`, `default_pos`, `forward_fn` | required | |
| `history_length` | `0` | Number of past blocks kept. Zero-filled at construction and on `reset()` |
| `action_scale`, `action_clip`, `action_beta` | `1.0`, `None`, `1.0` | |

### `asap_tracker`

`AsapTrackerController`, after RoboJuDo's `AsapPolicy`
([ASAP](https://github.com/LeCAR-Lab/ASAP)). The motion is baked into the
checkpoint, so `frames` is not read; the controller advances its own phase.

| Argument | Default | Description |
|---|---|---|
| `num_actions`, `default_pos`, `forward_fn` | required | |
| `motion_length_s` | required | Length of the baked-in motion, s |
| `control_hz` | `50` | |
| `use_history` | `False` | Insert the history term |
| `history_length` | `4` | Number of past blocks, newest first |
| `obs_scales` | `AsapObsScales()` | All `1.0`: `actions`, `base_ang_vel`, `dof_pos`, `dof_vel`, `projected_gravity`, `ref_motion_phase`, `history` |
| `action_scale`, `action_clip` | `1.0`, `None` | |

Observation: last action (n), `base_ang_vel` (3), `dof_pos - default_pos` (n),
`dof_vel` (n), history (`history_length * (3n + 7)`, only with `use_history`),
projected gravity (3), phase (1). The history holds earlier ticks with the
terms last action, ang vel, dof pos, dof vel, gravity, phase, laid out
term-major, newest first: all `history_length` last actions, then all ang vels,
and so on. It is zero-filled at construction and on `reset()`.

`phase = timestep / (control_hz * motion_length_s)`. After each `step`, the
timestep advances by `reference.extra.get("play_speed", 1.0)`. Once the phase
reaches `1.0` it stops advancing and `motion_done` becomes `True`. `reset()`
restarts the motion.

### `twist_tracker`

`TwistTrackerController`, after RoboJuDo's `TwistPolicy`
([TWIST](https://github.com/YanjieZe/TWIST)).

Extra arguments: `ang_vel_scale`, `dof_pos_scale`, `dof_vel_scale` (all `1.0`),
`zero_dof_vel_mask` (`None`; a `(n,)` array, `1` where the joint velocity is
zeroed in the observation).

| `frames` key | Content |
|---|---|
| `mimic_target` | Upper-body mimic target |

Observation = `mimic_target`, proprio, then `history_length` past
`[mimic_target, proprio]` blocks, oldest first. Proprio is
`base_ang_vel * ang_vel_scale` (3), roll and pitch (2),
`(dof_pos - default_pos) * dof_pos_scale` (n), `dof_vel * dof_vel_scale` (n),
last action (n). TWIST trains with ankle velocities zeroed; pass
`zero_dof_vel_mask` with `1` at the ankle joints (G1 23-DoF: indices 4, 5, 10, 11).

### `beyondmimic_tracker`

`BeyondMimicTrackerController`, after RoboJuDo's `BeyondMimicPolicy`
([BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking)).

Extra arguments:

| Argument | Default | Description |
|---|---|---|
| `without_state_estimator` | `False` | Set for checkpoints trained without the anchor position and base linear velocity terms (`*_wose`) |
| `motion_fn` | `None` | `motion_fn(obs, time_step) -> dict` with the model's `actions`, `joint_pos`, `joint_vel` and `body_quat_w` (`[w, x, y, z]`) outputs. Enables the behavior |
| `motion_name` | `None` | Behavior name. Required with `motion_fn` |
| `motion_frames` | `None` | Frames in the baked-in motion. Required with `motion_fn` |
| `anchor_body_index` | `None` | Index of the anchor body (`torso_link`) in `body_quat_w`. Required with `motion_fn` |

| `frames` key | Content | Needed |
|---|---|---|
| `command` | Reference joint command | always |
| `motion_anchor_pos_b` | Reference anchor position in the robot anchor frame | unless `without_state_estimator` |
| `motion_anchor_ori_b` | Reference anchor orientation in the robot anchor frame | always |
| `base_lin_vel` | Base linear velocity (from a state estimator) | unless `without_state_estimator` |

Observation: `command`, [`motion_anchor_pos_b`], `motion_anchor_ori_b`,
[`base_lin_vel`], `base_ang_vel` (3), `dof_pos - default_pos` (n), `dof_vel` (n),
last action (n). Bracketed terms are dropped with `without_state_estimator`.
No history, no observation scales. `obs_dim` is the observation length (154
without a state estimator, 160 with, for 29 joints).

**Behavior.** Each released checkpoint is trained on one motion, and its ONNX
file outputs that motion's reference for a given `time_step`. With `motion_fn`
the controller plays it the way RoboJuDo's `use_motion_from_model` mode does:

1. `start_behavior` resets the controller and runs one forward on a zero
   observation at frame 0. Its `actions` become the initial last action and its
   reference becomes the first frame. `BehaviorSpec.start_pos` is that frame's
   `joint_pos`.
2. Each tick, the observation reads the reference from the previous forward and
   the forward runs at the current frame (`reference.extra["time_step"]`), so
   ticks 0 and 1 both read frame 0.
3. The reference anchor heading is turned, yaw only, onto the robot's base
   heading at the first tick.
4. `motion_anchor_ori_b` is the reference anchor orientation relative to the
   robot's torso. The torso orientation is `state.extra["torso_quat"]` if given,
   else `base_quat` turned by the `waist_yaw`, `waist_roll` and `waist_pitch`
   joint angles (the G1 pelvis-to-torso chain).
5. With a state estimator, `motion_anchor_pos_b` is zero (the robot anchor
   position is taken equal to the reference's, RoboJuDo's
   `override_robot_anchor_pos`) and `base_lin_vel` comes from
   `state.base_lin_vel`; `next` raises `ValueError` if it is `None`.

The source is `done` after `motion_frames` ticks; stepped further, it repeats
the last frame. References that do not carry `time_step` go through
`forward_fn` as before.

### `protomotions_tracker`

`ProtoMotionsTrackerController`, after RoboJuDo's `ProtoMotionsTrackerPolicy`
([ProtoMotions](https://github.com/NVlabs/ProtoMotions)).

Extra arguments: `dof_vel_scale`, `ang_vel_scale` (`1.0`), `use_dof_pos_offset`
(`False`: feed absolute `dof_pos`; `True`: feed `dof_pos - default_pos`).

| `frames` key | Content |
|---|---|
| `task_obs` | The imitation observation, computed upstream (needs body FK) |

Observation: `dof_pos` or its offset (n), `dof_vel * dof_vel_scale` (n),
`base_ang_vel * ang_vel_scale` (3), projected gravity (3), `task_obs`, last
action (n), then the history, newest first. Each history block is the unscaled
absolute `dof_pos`, `dof_vel`, `base_ang_vel`, gravity, last action.

### `kungfubot_tracker`

`KungfuBotTrackerController`, after RoboJuDo's `KungfuBotGeneralPolicy`
([KungfuBot / PBHC](https://github.com/TeleHuman/PBHC)). The model takes three
inputs, so `observe` returns a tuple and `forward_fn` receives it.

Extra arguments: `ang_vel_scale`, `dof_pos_scale`, `dof_vel_scale`,
`roll_pitch_scale`, `actions_scale`, `history_scale` (all `1.0`).

| `frames` key | Content |
|---|---|
| `anchor_ref_rot` | Reference anchor rotation |
| `next_step_ref_motion` | Next-step reference motion |
| `future_motion_base_lin_vel`, `future_motion_base_yaw_vel`, `future_motion_dof_pos`, `future_motion_roll_pitch`, `future_motion_root_height` | Future-motion targets |

`observe` returns `(actor_obs, future_motion_targets, history)`:

- `actor_obs`: last action, `anchor_ref_rot`, ang vel, `dof_pos - default_pos`,
  dof vel, history, `next_step_ref_motion`, roll and pitch. This is the policy's
  name-sorted order; each proprio term is multiplied by its scale.
- `future_motion_targets`: the five `future_motion_*` terms in the order listed
  in the table.
- `history`: `history_length` past entries of last action, ang vel, absolute
  `dof_pos`, dof vel, roll and pitch (each scaled), laid out term-major, newest
  first, then multiplied by `history_scale`. Zero-filled after `reset()`.

## SuperMimic

`SuperMimicController` (`hiwbc.controllers.supermimic`), registered as
`supermimic`. An unofficial reproduction for the 29-DoF Unitree G1: it follows
the inference code of
[supermimic_deploy](https://github.com/KyrieYoung926/supermimic_deploy), whose
ONNX policy was trained with SuperMimic-Reproduced. Other reproductions use
different observation layouts and are not covered.

```python
from hiwbc.controllers.supermimic import SuperMimicController

ctrl = SuperMimicController.from_bundle("supermimic_deploy/assets")
# or: make_controller("supermimic", assets_dir="supermimic_deploy/assets")
```

`from_bundle(assets_dir, *, check_hash=True, providers=None, **kwargs)` reads
the bundle's `config.json` and `policy.onnx`. It raises `ValueError` if the
joint order differs from `SUPERMIMIC_JOINT_NAMES`, the history or future-frame
count is not 10, or the ONNX sha256 differs from the config. It needs the
`onnx` extra.

| Argument | Default | Description |
|---|---|---|
| `forward_fn` | required | `((930,), (680,)) -> (29,)`; `load_onnx_runner(path)` builds one |
| `default_pos` | required | `q0`, SDK joint order |
| `action_scale` | required | Per joint |
| `kp`, `kd` | `None` | Returned with every target |
| `effort` | `None` | Per-joint torque limit |
| `bound_torque` | `True` | Re-express the target so `kp * (target - q) - kd * dq` stays within `effort`. Needs `kp`, `kd`, `effort` |

`accepts = (MOTION_FRAMES,)`. `frames["goal"]` is `(T, 68)`, frame 0 being the
current tick at 50 Hz; frames `0..9` are used and the last one is repeated when
`T < 10`. `goal_features(joint_pos, joint_vel, root_quat, root_lin_vel_w,
root_ang_vel_w, root_z, default_pos)` computes the 68-d goals from a 50 Hz clip:
`q - q0` (29), `dq` (29), projected gravity (3), root angular velocity in the
root frame (3), root linear velocity in the heading frame (3), root height (1).

The model takes two inputs:

- `obs_history` (930): 10 ticks of `[q - q0, dq, base ang vel, projected gravity,
  last action]` (93 each), term-major, oldest first, current tick included. On
  the first tick after `reset()` all 10 slots hold the current entry.
- `goal` (680): the 10 goal frames, frame-major.

The action maps to `q0 + action_scale * action`, with no EMA and no clip.

## BlinkMimic

`BlinkMimicController` (`hiwbc.controllers.blinkmimic`), registered as
`blinkmimic`. It runs policies trained with
[BlinkMimic](https://github.com/fuyichen0725/blinkmimic) (PyTorch) on the
29-DoF Unitree G1 in SDK joint order at 50 Hz. BlinkMimic follows SuperMimic's
method, but its observation and goal differ from [`supermimic`](#supermimic)'s.
XMimic policies share this interface and run on the same controller
([XMimic](#xmimic)).

```python
import hiwbc
from hiwbc.controllers.blinkmimic import BlinkMimicController, load_pt_runner

ctrl = hiwbc.load("blinkmimic/unitree_g1", checkpoint="model_2700.pt")
# or by hand:
from hiwbc.controllers.blinkmimic import G1_ACTION_SCALE, G1_DEFAULT_POS, G1_EFFORT, G1_KD, G1_KP

fwd = load_pt_runner("model_2700.pt", default_pos=G1_DEFAULT_POS, action_scale=G1_ACTION_SCALE)
ctrl = BlinkMimicController(forward_fn=fwd, default_pos=G1_DEFAULT_POS,
                            action_scale=G1_ACTION_SCALE, kp=G1_KP, kd=G1_KD, effort=G1_EFFORT)
```

| Argument | Default | Description |
|---|---|---|
| `forward_fn` | `None` | `forward_fn((obs_history (10, 99), goal (730,))) -> (29,)`; `load_pt_runner` (or, for XMimic, `load_npz_runner`) builds one. `None` for a single-motion export, which then only plays its behavior. Give `forward_fn`, `motion_fn` or both |
| `default_pos` | required | `q0`, SDK joint order |
| `action_scale` | required | Per joint, `0.25 * effort / kp` |
| `kp`, `kd` | `None` | Returned with every target |
| `effort` | `None` | Per-joint torque limit |
| `bound_torque` | `True` | Re-express the target so `kp * (target - q) - kd * dq` stays within `effort`, as the training actuator clips the torque. Needs `kp`, `kd`, `effort` |
| `control_dt` | `0.02` | Control period in seconds |
| `motion_fn` | `None` | A single-motion export, `motion_fn(obs_history (10, 99), time_step) -> dict` with `actions`, `joint_pos` and `next_anchor_quat_w` (`[w, x, y, z]`); `load_onnx_motion` builds one. Enables the behavior `motion_name` |
| `motion_name` | `None` | Behavior name of the baked motion. Needed with `motion_fn` |
| `motion_frames` | `None` | Frames in the baked motion (50 Hz), at least 2. Needed with `motion_fn` |
| `anchor_quat_init` | `None` | The motion's root orientation at frame 0, `[w, x, y, z]`. Needed with `motion_fn` |

Runners (both stateless `forward_fn((obs_history, goal)) -> action`, so
`runtime="remote"` works):

| Function | Reads | Description |
|---|---|---|
| `load_npz_runner(path, *, default_pos, action_scale, precision="bf16")` | XMimic `ckpt_*.npz` | See [XMimic](#xmimic) |
| `load_pt_runner(path, *, default_pos=None, action_scale=None)` | BlinkMimic `.pt` | Reads the file with torch (`torch` extra), then runs the network in numpy, fp32. The residual uses the checkpoint's own `q0` and scale; `default_pos` / `action_scale`, if given, must agree with them |

`goal_features(root_pos, root_quat, joint_pos, *, frames=10, dt=0.02)` builds
the `(10, 73)` goal from consecutive 50 Hz reference rows `f, f+1, ..., f+11`
(fewer near the clip end, at least 2), as the trainers' `Motion.goal` does.

The network normalises both inputs with the checkpoint's running mean and
variance, encodes the goal, decodes it with the history, and adds the next
frame's reference joint angles, `(q_ref - q0) / action_scale` (the residual
stored in the checkpoint). The target is `q0 + action_scale * action`, with no
EMA and no clip.

`accepts = (MOTION_FRAMES,)`. `frames`:

- `goal` `(10, 73)`: reference frames `f+1 .. f+10` at 50 Hz, the last one
  repeated at the clip end. Per frame: root displacement (3) and rotation (6D,
  6) in the heading frame of the current reference frame `f`; root linear and
  angular velocity in the future frame's own root frame (3 + 3); joint angles
  (29); joint velocities × 0.1 (29). Velocities are central differences.
- `ref_root_quat` `(4,)`: the root orientation of reference frame `f`, world
  frame, `[x, y, z, w]`.

The model takes two inputs:

- `obs_history` (990): 10 ticks of `[q - q0, dq, base ang vel (body frame),
  projected gravity, previous action, 6D(R^T R_ref)]` (99 each), frame-major,
  oldest first, current tick included. The 6D term is the first two columns of
  the rotation, flattened by row. On the first tick after `reset()` all 10 slots
  hold the current entry.
- `goal` (730): the 10 goal frames, frame-major.

Because `R_ref` is a world orientation, the clip must be turned to the robot's
heading at the start; `ClipPlayer` does this.

The `onnx` variant of the profile loads a BlinkMimic `tools/export_onnx.py`
export with one motion baked in and plays it as a [behavior](#behaviors). It
needs the `onnx` extra. `load_onnx_motion(path, providers=None)` returns
`(motion_fn, metadata)`; the export takes `(obs_history, time_step)` and holds
the goal itself, so it cannot track other clips (`check_clip` raises).
`check_onnx_metadata(metadata, source)` (`hiwbc.configs.blinkmimic.unitree_g1`)
raises `ValueError` when the export's joint names, gains, default pose or action
scale differ from the `G1_*` values (relative tolerance 1e-5), or its history
length, goal frames, control period, observation names or anchor body (the
pelvis) differ from this controller's; the profile calls it on load.

The behavior plays the motion as upstream `tools/play_onnx.py` does:
`time_step` runs `0, 1, ..., frames - 2`, one tick per 50 Hz frame, so the
behavior lasts `(frames - 1) × 0.02` s. The 6D term uses the export's
`anchor_quat_w_init` at tick 0 and the previous tick's `next_anchor_quat_w`
after that. The previous action starts at zero. At the first tick the motion is
turned, yaw only, onto the robot base's heading. `spec.start_pos` is the
motion's first pose; `WbcRobot` ramps to it before starting. A tick whose
forward fails does not advance the motion.

## XMimic

[XMimic](https://github.com/nico-bohlinger/xmimic) (JAX; Bohlinger and Peters,
[arXiv:2611.20452](https://arxiv.org/abs/2611.20452)) is a rewrite of
BlinkMimic with the same policy interface. Its policies run on
[`BlinkMimicController`](#blinkmimic) through the profile's `xmimic` variant;
the arguments, frames, layout and gains above apply unchanged.

```python
import hiwbc
from hiwbc.controllers.blinkmimic import G1_ACTION_SCALE, G1_DEFAULT_POS, load_npz_runner, self_check

ctrl = hiwbc.load("blinkmimic/unitree_g1", variant="xmimic", checkpoint="ckpt_5800.npz")
# or the runner alone:
fwd = load_npz_runner("ckpt_5800.npz", default_pos=G1_DEFAULT_POS, action_scale=G1_ACTION_SCALE)
assert self_check(fwd) < 1e-4
```

`load_npz_runner(path, *, default_pos, action_scale, precision="bf16")` reads an
XMimic `ckpt_*.npz` with numpy only and returns a stateless
`forward_fn((obs_history, goal)) -> action`. `precision="bf16"` rounds the
matmul inputs to bf16 as XMimic's evaluation does; `"fp32"` does not. The
profile option `precision` passes through.

`self_check(runner, test_vectors=None) -> float` runs a runner on the test
vectors an XMimic `.npz` holds (or on `(obs_history, goal, action)` you pass)
and returns the largest action error relative to `max(1, |action|)`. A
`precision="bf16"` runner matches XMimic's vectors to about 1e-7; an `"fp32"`
runner differs by up to about 1e-2.

## SONIC

`SonicController` (`hiwbc.controllers.sonic`), registered as `sonic`. It drives
the 29-DoF Unitree G1 with NVIDIA's
[GEAR-SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl), which is two
models:

- the **encoder** maps a reference to a 64-d motion token;
- the **decoder** maps the token plus 10 ticks of proprioception history to 29
  joint actions.

The observation layout follows NVIDIA's C++ deployment
(`gear_sonic_deploy`) and the `observation_config.yaml` of each released
checkpoint. hiWBC ships no weights.

Checked in MuJoCo against a transliteration of the C++ deployment with the
released `default`, `low_latency` and `sonic_v1_1` checkpoints: encoder input,
token, decoder input and PD target agreed within 1e-5. See the
[SONIC guide](guides/sonic.md) for a full setup.

### Construction

The profile builds it from the checkpoint folder:

```python
ctrl = hiwbc.load("sonic/unitree_g1", variant="default")   # checkpoint_dir: ~/.cache/hiwbc/sonic
```

`encoder=False` skips the encoder (`LATENT` only); `encoder_mode` sets the
starting mode. Or build it directly:

```python
from hiwbc.configs.sonic.unitree_g1 import SONIC_G1, SONIC_VARIANTS
from hiwbc.controllers.sonic import SonicController

ctrl = SonicController.from_onnx(
    "model_decoder.onnx",
    "model_encoder.onnx",     # optional; needed only for MOTION_FRAMES
    robot=SONIC_G1,
    obs=SONIC_VARIANTS["default"],
    encoder_mode="joint",
)
```

`from_onnx(decoder_path, encoder_path=None, *, providers=None, **kwargs)` wraps
each file with `load_onnx_runner` and passes `kwargs` to the constructor. It
needs the `onnx` extra. `load_onnx_runner(path, *, providers=None)` returns a
`forward_fn` that feeds a `(1, dim)` float32 tensor to the model's only input;
`providers` defaults to `["CPUExecutionProvider"]`.

| Argument | Default | Description |
|---|---|---|
| `decoder_fn` | required | `(decoder_dim,) -> (n,)` |
| `encoder_fn` | `None` | `(encoder_dim,) -> (64,)`. Required for `MOTION_FRAMES` |
| `robot` | required | A `SonicRobotConfig`; for the 29-DoF G1, `SONIC_G1` |
| `obs` | required | A `SonicObsConfig`; the released ones are `SONIC_VARIANTS[...]` |
| `encoder_mode` | `"joint"` | `joint`, `teleop` or `smpl`; can be overridden per tick |
| `stand_root_z` | `0.793` | Root height used by `stand_reference`, m |
| `planner` | `None` | A `SonicPlanner`. With one, the controller also accepts `VELOCITY` |
| `cartesian_targets` | `None` | End-effector name -> VR 3-point name (`left_wrist`, `right_wrist`, `head`); the profile sets the two wrists when the planner is on |

`motion_spaces` is `("joint", "smpl")` when an encoder is loaded, `()` without
one. `clip_reference` sets `encoder_mode` to the clip's space and passes the
window the encoder reads ([Motion clips](#motion-clips)). `stand_reference`
holds the default pose at `stand_root_z`, facing the robot's heading, in
`joint` mode with `play=False`.

`SonicRobotConfig(joint_names, default_pos, action_scale, kp, kd,
lower_body_index, wrist_index=())` holds everything robot-specific, in the
controller's joint order. `kp` / `kd` are returned with every target.
`lower_body_index` and `wrist_index` select the joints the teleop and SMPL
encoder terms read; a robot without wrists leaves `wrist_index` empty, and a
checkpoint that uses wrist terms then raises `ValueError` at construction. Joint
terms in the observation are sized from the config, so nothing in the controller
assumes 29 joints.

No EMA and no clip are applied. For `SONIC_G1` the joint order is IsaacLab
order, `G1_JOINT_NAMES`; this differs from the G1's hardware (MuJoCo) order in
`G1_JOINT_NAMES_MUJOCO`, so remap joints by name upstream. `joint` is the
deployment's `g1` encoder mode (id 0), renamed because it means "robot
joint-space motion", not a specific robot.

### Variants

| Variant | Checkpoint | Encoder dim | Decoder dim |
|---|---|---|---|
| `default` | `policy/release/observation_config.yaml` | 1762 | 994 |
| `low_latency` | low-latency release | 1247 | 994 |
| `sonic_v1_1` | SONIC v1.1 | 1751 | 994 |

`ctrl.encoder_dim` and `ctrl.decoder_dim` give the sizes for the chosen variant.
Term names follow the deployment: `<term>_<N>frame_step<S>` means N frames, S
ticks apart. `default` samples future reference frames every 5th tick,
`low_latency` every tick; `sonic_v1_1` expresses anchor orientations in the
robot's heading (yaw-only) frame instead of the full base frame.

`SonicObsConfig(encoder_terms, encoder_modes, decoder_terms=..., token_dim=64)`
describes one checkpoint. `encoder_modes` maps a mode name to
`(mode_id, required_terms)`. To support a new checkpoint, build a
`SonicObsConfig` from its `observation_config.yaml`, term for term, and pass it
as `obs`.

### Reference

`accepts = (LATENT, MOTION_FRAMES)`, plus `VELOCITY` with a planner.

**`VELOCITY`** (planner only): `value` is `[vx, vy, wz]` in the body frame;
`[0, 0, 0]` stands. The planner turns it into future frames for the `joint`
encoder. `extra["ee_targets"]` (`{vr point name: [x, y, z, qx, qy, qz, qw]}`, root
frame) switches to `teleop` mode with those VR points; missing points take the
default pose. See the [SONIC guide](guides/sonic.md#7-walking-velocity-commands-and-the-planner).

**`LATENT`**: `value` is the 64-d token, for example a VLA's motion token. The
encoder is not run.

**`MOTION_FRAMES`**: `frames` holds a window of future reference frames at the
50 Hz control rate, frame 0 being the current tick. The controller samples frames
`0, S, 2S, ...` for each term and repeats the last frame when the window is
short; a single frame (no leading time axis) is also accepted. Motion playback,
aligning the clip to the robot's heading, and FK stay upstream.

| Mode | `frames` keys |
|---|---|
| `joint` | `joint_pos` (T, n), `joint_vel` (T, n), `root_quat` (T, 4) |
| `teleop` | `joint_pos`, `joint_vel` (lower-body joints are selected inside), `root_quat`, `vr_3point_pos` (3, 3), `vr_3point_quat` (3, 4) |
| `smpl` | `smpl_joints` (T, 24, 3) root-local, `root_quat`, `joint_pos` (wrist joints are selected inside) |

- Joint arrays are in `robot.joint_names` order (IsaacLab order for `SONIC_G1`).
- `root_quat` is the reference root orientation in the world frame, `[x, y, z, w]`,
  already aligned to the robot's heading. The encoder sees it as a 6-D rotation
  relative to the robot's base (or heading) frame.
- `vr_3point_*` are the left wrist, right wrist and head, in the reference root
  frame. Quaternions are `[x, y, z, w]`; the controller converts them to the
  `[w, x, y, z]` the encoder was trained on.

`reference.extra` options:

| Key | Default | Effect |
|---|---|---|
| `encoder_mode` | constructor value | Mode for this tick |
| `play` | `True` | `False` holds frame 0 for every sample and zeroes the velocities (the deployment's paused state) |

Terms the active mode does not use are zero-filled. A key the mode needs but
`frames` lacks raises `KeyError`. The deployment instead falls back to another
mode silently.

### Decoder input and output

The decoder input is the token followed by 10-tick histories, oldest first, of
base angular velocity (3), `dof_pos - default_pos` (29), `dof_vel` (29), last
action (29) and projected gravity (3): `64 + 10 * 93 = 994` values. One entry is
logged per `step` before inference, so its last action is the previous tick's
raw decoder output. Until the history fills it is padded with zeros, except
gravity, which is padded with `[0, 0, 1]` to match the deployment.

Each `step` returns `JointTarget(position, kp, kd)` with
`extra["motion_token"]` set to the token used. `ctrl.token` holds the same
array. `reset()` clears the history and the token.

### Locomotion planner

`hiwbc.controllers.sonic_planner.SonicPlanner(planner_fn, *, qpos_joint_index,
lower_body_index, default_height=0.78874, control_dt=0.02, look_ahead=2,
seed=1234, background=False)` ports the planner path of NVIDIA's C++
deployment (`localmotion_kplanner.hpp`). The `sonic/unitree_g1` profile builds
it; build one yourself only for another robot.

| Name | Description |
|---|---|
| `load_planner_runner(path, *, providers=None)` | `planner_fn` from `planner_sonic.onnx`; raises `ValueError` if the file is not the V2 planner |
| `start(joint_pos)` | First plan from the measured joints, idle. The controller calls it when a `VELOCITY` reference arrives after another kind |
| `set_command(mode, speed, movement_yaw, facing_yaw, height=-1.0)` | Request a plan (`LocomotionMode`, planner-frame yaw angles); applied at the next 10 Hz check |
| `velocity_command(planner, velocity, base_quat_wxyz, *, dt)` | `[vx, vy, wz]` -> `set_command`, as the gamepad does (see the guide) |
| `window(length, base_quat_wxyz) -> dict` | `joint_pos`, `joint_vel`, `root_quat` (`[x, y, z, w]`), `root_z` for the next `length` frames from the current one; the first call after `start` aligns the motion's heading to the robot |
| `advance(joint_pos)` | One tick after the policy ran: blend a finished plan, check whether to replan, play the next frame |
| `replans`, `command`, `frame` | Counters and the command in effect |

`background=True` (`planner_thread=True` in the profile) runs each replan on a
worker thread; the current motion keeps playing until the result is blended
in. A replan that raises a `TimeoutError` (remote inference) is retried at the
next check.

## echo

`EchoController` (`hiwbc.controllers.echo`), registered as `echo`. It accepts
every `ReferenceKind`, ignores the observation, and returns `default_pos`, plus
an optional fixed `residual`. It needs no model and is meant for wiring tests.

```python
make_controller("echo", num_actions=29, default_pos=pose, residual=None)
```

## Math helpers

`hiwbc.math_utils`, pure numpy. All quaternions are `[x, y, z, w]`.

| Function | Returns |
|---|---|
| `gravity_orientation(q)` | Projected gravity in the base frame (`unitree_loco`) |
| `quat_rotate_inverse(q, v)` | `v` rotated by the inverse of `q`; with `v = [0, 0, -1]`, projected gravity (trackers, SONIC) |
| `quat_to_euler(q)` | `[roll, pitch, yaw]`, rad |
| `matrix_from_quat(q)` | `(3, 3)` rotation matrix |
| `quat_mul(a, b)` | Hamilton product `a * b` |
| `quat_conjugate(q)` | Conjugate (the inverse of a unit quaternion) |
| `subtract_frame_transforms_rot(q01, q02)` | Relative rotation `q01⁻¹ * q02` |

## Writing a controller

1. Subclass `Controller`. Set `accepts` and pass `num_actions`, `default_pos`
   and the post-processing arguments to `super().__init__`. A policy trained
   without action smoothing uses `action_beta=1.0`.
2. Implement `observe`. Concatenate the terms in exactly the order the policy
   was trained with; return a tuple for a multi-input model. Read the last action
   from `self._last_action`.
3. Implement `forward` by calling an injected `forward_fn`. Do not load a
   checkpoint inside the controller.
4. If the controller keeps a clock or history, override `reset()` (call
   `super().reset()`) and advance the clock in `step` after `super().step(...)`.
5. Register a factory in `hiwbc/controllers/` and import its module in
   `hiwbc/controllers/__init__.py`:

   ```python
   @register_controller("my_tracker")
   def _build(**kwargs):
       return MyTrackerController(**kwargs)
   ```

6. Add a profile for each robot it runs on: `hiwbc/configs/<controller>/<robot>.py`
   calling `register_profile` with a `build(variant, options)` that loads the
   checkpoint and passes `joint_names`, `kp`, `kd` and the trained values.
7. Optionally, set `motion_spaces` and implement `check_clip` / `clip_reference`
   to take motion clips, or `behaviors` / `start_behavior` for a motion in the
   checkpoint.
8. Add a test that feeds known state values through a stub `forward_fn` and
   asserts the observation layout, term by term.

> Observation order is safety-critical. A policy fed reordered or rescaled terms
> still outputs actions, and they command the robot with garbage. Check a new
> controller in simulation before it reaches hardware.

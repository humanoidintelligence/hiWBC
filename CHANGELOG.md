# Changelog

User-facing changes in each hiWBC release, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Before 0.1.0, any
release may change the API incompatibly; such changes are listed under
"Removed" or "Changed" and marked **breaking**.

## [0.0.4] - 2026-11-28

Ready-made robot configs and controller descriptions for front ends such as
hiApp. Needs hiRobot 0.0.9 or later.

### Changed

- **breaking**: needs hiRobot 0.0.9 or later (was 0.0.7). From 0.0.9 a `wbc`
  layer on a part keeps ticking in simulation when an action leaves its part
  out, and hiRobot finds hiWBC's robot configs by name.

### Added

- `hiwbc.profile_info(profile, options=None)`: description, variant, driven
  joints, velocity limits and missing checkpoint files of a profile, without
  loading a checkpoint. `sonic/unitree_g1` and `unitree_loco/unitree_g1` fill
  every field (`register_profile(..., info=...)`); others give the description.
- Robot configs shipped in the package, each a free-standing MuJoCo G1 with a
  `wbc` layer: `g1_sonic` and `g1_unitree_loco`. hiWBC registers them under
  hiRobot's `hirobot.configs` entry point, so `make_robot("g1_sonic")` and
  `defaults: [robots/g1_sonic]` find them by name; `hiwbc.robot_configs()`
  returns name -> path.
- `WbcRobot.base_velocity_limits`: per-axis `(min, max)` of `base_velocity`
  the profile was tested at, for scaling a joystick (not enforced). Set from the
  new `Controller` argument `velocity_limits`; the `sonic/unitree_g1` (with
  planner) and `unitree_loco/unitree_g1` `default` profiles give
  `((-0.8, 1.0), (-0.5, 0.5), (-1.0, 1.0))`, each axis walked without falling
  on the free MuJoCo G1.
- A `BASE_VELOCITY`, `MOTION_LATENT` or `CARTESIAN` action with only
  `joint_names` + `joint_target` (no `base_velocity`, `latent` or `ee_targets`)
  moves the direct joints and leaves the reference and its timeout unchanged.

## [0.0.3] - 2026-11-27

The `blinkmimic` controller runs BlinkMimic and XMimic tracking policies on the
G1. Needs hiRobot 0.0.7 or later and Python 3.10 or later.

### Changed

- **breaking**: needs Python 3.10 or later (was 3.9).
- **breaking**: needs hiRobot 0.0.7 or later (was 0.0.5). hiRobot 0.0.7 moved
  the G1 model to `~/.cache/hirobot/assets/unitree_ros/...` (`hirobot assets
  fetch unitree_g1`); the default `mjcf` of the `kungfubot_tracker` and
  `protomotions_tracker` profiles now points there. Pass `mjcf=` to use a
  model elsewhere.

### Added

- **`blinkmimic` controller** (profile `blinkmimic/unitree_g1`): runs
  [BlinkMimic](https://github.com/fuyichen0725/blinkmimic) (PyTorch) policies
  on the 29-DoF Unitree G1. It follows SuperMimic's method but is not
  compatible with `supermimic`. Variants: `default` (BlinkMimic `.pt`, read
  with the `torch` extra) and `onnx` (BlinkMimic's single-motion ONNX export,
  played as a behavior named after the export's motion, needs the `onnx`
  extra; loading fails if its metadata differs from BlinkMimic's G1 values).
  Takes `goal` and `ref_root_quat` frames or a `joint` motion clip
  (`ClipPlayer`, `MOTION_CLIP`).
- **XMimic policies** through the `xmimic` variant of `blinkmimic/unitree_g1`:
  [XMimic](https://github.com/nico-bohlinger/xmimic) (JAX,
  [arXiv:2611.20452](https://arxiv.org/abs/2611.20452)) is a rewrite of
  BlinkMimic with the same policy interface. Reads its `ckpt_*.npz` with numpy
  only, bf16 matmuls by default (`precision`); `self_check` compares a runner
  with the test actions stored in the checkpoint.
- Neither BlinkMimic nor XMimic publishes weights: train one with the upstream
  repository and point `checkpoint=` at it. Policies trained on LAFAN1 are for
  non-commercial use only. Run in simulation only.

## [0.0.2] - 2026-11-14

Follows hiRobot 0.0.5's skill renames. Needs hiRobot 0.0.5 or later.

### Changed

- **breaking**: needs hiRobot 0.0.5 or later (was 0.0.3). hiRobot 0.0.5
  renamed its skill methods: `list_robot_skills()` / `run_robot_skill()` /
  `stop_robot_skill()` now list and run every skill (the wbc layer's `stand`
  and behaviors, then the vendor's), and the vendor robot's own skills are
  `list_onboard_skills()`. `WbcRobot` checks `has_onboard_skills` when it
  connects. The robot guide and the API reference use the new names.
- The README has a doc index next to the language switch, and the line under
  the hardware photos is gone.
- Author order: Zihan Zhuang before Yifei Zhang (`pyproject.toml`, BibTeX).

## [0.0.1] - 2026-11-12

First public release. Needs hiRobot 0.0.3 or later.

### Added

- **Controller contract**: `Controller` turns `ControllerState` (joint
  positions and velocities, base orientation and angular velocity) plus a
  `Reference` into a `JointTarget` (PD position target with per-joint gains).
  A controller is a pure step with no hardware I/O; the model forward is
  injected, so observations are tested with a stub and no weights. A
  `Reference` is a velocity command (`VELOCITY`), reference-motion frames
  (`MOTION_FRAMES`) or a motion latent (`LATENT`).
- **11 built-in controllers**, each reproducing the observation layout its
  policy was trained with:
  - velocity locomotion: `unitree_loco` (unitree_rl_gym / unitree_rl_lab),
    `smooth_loco` (Smooth Humanoid Locomotion), `amo_loco` (AMO);
  - motion tracking: `asap_tracker` (ASAP), `twist_tracker` (TWIST),
    `beyondmimic_tracker` (BeyondMimic), `protomotions_tracker`
    (ProtoMotions), `kungfubot_tracker` (KungfuBot / PBHC), `supermimic`
    (unofficial SuperMimic reproduction);
  - universal motion token: `sonic` (GEAR-SONIC: `default`, `low_latency`,
    `sonic_v1_1`), with NVIDIA's kinematic planner for velocity commands;
  - `echo`, a dependency-free stub for testing the contract.
- **Profiles** (`hiwbc.load("<controller>/<robot>", variant=...)`): the values
  each Unitree G1 checkpoint was trained with (joint order, gains, default
  pose, scales), kept apart from the controller code. Checkpoint variants are
  profile variants; behaviors baked into ASAP and BeyondMimic checkpoints are
  listed per profile.
- **Checkpoint loading**: `load_torchscript_runner` and `load_onnx_runner`
  behind the optional `torch` and `onnx` extras. hiWBC does not ship weights;
  the guides say where to download each one.
- **hiRobot layer** (`layers: [{type: wbc, profile: ...}]` in a robot config):
  the robot takes `BASE_VELOCITY`, `BASE_POSE`, `CARTESIAN`, `MOTION_LATENT`
  and `MOTION_CLIP` commands instead of joint targets. The layer runs the
  controller at the control rate, ramps to the default pose before handing
  over, stands when commands stop or time out, damps on any fault, and reports
  command progress in `RobotStatus.command`.
- **Motion clips**: `ClipPlayer` turns a hiRobot `MotionClip` into reference
  terms for `sonic`, `twist_tracker`, `supermimic`, `kungfubot_tracker` and
  `protomotions_tracker`.
- **Remote inference**: `python -m hiwbc.serve` runs the model forward on a
  workstation while the control loop, observation and safety checks stay on
  the robot; a forward that misses `remote_timeout_s` damps the robot. The
  service has no authentication; use it on a trusted network.
- **Guides** in English and Simplified Chinese: getting started, locomotion,
  motion tracking, SONIC, and running a controller as a hiRobot robot.
- This release has been run in simulation only. The controllers that follow
  RoboJuDo match it tick for tick in MuJoCo (we ran those policies on a real
  robot earlier through RoboJuDo); `sonic` matches NVIDIA's C++ deployment and
  `supermimic` matches supermimic_deploy.

[0.0.4]: https://github.com/humanoidintelligence/hiWBC/compare/v0.0.3...v0.0.4
[0.0.3]: https://github.com/humanoidintelligence/hiWBC/compare/v0.0.2...v0.0.3
[0.0.2]: https://github.com/humanoidintelligence/hiWBC/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/humanoidintelligence/hiWBC/releases/tag/v0.0.1

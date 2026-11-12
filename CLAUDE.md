# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

hiWBC is the low-level **whole-body control** layer of the hi embodied stack. A
controller maps `(ControllerState, Reference) -> JointTarget` — reference in,
balanced full-body joint targets out — behind one contract, so the layers above
never touch balance, locomotion, or the vendor control loop. It is where learned
controllers (e.g. NVIDIA SONIC, RL+IK, motion trackers) live, wrapped uniformly.

Sibling repos (expected under `../`). hiWBC depends on hiRobot (types, `DofMap`, the
`layers:` extension point) and is unaware of the layers above:

- **hiRobot** — the robot interface (the body) that drives a controller at control rate.
- **hiVLA** — the model serving layer (the motor cortex) that produces high-level commands.
- **hiAgent** — the agentic operating layer that drives the robot with VLMs. It talks
  to hiRobot, never to hiWBC directly.

hiWBC is a hiRobot plugin that registers a `wbc` layer (`plugin.py` → `hirobot.register_layer`, loaded through
the `hirobot.drivers` entry point), with a flat `controllers/`, profiles in
`configs/<controller>/<robot>.py`, and `references/` for motion clips. hiRobot holds no WBC
code: adding or removing a controller must not require changes in hiRobot.

## Commands

```bash
pip install -e .            # core: numpy
pip install -e '.[torch]'   # + torch   (run a controller policy forward in-process)
pip install -e '.[onnx]'    # + onnxruntime (SONIC / SuperMimic / ASAP ONNX checkpoints)
pip install -e '.[dev]'     # + pytest, ruff

python -m pytest                                  # full suite (no GPU/weights needed; needs hirobot)
python -m pytest tests/test_tracking.py           # one file
python -m pytest tests/test_asap.py::test_name    # one test
ruff check src tests                              # lint (line-length 100, target py39)
```

The `echo` controller needs no weights, torch, or GPU — use it to exercise the
contract. Every controller takes an injected `forward_fn(obs) -> action`, so a stub
can stand in for a checkpoint and tests verify the observation layout without weights.

## Architecture

```
Reference + ControllerState ──▶ Controller.step ──▶ JointTarget (+ gains)
                                      │
                                      └─ observe → forward (forward_fn) → post-process
```

Core pieces:

- **`types.py`** — `ControllerState` / `Reference` (+ `ReferenceKind`:
  `VELOCITY` / `MOTION_FRAMES` / `LATENT`) / `JointTarget`.
- **`Controller` (`controller.py`)** — ABC. `step` = `observe` → `forward` → shared
  post-processing (EMA `action_beta` → clip → scale → `+ default_pos`, the RoboJuDo
  `get_action` order). `accepts` restricts the `ReferenceKind`s a controller handles.
- **`registry.py`** — `register_controller` / `make_controller` / `available_controllers`,
  name-based (mirrors hiRobot's registry).
- **`math_utils.py`** — pure-numpy quaternion/gravity helpers (no scipy).
- **Profiles (`configs/<controller>/<robot>.py`, `hiwbc.load`)** — a controller with the
  values its checkpoint was trained with, checkpoint loaded; variants per checkpoint.
- **`references/clip.py`** — `ClipPlayer`: hiRobot `MotionClip` → per-tick reference
  (resample, heading alignment, windows). Behaviors (`start_behavior`) play a motion
  baked into a checkpoint (ASAP, BeyondMimic).
- **`robot.py`** — `WbcRobot`, the `wbc` layer: a hiRobot `Robot` around the robot
  underneath. Maps `Action` modes (`BASE_VELOCITY`, `BASE_POSE`, `CARTESIAN`,
  `MOTION_LATENT`, `MOTION_CLIP`) to references, runs one tick per control period (thread
  on hardware, stepped by `send_action` / `idle_step` in sim), start ramp, command
  timeout → `stand_reference`, `CARTESIAN` rate limit, `BASE_POSE` P control. Any
  exception in a tick damps and latches a fault until `reset()`.
- **`remote.py` / `serve.py`** — `runtime="remote"`: the controller is built locally and
  its `*_fn` callables (and SONIC's planner) are swapped for `RemoteFn`s calling
  `python -m hiwbc.serve`. Stateless server; JSON header + raw arrays, no pickle.
  `ForwardTimeout` → `WbcRobot` repeats the last target, damps after `max_missed_ticks`.
- **`controllers/`** — the built-ins, registered on import:
  - `echo` — dependency-free stub, returns the default pose.
  - `unitree_loco`, `smooth_loco` — **velocity-command family**.
  - `amo_loco` (`amo.py`) — velocity plus height / torso commands. Two models: an
    adapter MLP whose output is part of the observation (`adapter_fn`), and a policy
    with a second input of 25 proprio frames. Observes 23 G1 joints, drives 15.
    `load_amo_runners` builds both from the released TorchScript files.
  - `asap_tracker`, and `twist`/`beyondmimic`/`protomotions`/`kungfubot` trackers
    (`twist.py`, `beyondmimic.py`, `protomotions.py`, `kungfubot.py`, shared base in
    `_tracker.py`) — **reference-motion-tracking family**.
  - `sonic` (`sonic.py`, G1 values in `configs/sonic/unitree_g1.py`) — NVIDIA
    GEAR-SONIC, two models: encoder (reference → 64-d token) and decoder (token +
    10-tick proprio history → joint actions). Robot-agnostic: joint order, gains and
    lower-body / wrist indices come from a `SonicRobotConfig` (`SONIC_G1`). Accepts
    `LATENT` (token given, encoder skipped) or `MOTION_FRAMES` (`joint` / `teleop` /
    `smpl` encoder modes; `joint` is the deployment's `g1`). With `sonic_planner.py`
    (port of the deployment's `localmotion_kplanner`, `planner_sonic.onnx`) it also
    accepts `VELOCITY`, optionally with VR 3-point hand targets.
  - `supermimic` (`supermimic.py`) — unofficial SuperMimic reproduction, following
    `KyrieYoung926/supermimic_deploy`'s `controller.py`. Two inputs: a 930-d proprio
    history laid out **term-major** (all 10 `q - q0` frames, then all 10 `dq`, ...) and
    a 680-d goal (10 future 68-d frames from `goal_features`). Verified tick-for-tick
    against that repo in MuJoCo.

The locomotion and tracking controllers replicate RoboJuDo's observation layouts and
were checked tick for tick against RoboJuDo in its MuJoCo env (obs within ~1e-5). `kungfubot` and `amo_loco` are multi-input: `observe` may return a
tuple of arrays. History layouts differ per family (term-major vs frame-major, newest
vs oldest first) and are pinned by value in `tests/test_history_layout.py`.

`sonic` replicates the C++ deployment in `GR00T-WholeBodyControl/gear_sonic_deploy`
(`g1_deploy_onnx_ref.cpp` observation registry, `state_logger.cpp` history,
`CreatePolicyCommand` action mapping), not RoboJuDo. Its source of truth per checkpoint
is that checkpoint's `observation_config.yaml`; the presets in `configs/sonic/unitree_g1.py`
(`default` 1762/994, `low_latency` 1247/994, `sonic_v1_1` 1751/994 encoder/decoder dims)
were checked term for term against those YAMLs. G1 tables are kept in MuJoCo order as
in `policy_parameters.hpp` and permuted to IsaacLab order at import; the controller's
joint order is IsaacLab. Quaternions in hiWBC are `[x, y, z, w]`; the deployment uses
`[w, x, y, z]`, so the VR 3-point orientation term is converted on the way in.

**Invariants to preserve when changing any of this:**

- **Observation field order is safety-critical.** It must match the order each policy
  was trained with, or the robot is commanded with garbage. Each `observe` follows its
  RoboJuDo reference exactly; don't reorder terms. A wrong history order fits the input
  size but the policy falls within ~1 s, so tests pin layouts by value, not just shape.
- **The model forward is injected (`forward_fn`), decoupled from obs logic.** A backend
  supplies a torch-jit / ONNX runner; tests supply a stub. Don't hard-wire a checkpoint
  loader into a controller.
- **Post-processing lives in the base class.** Keep the EMA → clip → scale → offset order;
  a no-EMA policy is just `action_beta=1.0`. `action_scale` may be a scalar or per-joint
  (SONIC). Don't duplicate it per controller.
- **Motion playback and body-frame transforms stay upstream.** Trackers consume
  per-tick reference terms from `reference.frames`; FK / anchor transforms are the
  robot model's job (hiRobot), consistent with "reference in, joint targets out".
- **Heavy deps are optional and lazy.** The controllers import numpy only; `torch` and
  `onnx` (onnxruntime) are extras, imported inside the function that needs them. Keep
  backend deps out of the import path (`tests/test_runtime.py` checks `import hiwbc`).
- **`WbcRobot` never sends a target it did not check.** Faults damp; joints the
  controller does not own are `direct_joints`; hiRobot's safety gate sits underneath.
  A controller sending its own gains needs `direct_kp` / `direct_kd` for the rest.

## Status

All 11 controllers have tests and have been run with their real checkpoints in MuJoCo:
the RoboJuDo family against RoboJuDo, `sonic` against a transliteration of NVIDIA's C++
deployment, `supermimic` against supermimic_deploy. Every
controller has a profile. `WbcRobot` runs them through hiRobot's `make_robot`; SONIC
walks, goes to a pose and tracks hand targets in MuJoCo through it, also with remote
inference. The RoboJuDo-deployed policies were tested on a real robot earlier (not
through this release); `sonic`, `supermimic` and `WbcRobot` have run in simulation
only. Known gaps: SONIC hand targets overshoot (about
1.3x, 5–12 cm error) and it walks slower than commanded (0.43 for 0.5 m/s).

Tests need hiRobot installed (an editable checkout of the sibling repo during development).

User docs: `README.md`, `CHANGELOG.md`, `docs/api.md`, `docs/guides/` (getting started, locomotion,
motion tracking, SONIC, `robot.md` for the `wbc` layer). Each has a Simplified Chinese
version (`*.zh-CN.md`) that mirrors it heading for heading; change both together. Keep
them in sync with the code when an argument, option or default changes. The only CLI is `python -m hiwbc.serve`.

## Release

Public releases are cleaned snapshots of `main`, committed one per version onto a local
orphan `release` branch and pushed to `humanoidintelligence/hiWBC`; `main` itself is
never pushed there. Published files (`README.md`, `CHANGELOG.md` (+ `.zh-CN`), `CONTRIBUTING.md`, `docs/`, `src/`, `tests/`,
this file) hold no local paths, environment names or references to internal notes.

## Conventions

License (Apache-2.0), `src/hiwbc` layout, and ruff line-length 100 mirror the sibling
hiRobot/hiVLA/hiAgent repos — match them. pytest is pinned to `tests/` with `norecursedirs`.
Tests inject stub forward functions, so the suite runs with no GPU or weights.

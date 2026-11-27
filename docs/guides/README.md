# hiWBC guides

English | [简体中文](./README.zh-CN.md)

Step-by-step guides for running each built-in controller, from loading the
checkpoint to the first control tick on a robot. The [API reference](../api.md)
lists every argument and observation term; these guides show how to wire them
up for a released policy, with the values each policy was trained with.

| Guide | Controllers | Commanded with |
|---|---|---|
| [Getting started](getting_started.md) | all | the control loop, joint order, forward functions, first power-on |
| [Velocity-command locomotion](locomotion.md) | `unitree_loco`, `smooth_loco`, `amo_loco` | `VELOCITY`: a joystick-style `[vx, vy, wyaw]` (AMO: plus height and torso) |
| [Reference-motion tracking](motion_tracking.md) | `asap_tracker`, `twist_tracker`, `beyondmimic_tracker`, `kungfubot_tracker`, `protomotions_tracker`, `supermimic`, `blinkmimic` (BlinkMimic, XMimic) | `MOTION_FRAMES`: reference terms for this tick, a motion clip, or the checkpoint's own motion (a behavior) |
| [SONIC](sonic.md) | `sonic` | `LATENT` (a motion token), `MOTION_FRAMES`, a motion clip, or `VELOCITY` through its locomotion planner |
| [Running as a hiRobot robot](robot.md) | all | hiRobot `Action`s through the `wbc` layer: velocity, walk-to-pose, hand poses, clips; faults, status, remote inference |

Each controller has a profile per robot (`hiwbc.load("<controller>/<robot>")`)
that loads the checkpoint with its trained joint order, gains and scales; the
guides show both that and the hand-built form.

The examples use the Unitree G1 and the checkpoints
[RoboJuDo](https://github.com/HansZ8/RoboJuDo) deploys (under
`RoboJuDo/assets/models/g1/`), and SONIC's released checkpoints from
[nvidia/GEAR-SONIC](https://huggingface.co/nvidia/GEAR-SONIC). Each parameter
table was checked by running hiWBC tick for tick against RoboJuDo (SONIC: a
transliteration of NVIDIA's C++ deployment) in MuJoCo on the same state: the
observations and PD targets agree to about 1e-5.

> ⚠️ **Safety.** A controller fed a wrong observation (wrong joint order, wrong
> scale, wrong history layout) still outputs actions, and they command the
> robot with garbage. Run every new setup in simulation first, then on a robot
> that is hung or supported, with someone at the e-stop.

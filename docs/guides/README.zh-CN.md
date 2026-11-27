# hiWBC 指南

[English](./README.md) | 简体中文

这些分步指南讲解如何运行每个内置控制器，从加载 checkpoint 到机器人上的第一个控制周期。
[API 参考](../api.zh-CN.md)列出了每个参数和每个观测项；这些指南则展示如何为一个已发布的 policy
把它们接起来，并给出该 policy 训练时用的取值。

| 指南 | 控制器 | 命令方式 |
|---|---|---|
| [快速开始](getting_started.zh-CN.md) | 全部 | 控制循环、关节顺序、前向函数、首次上电 |
| [速度命令行走](locomotion.zh-CN.md) | `unitree_loco`、`smooth_loco`、`amo_loco` | `VELOCITY`：摇杆式的 `[vx, vy, wyaw]`（AMO 还有高度和躯干姿态） |
| [参考动作跟踪](motion_tracking.zh-CN.md) | `asap_tracker`、`twist_tracker`、`beyondmimic_tracker`、`kungfubot_tracker`、`protomotions_tracker`、`supermimic`、`blinkmimic`（BlinkMimic、XMimic） | `MOTION_FRAMES`：本控制周期的参考项、一个动作片段，或 checkpoint 自带的动作（一个行为（behavior）） |
| [SONIC](sonic.zh-CN.md) | `sonic` | `LATENT`（一个动作 token）、`MOTION_FRAMES`、一个动作片段，或通过其行走 planner 的 `VELOCITY` |
| [作为 hiRobot 机器人运行](robot.zh-CN.md) | 全部 | 通过 `wbc` layer 发送 hiRobot `Action`：速度、走到目标位姿、手部位姿、动作片段；故障、状态、远程推理 |

每个控制器针对每台机器人都有一个 profile（`hiwbc.load("<controller>/<robot>")`），它按训练时的关节顺序、
增益和缩放系数加载 checkpoint；指南会同时给出这种方式和手动构建的方式。

示例使用 Unitree G1，以及 [RoboJuDo](https://github.com/HansZ8/RoboJuDo) 部署的 checkpoint
（位于 `RoboJuDo/assets/models/g1/`）和 [nvidia/GEAR-SONIC](https://huggingface.co/nvidia/GEAR-SONIC)
发布的 SONIC checkpoint。每张参数表都经过检查：在 MuJoCo 中从相同状态出发，逐个控制周期对比 hiWBC 与
RoboJuDo（SONIC 对比的是 NVIDIA C++ 部署代码的逐行移植），观测和 PD 目标的差异约为 1e-5。

> ⚠️ **安全。** 控制器收到错误的观测（关节顺序错、缩放错、历史布局错）时仍会输出动作，
> 而这些动作会用无意义的命令驱动机器人。每个新配置都先在仿真中运行，再在吊起或有支撑的机器人上运行，
> 并有人守在急停旁。

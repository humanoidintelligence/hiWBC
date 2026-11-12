# 更新日志

每个 hiWBC 版本用户看得到的变化，新的在上面。格式参照
[Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。0.1.0 之前任何版本都可能有不兼容的接口改动，
这类改动列在"移除"或"变更"下，并标 **不兼容**。

## [0.0.1] - 2026-11-12

首个公开版本。需要 hiRobot 0.0.3 或更高版本。

### 新增

- **控制器接口约定**：`Controller` 把 `ControllerState`（关节位置和速度、基座姿态和角速度）加上一个
  `Reference` 变成 `JointTarget`（PD 位置目标，带每个关节的增益）。控制器是一步纯计算，不碰硬件；
  模型前向是注入的，所以观测用桩函数测试，不需要权重。`Reference` 是速度命令（`VELOCITY`）、
  参考动作帧（`MOTION_FRAMES`）或动作隐变量（`LATENT`）。
- **11 个内置控制器**，每个都复现其 policy 训练时的观测布局：
  - 速度命令行走：`unitree_loco`（unitree_rl_gym / unitree_rl_lab）、`smooth_loco`（Smooth Humanoid
    Locomotion）、`amo_loco`（AMO）；
  - 动作跟踪：`asap_tracker`（ASAP）、`twist_tracker`（TWIST）、`beyondmimic_tracker`（BeyondMimic）、
    `protomotions_tracker`（ProtoMotions）、`kungfubot_tracker`（KungfuBot / PBHC）、`supermimic`
    （SuperMimic 的非官方复现）；
  - 通用 motion token：`sonic`（GEAR-SONIC：`default`、`low_latency`、`sonic_v1_1`），速度命令经
    NVIDIA 的运动学 planner；
  - `echo`：不依赖任何包的桩控制器，用来测试接口约定。
- **Profile**（`hiwbc.load("<controller>/<robot>", variant=...)`）：每个 Unitree G1 checkpoint 训练时
  用的值（关节顺序、增益、默认姿态、缩放），和控制器代码分开放。checkpoint 的不同版本是 profile 的
  variant；ASAP 和 BeyondMimic checkpoint 里自带的动作按 profile 列出。
- **加载 checkpoint**：`load_torchscript_runner` 和 `load_onnx_runner`，分别需要可选的 `torch` 和
  `onnx` extra。hiWBC 不分发权重，指南里写了每个权重去哪下载。
- **hiRobot 层**（机器人配置里写 `layers: [{type: wbc, profile: ...}]`）：机器人接受 `BASE_VELOCITY`、
  `BASE_POSE`、`CARTESIAN`、`MOTION_LATENT` 和 `MOTION_CLIP` 命令，不再接关节目标。这一层按控制频率
  运行控制器，交接前先过渡到默认姿态，命令停止或超时就站住，任何故障都进阻尼，并在
  `RobotStatus.command` 里报告命令进度。
- **动作片段**：`ClipPlayer` 把 hiRobot 的 `MotionClip` 变成 `sonic`、`twist_tracker`、`supermimic`、
  `kungfubot_tracker` 和 `protomotions_tracker` 要的参考项。
- **远程推理**：`python -m hiwbc.serve` 在工作站上跑模型前向，控制循环、观测和安全检查留在机器人上；
  前向超过 `remote_timeout_s` 没返回，机器人就进阻尼。该服务没有身份认证，只在可信网络上用。
- **指南**，中英两版：快速开始、行走、动作跟踪、SONIC，以及把控制器作为 hiRobot 机器人运行。
- 这个版本只在仿真里跑过。沿用 RoboJuDo 的控制器在 MuJoCo 里和 RoboJuDo 逐 tick 一致（这些 policy
  以前经 RoboJuDo 在真机上跑过）；`sonic` 和 NVIDIA 的 C++ 部署一致，`supermimic` 和 supermimic_deploy 一致。

[0.0.1]: https://github.com/humanoidintelligence/hiWBC/releases/tag/v0.0.1

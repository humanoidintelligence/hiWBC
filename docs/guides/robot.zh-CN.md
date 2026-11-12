# 把控制器作为 hiRobot 机器人运行

[English](./robot.md) | 简体中文

其他指南都在你自己的循环里调用 `Controller.step`。本指南介绍运行控制器的另一种方式：作为
[hiRobot](https://github.com/humanoidintelligence/hiRobot) 的一个 **layer**。安装 hiWBC 时会向
hiRobot 注册一个名为 `wbc` 的 layer。列出它的机器人配置会得到一个自己保持平衡的 `Robot`。这个 layer：

- 每个控制周期运行一次控制器，
- 接收高层命令（`BASE_VELOCITY`、`BASE_POSE`、`CARTESIAN`、`MOTION_LATENT`、`MOTION_CLIP`），
  而不是关节目标，
- 命令停止时站住，
- 一个 tick 中任何一步出错时让机器人进入阻尼。

hiRobot 本身不含 WBC 代码；它通过 `hirobot.drivers` entry point 找到这个 layer。请先阅读
[快速开始](getting_started.zh-CN.md)。

## 1. 带 `wbc` layer 的机器人配置

```yaml
# g1_sonic.yaml：MuJoCo G1 上的 SONIC
defaults: [robots/unitree_g1_mujoco_standing]   # 自由站在地面上的 G1

assembly:
  parts:
    body:
      overrides:
        model:
          # 从 SONIC 的站立姿态开始（G1 SDK 顺序）。SONIC 必须从第 0 个 tick 起接管。
          home: [-0.312, 0, 0, 0.669, -0.363, 0, -0.312, 0, 0, 0.669, -0.363, 0,
                 0, 0, 0, 0.2, 0.2, 0, 0.6, 0, 0, 0, 0.2, -0.2, 0, 0.6, 0, 0, 0]

sim:
  realtime: false          # 调用方发多快就步进多快
  ramp_seconds: 0.0

layers:
  - type: wbc
    profile: sonic/unitree_g1    # 任意 hiwbc profile
    start_seconds: 0.0           # 先插值到控制器的默认姿态（默认 3 s）
```

`wbc` 条目中除 `type` 以外的每个键都交给 `WbcRobot`，它不认识的键再作为 profile 选项交给
`hiwbc.load`（`variant`、`checkpoint_dir`、`device`、`providers` 等）。见
[Layer 选项](#6-layer-选项)。

```python
import numpy as np
from hirobot import Action, ControlMode, make_robot

robot = make_robot("g1_sonic.yaml")
robot.action_modes        # (BASE_VELOCITY, MOTION_LATENT, MOTION_CLIP, CARTESIAN, BASE_POSE)
robot.connect()

for _ in range(250):      # 50 Hz 下 5 s
    robot.send_action(Action(mode=ControlMode.BASE_VELOCITY,
                             base_velocity=np.array([0.4, 0.0, 0.0])))

goal = Action(mode=ControlMode.BASE_POSE, base_target=np.array([0.0, 0.0, 0.0]))
while not robot.get_status().command.done:
    robot.send_action(goal)   # 仿真中每次发送推进一个外层周期

robot.hold()              # 站住
robot.disconnect()
```

在 MuJoCo G1 中，这段代码让机器人向前走约 1.5 m，再回到距起点约 10 cm 以内。

## 2. 时间：仿真与真机

- **仿真**（内层机器人是仿真的）：只有调用 `send_action` 时时间才推进。每次调用运行一个外层周期
  （`outer_hz`，默认是一个控制周期）；`outer_hz: 10` 时每次调用运行五个 50 Hz 的 tick。要让
  `BASE_POSE` 或 `MOTION_CLIP` 运行到结束，就每步再发送同一个 `Action` 对象，直到
  `get_status().command.done` 为真。重发同一个 `BASE_POSE` 目标就是继续朝它走。片段只有在重发
  正在播放的同一个 `MotionClip` 对象时才接着播；新建的片段即使内容相同也从第一帧开始，`done` 之后
  再发也从头播。`done` 为真后就停止发送。
- **真机**：`connect()` 启动一个线程，以 `1 / control_dt` 运行控制器。`send_action` 只更新命令；
  你什么都不发时，这个线程也会继续保持平衡。

在 `connect()` 和 `reset()` 时，layer 在 `start_seconds` 内把控制器的关节从实测姿态插值到
`default_pos`，然后立即交给控制器。这段插值是纯 PD，没有平衡：

- **真机**：插值期间让机器人吊在龙门架上或有支撑。
- **自由浮动的仿真**（`unitree_g1_mujoco_standing`）：机器人会在插值期间摔倒（在 MuJoCo 中，默认的
  3 s 之后 G1 已经倒在地上，随后 SONIC 的第一个 tick 就进入了故障）。让仿真器从控制器的默认姿态
  启动（本体的 `model: {home: ...}`），并设 `start_seconds: 0`，如上面的配置所示。
  `unitree_g1_mujoco` 像龙门架一样焊住骨盆，所以在那里插值是安全的。

### 仿真 G1

hiRobot 的 MuJoCo G1 是真实 G1 的配置加上一层 MuJoCo 覆盖，所以关节名、关节顺序和相机都与真机一致，
同一个 `layers:` 条目在两者上都能运行。`unitree_g1_mujoco_standing` 让它自由站在地面上；
`unitree_g1_mujoco` 焊住骨盆。两者都需要 Unitree `unitree_ros` 中的 G1 MJCF，按 hiRobot 的
[设置指南](https://github.com/humanoidintelligence/hiRobot/blob/main/docs/setup/unitree.md)获取。

- `robot.get_observation().images["head"]` 是头部相机（1920x1080），每次调用时渲染（约 1 ms）。
  layer 不使用它；它是留给上层 policy 的。没有 GL 上下文时，机器人照常运行，只是没有图像。
- 仿真中 `make_robot` 直接返回 `WbcRobot` 本身（跳过真实的相机和音频驱动）。在真实 G1 上它返回
  hiRobot 的 `RobotAssembly`，相机和音频包在 layer 外面；`send_action`、`get_status` 等调用照样
  到达 layer。
- `hirobot sim <config> --viewer` 连接一台仿真机器人并保持它的起始姿态，可以快速检查模型和增益是否
  加载成功。它发送的是纯 PD 关节目标，所以不运行 `wbc` layer，自由站立的 G1 会摔倒。

## 3. 命令

`robot.action_modes` 列出已加载控制器支持的模式：

| 模式 | 字段 | 支持的控制器 | 效果 |
|---|---|---|---|
| `BASE_VELOCITY` | `base_velocity` `[vx, vy, wz]`，本体坐标系 | 速度控制器；`sonic`（通过它的 planner） | 以该速度行走 |
| `BASE_POSE` | `base_target` `[x, y, yaw]`，世界坐标系 | 所有支持 `BASE_VELOCITY` 的控制器，且机器人要报告 `base_position` | 走到目标后站住。进度见 `status.command`（[第 5 节](#5-状态)） |
| `CARTESIAN` | `ee_targets` `{name: [x, y, z, qx, qy, qz, qw]}`，基座坐标系，可选 `base_velocity` | `sonic`（`left_hand`、`right_hand`） | 用 SONIC 的 VR 三点遥操作编码器跟踪手部位姿，同时以 `base_velocity` 行走 |
| `MOTION_LATENT` | `latent` | `sonic`（64 维 token） | 本 tick 使用这个动作 token |
| `MOTION_CLIP` | `motion`（`MotionClip`）、`append` | 带 `motion_spaces` 的控制器 | 播放动作片段，然后站住。`append=True` 把它排在当前片段之后 |

适用于所有模式的规则：

- **超时。** 超过 `timeout_s`（0.5 s）的 `BASE_VELOCITY`、`CARTESIAN` 或 `MOTION_LATENT` 命令会被
  丢弃，机器人站住。调用方必须持续发送。`BASE_POSE` 和 `MOTION_CLIP` 运行到结束为止。
- **`hold()`** 原地站住。
- **控制器不驱动的关节**（手指，或只控制腿的 policy 的手臂）列在 `robot.direct_joints` 中。在同一个
  `Action` 里用 `joint_names` + `joint_target` 控制它们。
- **Layer skill。** `stand` 和控制器的行为（behavior，例如 BeyondMimic 的 `dance`）由
  `robot.list_layer_skills()` 列出，用 `robot.run_layer_skill(name)` 启动，用
  `robot.stop_layer_skill()` 结束。行为运行期间 layer 持有命令流，所以直接调用 `send_action` 会抛出
  `RobotBusyError`，直到行为结束。`robot.list_robot_skills()` 仍然是厂商机器人自带的 skill
  （仿真 G1 上没有）。

`CARTESIAN` 目标有速率限制：SONIC 跟踪的位姿以最多 `cartesian_max_speed`（0.5 m/s）和
`cartesian_max_rate`（1.5 rad/s）的速度向命令位姿移动。它从 SONIC 的默认 VR 位姿出发，命令结束时
回到该位姿。在 MuJoCo 中，一个 tick 内 15–20 cm 的跳变让 SONIC 输出了超出 ±3 rad 安全限位的关节
目标，所以不要把这些限制调高太多。在 MuJoCo 中，距静止位置 16–29 cm 的手部目标达到了 5–12 cm
以内的误差；SONIC 是学习得到的跟踪器，不是 IK。

## 4. 故障

一个 tick 中的异常（来自控制器、下面的安全门或驱动）会进入故障状态并对整个栈急停：hiRobot 先在最底层
停下机器人（Unitree 或 MuJoCo 本体进入阻尼，kp 0），然后 layer 停止自己的线程。之后：

- `send_action` 抛出 `RobotFaultError`，
- `robot.get_status().fault` 记录原因，
- `robot.reset()` 清除故障，并再次插值到 `default_pos`。

保持 hiRobot 的安全门开启。正是它让一个错误的关节目标变成一台进入阻尼的机器人，而不是一条被发出去的命令。

## 5. 状态

`robot.get_status().command` 是 hiRobot 通用的 `CommandProgress`，表示调用方最近一条高层命令的进度。判断机器人
是否到达、片段是否播完，读它，不要读 `extra["wbc"]`：

| 字段 | `BASE_POSE` | `MOTION_CLIP` | `BASE_VELOCITY`、`MOTION_LATENT`、`CARTESIAN` |
|---|---|---|---|
| `mode` | `BASE_POSE` | `MOTION_CLIP` | 该模式 |
| `done` | 到达目标后为 `True` | 片段播完后为 `True` | `None`（没有终点） |
| `goal` | `[x, y, yaw]`，坐标系同 `base_position` | `None` | `None` |
| `clip_time`、`clip_duration` | `None` | 已播放的秒数和总长，按整个 `append` 队列算 | `None` |

已结束的命令（已到达、已播完，或速度命令超时后站住）保留到下一条命令替换它。还没发过命令、`hold()` 之后、以及
layer skill 运行期间，`command` 为 `None`：技能占着命令流，它的进度看 `layer_skill_running`，不看 `command`。

`robot.get_status().extra["wbc"]` 是 layer 的诊断信息：

| 键 | 含义 |
|---|---|
| `profile`、`variant` | 已加载的控制器 |
| `mode` | `idle`、`ramp`、`stand`、`command`、`goto`、`clip` 或 `behavior` |
| `clip_time`、`clip_duration`、`clip_done` | 动作片段进度，或 `None` |
| `goal`、`arrived` | `BASE_POSE` 目标，以及是否已到达 |
| `ee_targets` | SONIC 当前正在跟踪的手部位姿（经过速率限制之后） |
| `ticks` | 已运行的控制周期数 |
| `missed_ticks` | 前向超时（远程推理）或运行迟到（实时）的 tick 数 |
| `late_ticks`、`step_ms_max` | 单步计时 |

## 6. Layer 选项

| 选项 | 默认值 | 说明 |
|---|---|---|
| `profile` | 必填 | `hiwbc.load` 的 profile，例如 `sonic/unitree_g1` |
| `variant` | profile 的默认值 | profile 变体 |
| `timeout_s` | 0.5 | 这么长时间没有收到命令就站住 |
| `outer_hz` | 控制频率 | 仿真：调用方发送的频率 |
| `start_seconds` | 3.0 | 启动时插值到 `default_pos` 的时长 |
| `direct_kp`、`direct_kd` | 驱动的增益 | 控制器不驱动的关节的增益（标量或每个关节一个值）。控制器发送自己的增益（SONIC）且有些关节不归它管时必填 |
| `forward_timeout_s`、`max_missed_ticks` | 0.015, 2 | 超过 `forward_timeout_s` 的一步算作错过；连续错过超过 `max_missed_ticks` 次就进入阻尼。墙钟迟到只在真机上计数；远程超时始终计数 |
| `stale_s` | 0.1 | 某个 tick 时本体的关节状态（`obs.stamps["body"]`）按机器人时钟比这更旧就进入故障。`None` 关闭此检查 |
| `goto` | 见下文 | `BASE_POSE` 的增益和限制 |
| `cartesian_max_speed`、`cartesian_max_rate` | 0.5, 1.5 | `CARTESIAN` 速率限制，单位 m/s 和 rad/s |
| `runtime`、`endpoint`、`remote_timeout_s` | `local` | [远程推理](#8-远程推理) |
| 其他键 | | 交给 `hiwbc.load` 的 profile 选项 |

`goto` 的键：`kp_xy` 1.0（m/s 每 m），`kp_yaw` 1.0（rad/s 每 rad），`max_speed` 0.6 m/s，
`max_yaw_rate` 0.6 rad/s，`pos_tol` 0.10 m，`yaw_tol` 0.15 rad。每个 tick 把误差换算成一条本体坐标系
下的速度命令；两个容差都满足时机器人站住。

## 7. SONIC 的行走 planner

SONIC 本身只跟踪参考动作。为了按速度命令行走，`sonic/unitree_g1` profile 像 C++ 部署代码一样加入了
NVIDIA 的运动学 planner（规划器，`planner_sonic.onnx`）：planner 把命令变成未来的下半身帧，SONIC
再跟踪这些帧。详见 [SONIC 指南](sonic.zh-CN.md#7-行走速度命令与-planner)。

## 8. 远程推理

50 Hz 循环、观测、安全检查和故障处理都留在机器人上；只有模型前向在工作站上运行。网络或工作站出故障时，
机器人会发现超时并让自己进入阻尼。

在工作站上：

```bash
python -m hiwbc.serve --profile sonic/unitree_g1 --host 0.0.0.0 --port 5555
#   [--variant low_latency] [--option checkpoint_dir=/data/sonic] [--option providers=CUDAExecutionProvider]
```

在机器人上，给 `wbc` layer 加上：

```yaml
layers:
  - type: wbc
    profile: sonic/unitree_g1
    runtime: remote
    endpoint: tcp://192.168.123.10:5555
    remote_timeout_s: 0.015
```

或者在 Python 中：`hiwbc.load("sonic/unitree_g1", runtime="remote", endpoint="tcp://...")`。

- 机器人上仍然需要 checkpoint：它先用 checkpoint 构建控制器，再把每次模型调用换成对服务器的调用。
  客户端的 profile 或变体与服务器不同时，服务器会拒绝它。
- 在 `remote_timeout_s` 内没有返回（或发现服务器已不在）的调用算作一个错过的 tick：layer 再次发送上一个
  关节目标。`max_missed_ticks` 默认为 2，连续第三个错过的 tick 会让机器人进入阻尼。客户端在下一次调用时
  重新连接。
- 服务器不保存状态，所以机器人随时可以重连。循环模型（`unitree_loco` 的 `default` LSTM）在两端都会被
  拒绝；请在本地运行它们。
- **该服务没有身份认证。** 除非传入 `--host`，它只监听 127.0.0.1。只在可信的有线网络上使用；Wi-Fi 的
  延迟太不稳定，满足不了 15 ms 的预算。

在 MuJoCo 中行走 8 s，远程推理和本地推理在每个 tick 给出的关节目标完全相同。通过 localhost，每次调用
增加约 0.1 ms。

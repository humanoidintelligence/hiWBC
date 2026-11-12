# 速度命令行走

[English](./locomotion.md) | 简体中文

这些控制器根据摇杆式的命令让机器人行走：`ReferenceKind.VELOCITY`，`value = [vx, vy, wyaw]`（AMO 还有高度和
躯干姿态）。控制循环、关节映射和首次上电见[快速开始](getting_started.zh-CN.md)，请先阅读。

| 控制器 | Policy | G1 checkpoint（RoboJuDo `assets/models/g1/`） | 驱动的关节 |
|---|---|---|---|
| `unitree_loco` | [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym) | `unitree/policy_lstm_1.pt` | 12 个腿部关节 |
| `unitree_loco`（`with_gait=False`） | [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab) | `unitree/policy_wo_gait.pt` | 29 个（IsaacLab 顺序） |
| `smooth_loco` | [Smooth Humanoid Locomotion](https://github.com/zixuan417/smooth-humanoid-locomotion) | `smooth/for_test-2000-jit.pt` | 21 个 |
| `amo_loco` | [AMO](https://github.com/OpenTeleVision/AMO) | `amo/amo_jit.pt` + `adapter_jit.pt` + `adapter_norm_stats.pt` | 驱动 15 个，观测 23 个 |

全部以 50 Hz 运行。下文的关节名省略了 `_joint` 后缀。

## 关节顺序

**12 自由度腿部**（`unitree_loco`；除 IsaacLab 外，下面每种顺序的前 12 个也是它）：

```
left_hip_pitch, left_hip_roll, left_hip_yaw, left_knee, left_ankle_pitch, left_ankle_roll,
right_hip_pitch, right_hip_roll, right_hip_yaw, right_knee, right_ankle_pitch, right_ankle_roll
```

**21 自由度**（`smooth_loco`）：先是 12 个腿部关节，然后是
`waist_yaw, left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw, left_elbow,
right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw, right_elbow`。

**23 自由度**（`amo_loco` 的观测）：先是 12 个腿部关节，然后是 `waist_yaw, waist_roll, waist_pitch`，再是
8 个手臂关节（肩 pitch / roll / yaw、肘；先左后右）。AMO 驱动前 15 个（腿部 + 腰部）。

**29 自由度 IsaacLab 顺序**（`with_gait=False` 的 `unitree_loco`）：关节按左 / 右 / 腰交错排列：

```
left_hip_pitch, right_hip_pitch, waist_yaw, left_hip_roll, right_hip_roll, waist_roll,
left_hip_yaw, right_hip_yaw, waist_pitch, left_knee, right_knee,
left_shoulder_pitch, right_shoulder_pitch, left_ankle_pitch, right_ankle_pitch,
left_shoulder_roll, right_shoulder_roll, left_ankle_roll, right_ankle_roll,
left_shoulder_yaw, right_shoulder_yaw, left_elbow, right_elbow,
left_wrist_roll, right_wrist_roll, left_wrist_pitch, right_wrist_pitch,
left_wrist_yaw, right_wrist_yaw
```

## `unitree_loco`：unitree_rl_gym（带步态）

```python
import numpy as np
import torch
from hiwbc import ControllerState, Reference, ReferenceKind, make_controller

LEG_DEFAULT = [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0] * 2
KP = [100, 100, 100, 150, 40, 40] * 2          # 在机器人一侧设置
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
    # 默认值与 RoboJuDo 的 G1 配置一致：cycle_time=0.8，max_cmd=(0.8, 0.5, 1.57)，
    # 观测缩放 ang_vel 0.25 / dof_vel 0.05 / command (2, 2, 0.25)，
    # action_scale=0.25，action_beta=0.8
)
ctrl.reset()

ref = Reference(ReferenceKind.VELOCITY, value=np.array([0.5, 0.0, 0.0]))
target = ctrl.step(state, ref)   # state：12 个腿部关节，见快速开始
```

**命令是归一化的。** `value` 每个轴的取值范围是 `[-1, 1]`，控制器把它乘以 `max_cmd`，所以 `[0.5, 0, 0]`
请求的是向前 `0.5 * 0.8 = 0.4 m/s`。正方向是向前、向左和逆时针。要用物理单位下命令，先除以 `max_cmd`。

步态时钟（`phase = t / (50 * 0.8)`）从 `reset()` 开始运行，不论机器人是否在动。`policy_lstm_1.pt` 是循环
网络：它的隐藏状态保存在模型对象中，所以每个控制器加载一个模型。`ctrl.reset()` 把它恢复到刚加载时的状态
（`load_torchscript_runner` 会保存缓冲区的快照）。

手臂和腰部不被驱动：用你自己的 PD 增益把它们保持在固定姿态。

## `unitree_loco`：unitree_rl_lab（`with_gait=False`）

unitree_rl_lab 的 policy 按 IsaacLab 顺序驱动全部 29 个关节，没有步态时钟，并接收 5 个控制周期的历史。

```python
from hiwbc.controllers.unitree import ObsScales

G1_29_DEFAULT = [  # IsaacLab 顺序
    -0.1, -0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.3, 0.3, -0.2, -0.2,
    0.25, -0.25, 0.0, 0.0, 0.0, 0.0, 0.97, 0.97, 0.15, -0.15, 0.0, 0.0, 0.0, 0.0,
]

ctrl = make_controller(
    "unitree_loco",
    num_actions=29,
    default_pos=G1_29_DEFAULT,
    forward_fn=forward_fn,               # 包装 policy_wo_gait.pt
    with_gait=False,
    history_length=5,
    gravity_scale=1.0,
    obs_scales=ObsScales(ang_vel=0.2, dof_vel=0.05, command=(1.0, 1.0, 1.0)),
    action_beta=1.0,                     # 这个 policy 训练时没有 EMA
)
```

观测由 6 项组成（角速度、重力、命令、关节位置、关节速度、上一个动作），每项保留 5 个控制周期：
`5 * (9 + 3 * 29) = 480` 个值。历史按项排列（term-major），且包含当前控制周期，所以 `reset()` 之后的第一个
控制周期看到的是四个全零帧，然后是当前帧。命令仍然是归一化的，并乘以 `max_cmd`。

## `smooth_loco`

```python
from hiwbc.controllers.smooth import SmoothObsScales

SMOOTH_DEFAULT = [-0.4, 0.0, 0.0, 0.8, -0.35, 0.0] * 2 + [0.0] * 9   # 21 自由度顺序

ctrl = make_controller(
    "smooth_loco",
    num_actions=21,
    default_pos=SMOOTH_DEFAULT,
    forward_fn=forward_fn,               # 包装 for_test-2000-jit.pt
    cycle_time=0.64,
    n_priv_latent=50,                    # 4 + 1 + 2 * 21 + 3，填零
    history_length=10,
    obs_scales=SmoothObsScales(ang_vel=0.25, dof_vel=0.05),
    action_scale=0.5,
    action_clip=10.0,
    action_beta=0.8,
)
```

观测由当前的本体感知块（`10 + 3 * 21 = 73`）、50 个零组成的特权信息槽，以及之前的 10 个块（最旧的在前）
组成：`73 + 50 + 730 = 853` 个值。

命令不经缩放直接进入观测。RoboJuDo 的 G1 配置把摇杆映射到下面的范围（摇杆最小 / 中位 / 最大）；照此复现，
才能给出 policy 训练时见过的命令：

| `value` 索引 | 摇杆 | 范围 |
|---|---|---|
| 0 | 左摇杆 Y | -1.5 / **-0.5** / 0.5 |
| 1 | 左摇杆 X | 0.25 / 0.0 / -0.25 |
| 2 | 右摇杆 X | 0.5 / **-0.5** / -1.5 |

中位命令是 `[-0.5, 0.0, -0.5]`，不是零。在逐周期对比检查中，这个 checkpoint（`for_test-2000`）在按脚本
行走约 7 s 后摔倒，单独在 RoboJuDo 下也同样摔倒；把它当作测试用的 checkpoint。

## `amo_loco`

AMO 用速度命令控制下半身，用高度和躯干姿态控制上半身。一个 adapter MLP 把这个命令连同手臂关节角度转换成
观测的一部分；policy 还有第二个输入，即最近 25 帧本体感知。

```python
from hiwbc import make_controller
from hiwbc.controllers.amo import load_amo_runners

AMO_OBS_DEFAULT = (
    [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0] * 2         # 腿部
    + [0.0, 0.0, 0.0]                            # 腰部 yaw / roll / pitch
    + [0.5, 0.0, 0.2, 0.3, 0.5, 0.0, -0.2, 0.3]  # 手臂
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

`load_amo_runners` 需要 `torch` extra。它返回 `forward_fn`、`adapter_fn` 和 adapter 的归一化统计量；在 CPU
上运行时，它会修补已发布 TorchScript 文件中写死的 CUDA 设备常量。

`state.dof_pos` / `dof_vel` 包含全部 23 个观测关节。`target.position` 覆盖 15 个被驱动的关节；手臂不由 AMO
驱动。把手臂保持在你选定的姿态：它们的实测角度会进入观测，所以 policy 会为它们保持平衡。

命令有 7 个元素（或 3 个，此时后 4 个取自 `default_command`）：

| 索引 | 含义 | RoboJuDo 摇杆范围（最小 / 中位 / 最大） |
|---|---|---|
| 0 | 线速度，来自左摇杆 Y | -1.0 / 0.0 / 1.0 |
| 1 | 偏航命令；偏航项为 `dyaw = -value[1]` | 0.2 / 0.0 / -0.2（右摇杆 X） |
| 2 | 线速度，来自左摇杆 X | 0.8 / 0.0 / -0.8 |
| 3 | 基座高度，m | 静止时 0.75；RoboJuDo 允许 0.3 到 0.9 |
| 4, 5, 6 | 躯干 yaw、pitch、roll，rad | 静止时为 0 |

RoboJuDo 的配置把索引 0 标为 `vel_y`、索引 2 标为 `vel_x`；在硬件上使用之前，先在仿真中确认各轴方向。

```python
# 先行走，然后蹲到 0.6 m，同时让躯干前倾
walk   = Reference(ReferenceKind.VELOCITY, value=np.array([0.0, 0.0, 0.4]))
crouch = Reference(ReferenceKind.VELOCITY,
                   value=np.array([0.0, 0.0, 0.0, 0.6, 0.0, 0.3, 0.0]))
```

当三个速度元素都小于 0.1 时，步态时钟让双脚同相（站立）；超过这个值时切换为反相。逐周期对比检查覆盖了站立、
行走、转向、侧移、蹲到 0.6 m 并让躯干 pitch，以及躯干 yaw / roll，持续 24 s 没有摔倒。

有两个行为按原样照搬自 RoboJuDo，因为 checkpoint 就是在这样的条件下运行的：观测中的关节速度没有加掩码
（RoboJuDo 计算了踝关节 / 腰部的掩码，但没有应用），以及示例观测中的高度元素是 `0.75 + value[3]`。

## 增益

来自 RoboJuDo 的 G1 配置（顺序为各控制器的关节顺序）：

| 控制器 | kp | kd |
|---|---|---|
| `unitree_loco`（带步态） | 腿部 `[100, 100, 100, 150, 40, 40] * 2` | `[2, 2, 2, 4, 2, 2] * 2` |
| `amo_loco` | 腿部 `[150, 150, 150, 300, 80, 20] * 2`，腰部 `[400] * 3`，手臂 `[80, 80, 40, 60] * 2` | 腿部 `[2, 2, 2, 4, 2, 1] * 2`，腰部 `[15] * 3`，手臂 `[2, 2, 1, 1] * 2` |
| `smooth_loco` | 腿部 `[200, 150, 150, 200, 20, 20] * 2`，腰部 `200`，手臂 `[40] * 8` | 腿部 `[5, 5, 5, 5, 4, 4] * 2`，腰部 `5`，手臂 `[10] * 8` |

`with_gait=False` 的 `unitree_loco`，按 IsaacLab 顺序：

```python
KP = [100, 100, 200, 100, 100, 40, 100, 100, 40, 150, 150] + [40] * 18
KD = [2, 2, 5, 2, 2, 5, 2, 2, 5, 4, 4, 10, 10, 2, 2, 10, 10, 2, 2] + [10] * 10
```

## 常见的坑

- **给 `unitree_loco` 传入以 m/s 为单位的命令。** 控制器会乘以 `max_cmd`；`value=[0.8, 0, 0]` 请求的是
  0.64 m/s，不是 0.8。
- **`smooth_loco` 的中位命令。** 零不代表"站住"；像 RoboJuDo 那样使用 `[-0.5, 0, -0.5]`。
- **AMO 的关节子集。** `default_pos` 必须等于 `obs_default_pos[:15]`，否则构造函数会报错。`state` 中要传入
  全部 23 个关节。
- **跨 episode 的 LSTM 状态。** 只有当 runner 带有 `reset()` 时（`load_torchscript_runner` 有），`ctrl.reset()`
  才会重置隐藏状态；包装循环网络模型的手写 `forward_fn` 必须自己提供重置。

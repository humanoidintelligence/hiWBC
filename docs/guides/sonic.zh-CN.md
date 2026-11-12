# SONIC

[English](./sonic.md) | 简体中文

`sonic` 在 29 自由度的 Unitree G1 上运行 NVIDIA 的
[GEAR-SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl) 行为基础模型。SONIC 由两个模型组成：

- **编码器**把一个参考（G1 动作帧、VR 三点遥操作目标或 SMPL 人体关节）映射为一个 64 维的动作 token；
- **解码器**把 token 加上 10 个 tick 的本体感知历史映射为 29 个关节动作。

VLA 可以直接输出 token（`ReferenceKind.LATENT`），这样就跳过了编码器。请先阅读
[快速开始](getting_started.zh-CN.md)。

hiWBC 的 `sonic` 遵循 NVIDIA 的 C++ 部署代码（`gear_sonic_deploy`）。在 MuJoCo 中与这份 C++ 的逐行移植
对比 600 个 tick，每个编码器输入、token、解码器输入和 PD 目标的差异都在 1e-5 以内。

## 1. 获取 checkpoint

hiWBC 不附带 SONIC 权重。请从 [nvidia/GEAR-SONIC](https://huggingface.co/nvidia/GEAR-SONIC)
下载（NVIDIA Open Model License）：

```bash
pip install -e '.[onnx]' huggingface_hub
huggingface-cli download nvidia/GEAR-SONIC --local-dir ~/.cache/hiwbc/sonic
```

| 变体 | 文件 | 编码器输入 | 解码器输入 |
|---|---|---|---|
| `default` | `model_encoder.onnx`、`model_decoder.onnx` | 1762 | 994 |
| `low_latency` | `low_latency/` | 1247 | 994 |
| `sonic_v1_1` | `sonic_v1_1/` | 1751 | 994 |

每个文件夹里都有一个 `observation_config.yaml`；`hiwbc.configs.sonic.unitree_g1` 中的
`SONIC_VARIANTS` 与这些文件中的项顺序一致。同一次下载还包含行走 planner（规划器）
`planner_sonic.onnx`（740 MB，所有变体共用；见[第 7 节](#7-行走速度命令与-planner)）。

## 2. 构建控制器

profile 从 checkpoint 文件夹加载一个变体：

```python
import hiwbc

ctrl = hiwbc.load("sonic/unitree_g1", variant="default")   # checkpoint_dir: ~/.cache/hiwbc/sonic
# 只用 LATENT 时设 encoder=False；encoder_mode="joint" | "teleop" | "smpl"
# planner=False 跳过行走 planner；planner_thread=True 在控制线程之外重新规划
```

或者直接构建：

```python
from hiwbc.configs.sonic.unitree_g1 import SONIC_G1, SONIC_VARIANTS
from hiwbc.controllers.sonic import SonicController

ckpt = "~/.cache/hiwbc/sonic"
ctrl = SonicController.from_onnx(
    f"{ckpt}/model_decoder.onnx",
    f"{ckpt}/model_encoder.onnx",   # 只用 LATENT 时省略
    robot=SONIC_G1,                 # 29 自由度 G1 的关节顺序、站立姿态和增益
    obs=SONIC_VARIANTS["default"],
    encoder_mode="joint",
)
ctrl.reset()
print(ctrl.encoder_dim, ctrl.decoder_dim)   # 1762 994
```

每次 `step` 返回 `JointTarget(position, kp, kd)`：SONIC 自带每个关节的增益（`SONIC_G1.kp`、
`SONIC_G1.kd`），所以要把它们和目标一起发送。

## 3. 关节顺序

SONIC 使用 **IsaacLab 顺序**（`G1_JOINT_NAMES`）。机器人和 MuJoCo 使用 SDK 顺序
（`G1_JOINT_NAMES_MUJOCO`）。两者都使用 hiRobot 的关节名（没有 `_joint` 后缀），所以 hiRobot 的
`DofMap` 可以按名称完成重排；`ISAACLAB_TO_MUJOCO_INDEX` 是同一个置换的索引数组形式：

```python
import numpy as np
from hiwbc import ControllerState
from hiwbc.configs.sonic.unitree_g1 import ISAACLAB_TO_MUJOCO_INDEX as I2M

# 机器人（SDK 顺序）-> SONIC（IsaacLab 顺序）
state = ControllerState(
    dof_pos=q_sdk[I2M], dof_vel=dq_sdk[I2M],
    base_quat=quat_wxyz[[1, 2, 3, 0]],      # SDK / MuJoCo 给出的是 [w, x, y, z]
    base_ang_vel=gyro,
)
target = ctrl.step(state, ref)

# SONIC -> 机器人
q_cmd = np.empty(29); q_cmd[I2M] = target.position
kp = np.empty(29); kp[I2M] = target.kp
kd = np.empty(29); kd[I2M] = target.kd
```

## 4. 跟踪参考动作（`joint` 模式）

有了 hiRobot 的 `MotionClip`（空间为 `joint` 或 `smpl`，带 `root_quat`），`ClipPlayer` 会在每个
tick 构建编码器窗口：

```python
from hiwbc.references import ClipPlayer

player = ClipPlayer(clip, ctrl.control_dt)
ctrl.check_clip(player)
while not player.done:
    state = read_state()
    send(ctrl.step(state, ctrl.clip_reference(player, state)))
    player.advance()
```

player 把动作片段重采样到 50 Hz，在第一个 tick 把它的朝向和 xy 起点对齐到机器人（即部署代码中的
`apply_delta_heading`），传入编码器读取的窗口（`default` 为 46 帧），并把 `encoder_mode` 设为片段的
空间。在两个 GR00T 片段（497 和 1375 个 tick）、全部三个变体上，机器人朝向从 1.3 rad 开始并持续漂移，
与部署代码风格的 player 对比：编码器输入的差异在 4.8e-7 以内。

自己构建帧：

`joint` 编码器（部署代码中的 `g1` 模式）接收未来的 G1 帧：`joint_pos` 和 `joint_vel`（T, 29），
IsaacLab 顺序；`root_quat`（T, 4），`[x, y, z, w]`；以及 `root_z`（T,），即参考根节点高度，
`default` 变体会读取它。传入从当前帧开始的窗口；控制器按每个项的步长采样（`default` 为每 5 帧取一帧），
窗口不够长时重复最后一帧。`default` 配置向前看 45 帧，所以传入 46 帧。

```python
from hiwbc import Reference, ReferenceKind

# motion：来自动作片段的 50 Hz 数组，例如部署代码的参考 CSV
# joint_pos.csv / joint_vel.csv（IsaacLab 顺序）、body_quat.csv（根节点 = body 0，wxyz）
T = len(motion["joint_pos"])
for t in range(T):
    win = slice(t, min(t + 46, T))
    ref = Reference(ReferenceKind.MOTION_FRAMES, frames={
        "joint_pos": motion["joint_pos"][win],
        "joint_vel": motion["joint_vel"][win],
        "root_quat": aligned_root_xyzw[win],      # 见下文
        "root_z": motion["body_pos"][win, 0, 2],  # 根节点高度，m
    })
    target = ctrl.step(read_state(), ref)
    send(target)
```

自己构建帧时，**朝向对齐**由你负责。朝某个方向录制的片段必须只绕偏航旋转，使它的第一帧朝向与机器人
起始时的朝向一致：

```python
from hiwbc.math_utils import quat_mul, quat_conjugate

def yaw_only(q):                       # [x, y, z, w]
    q = np.array([0.0, 0.0, q[2], q[3]])
    return q / np.linalg.norm(q)

align = quat_mul(yaw_only(robot_quat_at_start), quat_conjugate(yaw_only(clip_root_xyzw[0])))
aligned_root_xyzw = np.array([quat_mul(align, q) for q in clip_root_xyzw])
```

传入 `extra={"play": False}` 可以暂停：控制器保持第 0 帧并把速度置零，与部署代码相同。

## 5. 用 token 驱动 SONIC（`LATENT`）

```python
ref = Reference(ReferenceKind.LATENT, value=token)     # (64,)，例如 VLA 输出的动作 token
target = ctrl.step(state, ref)
```

编码器不运行。`target.extra["motion_token"]`（以及 `ctrl.token`）保存所用的 token，所以在
`MOTION_FRAMES` 模式下运行时可以录下 token，之后通过 `LATENT` 重放。用这种方式重放编码器自己的 token，
跟踪效果与运行编码器相同。

## 6. Teleop 与 SMPL 模式

| 模式 | `frames` 键 |
|---|---|
| `teleop` | `joint_pos`、`joint_vel`（内部选取下半身）、`root_quat`、`vr_3point_pos`（3, 3）和 `vr_3point_quat`（3, 4，`[x, y, z, w]`），依次为左腕、右腕、头部，位于参考根节点坐标系 |
| `smpl` | `smpl_joints`（T, 24, 3），相对根节点的 SMPL 关节；`root_quat`；`joint_pos`（内部选取腕关节） |

用 `extra={"encoder_mode": "teleop"}` 逐 tick 切换。模式需要但 `frames` 中缺少的键会抛出
`KeyError`；C++ 部署代码则会静默回退到另一种模式。

## 7. 行走：速度命令与 planner

SONIC 没有速度输入。NVIDIA 的 C++ 部署代码用一个单独的运动学 planner（`planner_sonic.onnx`）让它行走：
planner 根据最近 4 帧和一条移动命令预测接下来 64 帧的 G1 动作，SONIC 再以 `joint` 模式跟踪这些帧。
`sonic/unitree_g1` profile 默认加载 planner（`planner=True`，需要 `encoder=True`），此时控制器也接受
`VELOCITY`：

```python
ref = Reference(ReferenceKind.VELOCITY, value=np.array([0.5, 0.0, 0.0]))   # [vx, vy, wz]，本体坐标系
target = ctrl.step(state, ref)
```

`[0, 0, 0]` 表示站住（planner 的 idle 模式）。`hiwbc.controllers.sonic_planner` 遵循部署代码的 planner
流程：

- `[vx, vy, wz]` 按手柄的方式变成 planner 命令：`wz` 转动命令朝向，最多领先机器人实际朝向 90°；速度
  决定模式：低于 0.05 m/s 为 idle，0.8 以下为慢走，1.5 以下为行走，3.0 m/s 以下为跑。
- 预测结果（30 Hz）被重采样到 50 Hz，并混入正在播放的动作：从取 planner 上下文的那一帧之后 2 帧开始，
  在 8 帧内线性过渡。
- 以 10 Hz 检查 planner，命令变化时重新规划，移动中每 0.1–1 s 也重新规划一次。站立时，如果下半身参考
  与实测关节偏离，就把参考往实测关节拉。
- 从 `VELOCITY` 切换到别的参考（动作片段、token）再切回来时，planner 从实测关节重新开始。

一次 planner 调用在台式机 CPU 上约需 35 ms，比 20 ms 的 tick 还长。默认（`planner_thread=False`）下，
它在需要重新规划的 tick 里于 `step` 内部运行；真机应使用 `planner_thread=True`，它在工作线程上运行
planner，结果就绪后再混入，与部署代码一致。

**边走边跟踪手部目标。** 在 `VELOCITY` 参考上设置 `extra["ee_targets"]` 会把编码器切换到 `teleop`：
planner 仍然驱动下半身，VR 三点项来自给定的手腕位姿（`{"left_wrist": [x, y, z, qx, qy, qz, qw], ...}`，
根节点坐标系；缺少的点保持部署代码的默认姿态）。要对目标做速率限制：一个 tick 内 15–20 cm 的跳变让
SONIC 输出了接近 −4 rad 的肘关节目标。通过 `wbc` layer 使用 hiRobot 的 `CARTESIAN` 模式时，限制会替你
完成（[作为 hiRobot 机器人运行](robot.zh-CN.md#3-命令)）。

通过 `wbc` layer 在 MuJoCo 中的结果（2 ms 物理步长，50 Hz 控制）：

| 命令 | 结果 |
|---|---|
| 站立 10 s | 前 2 s 移动了 5 cm，之后静止 |
| vx 0.5 m/s，8 s | 3.44 m（0.43 m/s），侧向漂移 0.27 m |
| wz 0.5 rad/s，6 s | 151°（0.44 rad/s） |
| `BASE_POSE` 到 (1.5 m, 1.0 m, 90°) | 8.7 s 到达，误差 9.7 cm / 8° 以内 |
| 距静止位置 16–29 cm 的手部目标 | 达到 5–12 cm 以内；手的移动距离约为要求的 1.3 倍 |
| 以 0.4 m/s 行走并举起左手 | 5 s 走了 1.61 m，手腕平均误差 10.9 cm |

机器人一次也没有摔倒。行走比命令慢，手部跟踪有过冲；这两点都还没有与 C++ 部署代码对比过。

## 8. 仿真结果

在 MuJoCo 中使用 GR00T 的 G1 场景运行，50 Hz 控制：

| 动作 | 变体 | tick 数 | 平均关节误差（rad） | 摔倒 |
|---|---|---|---|---|
| macarena | default | 600 | 0.087 | 否 |
| macarena | low_latency | 600 | 0.103 | 否 |
| macarena | sonic_v1_1 | 600 | 0.104 | 否 |
| macarena | default，token 经 `LATENT` | 600 | 0.087 | 否 |
| neutral_kick | default | 165（片段结束） | 0.097 | 否 |
| forward_lunge | default | 399（片段结束） | 0.104 | 否 |
| dance_in_da_party | default | 497（片段结束） | 0.130 | 否 |

**物理步长。** 这些运行使用 2 ms 的物理步长。在 5 ms（GR00T 仿真配置中的步长）下，机器人约 1.4 s 后
摔倒。C++ 移植版本也以同样的方式摔倒，所以原因不在 hiWBC 的观测和动作代码；真正的原因尚未找到，GR00T
自己的仿真循环可能做了这个测试环境没做的事。在 MuJoCo 中请使用 2.5 ms 或更小的步长。

## 常见的坑

- **关节顺序。** SONIC 用 IsaacLab 顺序；机器人用 SDK 顺序。状态和目标都要映射，包括 `kp` / `kd`。
- **增益。** 发送 `target.kp` / `kd`。机器人的默认增益不是 SONIC 的增益。
- **朝向。** 未对齐的 `root_quat` 会让机器人一开始就转向片段录制时的朝向。`ClipPlayer` 会做对齐；你自己
  构建的帧必须先对齐。
- **Planner 延迟。** 在真机上设置 `planner_thread=True`；同步重新规划耗时超过一个 tick。
- **窗口太短。** 传入足够多的未来帧（`default` 为 46 帧）；帧数不够时会重复最后一帧，机器人的预判会变少。

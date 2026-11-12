# 快速开始

[English](./getting_started.md) | 简体中文

本指南介绍每个控制器都需要的内容，与它运行哪个 policy 无关：控制循环、从机器人数据构建
`ControllerState`、映射关节顺序、把 checkpoint 包装成前向函数、PD 增益，以及首次上电。各类控制器的指南
（[行走](locomotion.zh-CN.md)、[动作跟踪](motion_tracking.zh-CN.md)、[SONIC](sonic.zh-CN.md)）在此基础上展开。
如果想让 hiRobot 运行这个循环而不是自己写，见[作为 hiRobot 机器人运行](robot.zh-CN.md)。

## 1. 安装

```bash
git clone https://github.com/humanoidintelligence/hiWBC.git && cd hiWBC
pip install -e .            # 核心：numpy + hiRobot（控制器只导入 numpy）
pip install -e '.[torch]'   # TorchScript checkpoint（.pt）
pip install -e '.[onnx]'    # ONNX checkpoint（.onnx）
python -m pytest            # 使用桩模型，不需要 GPU 或权重
```

## 2. 控制循环

控制器是一个纯粹的单步函数：`(ControllerState, Reference) -> JointTarget`。它从不直接与硬件通信。
调用方（通常是 hiRobot）以 policy 的控制频率运行这个循环（所有内置控制器都是 50 Hz）：

```python
import time
import numpy as np
import hiwbc
from hiwbc import ControllerState, Reference, ReferenceKind

ctrl = hiwbc.load("unitree_loco/unitree_g1")   # 或 make_controller(...)，见各类控制器的指南
ctrl.reset()                                    # 每个 episode 调用一次

DT = 0.02                                       # 50 Hz
next_tick = time.perf_counter()
while running:
    raw = robot.read()                          # 你的机器人接口
    state = ControllerState(
        dof_pos=raw.q[policy_idx],              # policy 的关节顺序（第 4 节）
        dof_vel=raw.dq[policy_idx],
        base_quat=raw.imu_quat_xyzw,            # [x, y, z, w]
        base_ang_vel=raw.imu_gyro,              # 本体坐标系，rad/s
    )
    ref = Reference(ReferenceKind.VELOCITY, value=np.array([0.5, 0.0, 0.0]))
    target = ctrl.step(state, ref)

    q_cmd = default_full_body.copy()            # policy 不驱动的关节
    q_cmd[policy_idx] = target.position         # 换回机器人的顺序
    kp[policy_idx], kd[policy_idx] = target.kp, target.kd   # 增益，控制器提供时使用（第 6 节）
    robot.send_position(q_cmd, kp=kp, kd=kd)    # 由机器人的 PD 环跟踪

    next_tick += DT
    time.sleep(max(0.0, next_tick - time.perf_counter()))
```

每次 `step` 依次执行 `observe` → `forward` → 后处理（EMA、裁剪、缩放、加上默认姿态），返回一个绝对的
PD 位置目标。带有时钟或历史的控制器（步态相位、动作相位、本体感知历史）在 `step` 内推进它们，所以每个
控制周期恰好调用一次 `step`，并在每个 episode 开始时调用 `reset()`。

## 3. 构建 `ControllerState`

| 字段 | 填什么 | 常见错误 |
|---|---|---|
| `dof_pos`、`dof_vel` | 实测的关节位置 / 速度，按 **policy 的**关节顺序 | 传入机器人原生的顺序 |
| `base_quat` | 骨盆（IMU）在世界坐标系中的姿态，**`[x, y, z, w]`** | Unitree SDK 和 MuJoCo 给出的是 `[w, x, y, z]`：用 `q[[1, 2, 3, 0]]` 重排 |
| `base_ang_vel` | 骨盆在**本体**坐标系中的角速度，rad/s | 使用世界坐标系的值。IMU 陀螺仪本来就是本体坐标系 |
| `base_lin_vel` | 本体坐标系中的线速度，仅当有状态估计器提供时 | 只有使用状态估计器的 `beyondmimic_tracker` checkpoint（`violin`、`waltz`）会读取它 |
| `extra` | 可选的额外数据。`torso_quat`（`[x, y, z, w]`）：躯干姿态，如果你能测量（BeyondMimic；否则由腰部关节计算） | |

在 IMU 装在骨盆上的 Unitree G1 上，`base_quat` 和 `base_ang_vel` 直接取自 `low_state.imu_state`
（四元数需重排）。在 MuJoCo 中，它们是自由关节的 `qpos[3:7]`（需重排）和 `qvel[3:6]`。

## 4. 关节顺序

每个 policy 训练时都有自己的关节顺序和关节子集，通常与机器人的不同。`dof_pos`、`dof_vel`、`default_pos`
和 `JointTarget.position` 都按 policy 的顺序。由 profile 构建的控制器在 `ctrl.joint_names` 中列出这个顺序
（hiRobot 名称，不带 `_joint` 后缀），并在 `ctrl.obs_joint_names` 中列出 `dof_pos` 必须包含的关节。
在启动时按关节名映射一次：

```python
robot_joints = [...]       # 机器人的顺序，例如 Unitree SDK / MuJoCo 的顺序
policy_joints = ctrl.joint_names   # policy 训练时用的顺序
policy_idx = np.array([robot_joints.index(n) for n in policy_joints])

state_q = raw.q[policy_idx]              # 机器人 -> policy
q_cmd[policy_idx] = target.position      # policy -> 机器人
```

policy 不驱动的关节（例如 23 自由度 policy 下的手腕）保持固定姿态：给它们发送默认位置，增益用 policy
配置中为它们给出的值。

内置示例使用的关节顺序见各类控制器的指南。有两个陷阱：

- 对 29 自由度的 G1，Unitree SDK / MuJoCo 的顺序（`left_hip_pitch, left_hip_roll, ...`）与 IsaacLab 的顺序
  （`left_hip_pitch, right_hip_pitch, waist_yaw, ...`）不同。用 IsaacLab 训练的 policy（BeyondMimic、
  unitree_rl_lab、SONIC）使用 IsaacLab 顺序。
- AMO 观测 23 个关节，但只驱动前 15 个。被驱动的关节必须排在观测关节的最前面。

## 5. 前向函数

控制器从不加载 checkpoint。它们接收一个 `forward_fn(obs) -> action`，所以测试可以传入桩函数，你也可以使用
任何运行时。profile（`hiwbc.load`）会替你构建前向函数；本节其余部分针对手动构建控制器的情况。

TorchScript（`.pt`）：

```python
import numpy as np
import torch

model = torch.jit.load("policy.pt", map_location="cpu").eval()

@torch.no_grad()
def forward_fn(obs):
    return model(torch.from_numpy(np.asarray(obs, np.float32))[None]).numpy().squeeze()
```

ONNX（`.onnx`），单输入：

```python
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession("policy.onnx", providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0].name

def forward_fn(obs):
    return sess.run(None, {inp: np.asarray(obs, np.float32)[None]})[0].squeeze()
```

接入 checkpoint 之前，先检查它的输入名称和尺寸；尺寸不匹配是观测布局出错的第一个信号：

```python
print([(i.name, i.shape) for i in sess.get_inputs()])
print([o.name for o in sess.get_outputs()])
```

> 循环结构的 TorchScript policy（例如 `policy_lstm_1.pt`）把隐藏状态保存在模型对象内部。每个控制器使用一个
> 模型对象。`hiwbc.load_torchscript_runner` 会保存缓冲区的快照，`ctrl.reset()` 将其恢复；手写的
> `forward_fn` 必须自己重置隐藏状态。

多输入模型（`kungfubot_tracker`、`supermimic`、`amo_loco`）接收一个元组；各类控制器的指南给出了到模型输入的
对应关系。

## 6. PD 增益

`JointTarget.position` 是给机器人关节 PD 环的位置目标。增益属于 policy：使用 policy 训练时的刚度和阻尼。
由 profile 构建的控制器在 `JointTarget.kp` / `kd` 中返回增益（来自 RoboJuDo 的配置、ONNX 元数据，或
SONIC / SuperMimic 的配置）；把它们和目标一起发送。用 `make_controller` 构建且未传 `kp` / `kd` 的控制器返回
`None`，此时在机器人一侧配置增益（各类控制器的指南列出了这些增益）。

增益错误是接线正确的 policy 仍然摔倒的常见原因：在髋关节 kp 为 100 时训练的 policy，在 kp 为 40 时站不住。

## 7. 首次上电

1. **先仿真。** 在 MuJoCo 中运行完整的循环，关节映射、增益和控制频率与将要在硬件上使用的完全相同。
2. **把机器人吊起或支撑住**，清空周围区域，并有人守在急停旁。
3. **插值到默认姿态**：从实测姿态出发，用 2 到 3 s 过渡，增益从低值升到 policy 的增益。绝不能让第一条命令
   直接跳到目标。hiRobot 的 `reset()` 会这样做。
4. **然后调用 `ctrl.reset()` 并开始调用 `step`。** 控制器的历史和时钟从这一刻开始；之前运行残留的历史会让
   最初的动作出错。
5. 先保持站立（零速度命令，或动作的第一帧），再把机器人放到地面上。

## 8. 常见的坑

| 现象 | 可能的原因 |
|---|---|
| `forward returned shape (k,), expected (n,)` | `num_actions` 与 checkpoint 不匹配 |
| ONNX 输入报 "invalid dimensions" | 观测尺寸错误：检查 `history_length`、`use_history`、`n_priv_latent`、`frames` 中的项、`without_state_estimator` |
| 机器人 1 s 内摔倒，而 policy 在它自己的仓库里表现正常 | 关节顺序、四元数顺序（`wxyz` 与 `xyzw`）、增益，或历史布局不一致 |
| 机器人持续向一侧倾斜 | `default_pos` 不是 policy 的关节顺序，或映射中某个关节的符号反了 |
| 起步几步发抖 | 第一次 `step` 之前没有调用 `reset()`，或机器人不是从默认姿态起步 |
| 跟踪器几秒后漂移 | 上游计算参考项时用错了坐标系（anchor、朝向对齐） |

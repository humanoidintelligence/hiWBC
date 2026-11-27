# 参考动作跟踪

[English](./motion_tracking.md) | 简体中文

这些控制器让机器人跟随参考动作。它们接受 `ReferenceKind.MOTION_FRAMES`，从 `reference.frames`
读取当前控制周期的参考项；其中一些会自己从动作片段或 checkpoint 中的动作构建这些项。请先阅读
[快速上手](getting_started.zh-CN.md)。

| 控制器 | Policy | G1 checkpoint | 关节 | `frames` |
|---|---|---|---|---|
| `asap_tracker` | [ASAP](https://github.com/LeCAR-Lab/ASAP)、KungfuBot v1 | `asap/mimic/<motion>/*.onnx` | 23 | 无（动作已训练进 checkpoint，是一个行为） |
| `twist_tracker` | [TWIST](https://github.com/YanjieZe/TWIST) | `twist/twist_general_motion_tracker.pt` | 23 | `mimic_target`，或一个动作片段 |
| `beyondmimic_tracker` | [BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking) | `beyondmimic/{Jump_wose,Dance_wose,Violin,Waltz}.onnx` | 29（IsaacLab） | `command`、`motion_anchor_ori_b`（+ `motion_anchor_pos_b`、`base_lin_vel`）；或 checkpoint 中的动作，作为行为 |
| `kungfubot_tracker` | [KungfuBot2 / PBHC](https://github.com/TeleHuman/PBHC) | `kungfubot2/horse_test_43000.onnx` | 23 | anchor 旋转、下一步和未来动作项 |
| `protomotions_tracker` | [Human2Humanoid](https://github.com/LeCAR-Lab/human2humanoid) 学生模型（ProtoMotions 格式） | `h2h/h2h.pt` | 21 | `task_obs` |
| `supermimic` | SuperMimic，非官方（[supermimic_deploy](https://github.com/KyrieYoung926/supermimic_deploy)） | bundle 中的 `policy.onnx` | 29（SDK 顺序） | `goal`，或一个动作片段 |
| `blinkmimic` | [BlinkMimic](https://github.com/fuyichen0725/blinkmimic) | 你自己训的：`blinkmimic/policy.pt`，或单动作的 `.onnx` 导出文件 | 29（SDK 顺序） | `goal`、`ref_root_quat`，或一个动作片段；`.onnx` 导出文件播放自带的动作，作为行为 |
| `blinkmimic`（`xmimic` variant） | [XMimic](https://github.com/nico-bohlinger/xmimic) | 你自己训的：`xmimic/policy.npz` | 29（SDK 顺序） | 同 `blinkmimic` |

## 参考从哪里来

每个跟踪器都需要当前控制周期的参考项。有三种方式获得：

| 来源 | 控制器 | 做法 |
|---|---|---|
| 动作片段 | `twist_tracker`、`supermimic`、`blinkmimic`、`kungfubot_tracker`、`protomotions_tracker`（H2H profile）、`sonic` | 把 hiRobot 的 `MotionClip` 包进 `ClipPlayer`；`ctrl.clip_reference(player, state)` 每个 tick 构建参考项 |
| checkpoint 自带的动作 | `asap_tracker`、`beyondmimic_tracker`、`blinkmimic`（`onnx` variant） | `ctrl.start_behavior(name)` 返回一个每个 tick 构建参考项的 source |
| 你自己的代码 | 全部 | 自己计算参考项，按下文的名称放进 `frames` |

最快的入门方式是 profile，它用 checkpoint 训练时的取值加载 checkpoint（`checkpoint_dir` 默认为
`$HIWBC_ROBOJUDO_MODELS`）：

```python
import hiwbc
from hiwbc.references import ClipPlayer

ctrl = hiwbc.load("twist_tracker/unitree_g1", checkpoint_dir="RoboJuDo/assets/models/g1")
player = ClipPlayer(clip, ctrl.control_dt, **ctrl.clip_player_options)   # clip：hiRobot MotionClip，space 为 "joint"
ctrl.check_clip(player)                      # 缺少 root_pos / root_quat / 关节时抛出 ValueError
ctrl.reset()
while not player.done:
    state = read_state()
    send(ctrl.step(state, ctrl.clip_reference(player, state)))
    player.advance()
```

player 把片段重采样到 50 Hz，在第一个控制周期把它转到机器人的朝向，计算片段中没有的速度，并按名称
匹配关节（[API：动作片段](../api.zh-CN.md#动作片段)）。

如果用你自己的代码，下面是 RoboJuDo 的动作控制器（`robojudo/controller/motion_*_ctrl.py`）执行的
步骤，也是每个 checkpoint 的参考实现：

1. **推进动作时钟**（50 Hz，除非改变播放速度，否则每个 tick 前进一帧），并采样 policy 需要的参考帧。
2. **把动作对齐到机器人**，通常在开始时按 yaw（朝向）和 xy 位置对齐，这样朝 +x 录制的片段会朝机器人
   面对的方向播放。
3. **计算 policy 的参考项**：关节目标、根高度和速度，以及需要刚体位姿的项（机器人坐标系中的 anchor
   姿态、关键刚体位置）。机器人自身的刚体位姿由实测关节角做 FK 得到。
4. 把这些项按下文的名称放进 `frames`。

`kungfubot_tracker` 的关键刚体项和 `protomotions_tracker` 的 `task_obs` 需要刚体 FK。它们的 profile
在 G1 29 自由度 MJCF（选项 `mjcf`）上使用 `hirobot.geometry.kinematics`，只在跟踪片段时才导入。与
PHC 的 motion library 一样，player 以片段自己的帧率做 FK，并平滑刚体和根的速度（高斯，sigma = 2 个
源帧），然后重采样到 50 Hz。

**起始姿态。** 跟踪器要求机器人从接近动作第一帧的姿态开始。在第一次 `step` 之前，从实测姿态插值到
该帧（RoboJuDo：policy 的 `get_init_dof_pos()`；行为：`BehaviorSpec.start_pos`），然后调用
`ctrl.reset()`（`start_behavior` 会替你 reset）。

除 H2H（`action_beta=0.8`）外，这些 policy 都不使用动作 EMA。

## 关节顺序

**23 自由度**（`asap_tracker`、`twist_tracker`、`kungfubot_tracker`）：腿（12 个），
`waist_yaw, waist_roll, waist_pitch`，然后是肩 pitch / roll / yaw 和肘，先左后右。6 个腕关节不驱动。

**21 自由度**（H2H 的 `protomotions_tracker`）：腿（12 个），`waist_yaw`，然后是 8 个手臂关节。

**29 自由度**（`beyondmimic_tracker`）：IsaacLab 顺序，见[运动控制指南](locomotion.zh-CN.md#关节顺序)。
以 ONNX 文件的元数据（`joint_names`、`default_joint_pos`、`joint_stiffness`、`joint_damping`、
`action_scale`）为准：

```python
import onnxruntime as ort
sess = ort.InferenceSession("Jump_wose.onnx", providers=["CPUExecutionProvider"])
meta = sess.get_modelmeta().custom_metadata_map
floats = lambda k: [float(x) for x in meta[k].split(",")]
joint_names = meta["joint_names"].split(",")
default_pos, kp, kd = floats("default_joint_pos"), floats("joint_stiffness"), floats("joint_damping")
action_scale = floats("action_scale")
print(meta["observation_names"])   # 例如 command,motion_anchor_ori_b,base_ang_vel,...
```

**29 自由度 SDK 顺序**（`supermimic`、`blinkmimic`）：`hiwbc.controllers.supermimic` 中的
`SUPERMIMIC_JOINT_NAMES`，与 Unitree SDK / MuJoCo 顺序相同。

## `asap_tracker`

ASAP checkpoint 已把动作训练在里面。控制器只维护一个相位 `t / (50 * motion_length_s)`，所以
`frames` 保持为空。

```python
import numpy as np
import onnxruntime as ort
from hiwbc import Reference, ReferenceKind, make_controller
from hiwbc.controllers.asap import AsapObsScales

G1_23_DEFAULT = [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0] * 2 + [0.0] * 11

sess = ort.InferenceSession("asap/mimic/CR7_level1/model_191500.onnx",
                            providers=["CPUExecutionProvider"])

def forward_fn(obs):
    return sess.run(["action"], {"actor_obs": obs[None]})[0].squeeze()

ctrl = make_controller(
    "asap_tracker",
    num_actions=23,
    default_pos=G1_23_DEFAULT,
    forward_fn=forward_fn,
    motion_length_s=3.967,          # CR7_level1
    use_history=True,
    history_length=4,
    obs_scales=AsapObsScales(base_ang_vel=0.25, dof_vel=0.05),
    action_scale=0.25,
    action_clip=100.0,
)
ctrl.reset()

ref = Reference(ReferenceKind.MOTION_FRAMES)
while not ctrl.motion_done:
    target = ctrl.step(read_state(), ref)
    send(target)
```

观测共 380 个值：上一步动作、角速度、关节位置、关节速度、4 个 tick 的历史、重力、相位。历史是
**按项排列（term-major）、最新的在前**：先是全部 4 个过去的上一步动作，然后是全部 4 个过去的角速度，
依此类推。

使用 profile 时，动作是一个行为（按 variant 分别为 `cr7_level1`、`horse_squat` 或 `dance_0605`），
它接受 `(0, 2]` 范围内的 `play_speed`：

```python
ctrl = hiwbc.load("asap_tracker/unitree_g1", variant="default")   # CR7_level1
(spec,) = ctrl.behaviors
ramp_to(spec.start_pos)
source = ctrl.start_behavior(spec.name, play_speed=1.0)
while not source.done:
    state = read_state()
    send(ctrl.step(state, source.next(state)))
```

相位每个 tick 前进 `reference.extra.get("play_speed", 1.0)`；传 `extra={"play_speed": 0.0}` 可以暂停。
相位到达 1 后停止，`ctrl.motion_done` 变为 `True`；这时切换到一个站立 policy。

| Checkpoint | `motion_length_s` | 上半身起始姿态（腰、手臂） |
|---|---|---|
| `CR7_level1` | 3.967 | `0.200, 0.077, -0.288, 0.217, 0.156, -0.517, 0.578, 0.257, -0.250, 0.225, 0.513` |
| `kongfu/horse_squat`（KungfuBot v1） | 6.67 | `-0.0005, 0.0007, 0.0029, 0.206, 0.196, 0.005, 0.890, 0.208, -0.197, -0.002, 0.888` |
| `robomimic/dance_0605` | 18.0 | `0, 0, 0, 0.35, 0.18, 0, 0.87, 0.35, -0.18, 0, 0.87` |

机器人必须从动作的第一个姿态开始：腿处于默认姿态，上半身处于表中所列的姿态。所有关节都从默认姿态开始
会让 CR7 摔倒。KungfuBot v1 在其 `default_pos` 中使用默认手臂姿态 `0.2, 0.2, 0.0, 0.9`（左）和
`0.2, -0.2, 0.0, 0.9`（右）。CR7 配置的增益：

```python
KP = [100, 100, 100, 200, 20, 20] * 2 + [400] * 3 + [90, 60, 20, 60] * 2
KD = [2.5, 2.5, 2.5, 5.0, 0.2, 0.1] * 2 + [5.0] * 3 + [2.0, 1.0, 0.4, 1.0] * 2
```

## `twist_tracker`

TWIST 跟踪每个 tick 的 "mimic target"，它来自动作播放器或遥操作重定向器。

```python
import torch

TWIST_DEFAULT = ([-0.2, 0.0, 0.0, 0.4, -0.2, 0.0] * 2 + [0.0] * 3
                 + [0.0, 0.2, 0.0, 1.2, 0.0, -0.2, 0.0, 1.2])
ankle_mask = np.zeros(23); ankle_mask[[4, 5, 10, 11]] = 1.0

model = torch.jit.load("twist/twist_general_motion_tracker.pt", map_location="cpu")

@torch.no_grad()
def forward_fn(obs):
    return model(torch.from_numpy(obs)[None]).numpy().squeeze()

ctrl = make_controller(
    "twist_tracker",
    num_actions=23,
    default_pos=TWIST_DEFAULT,
    forward_fn=forward_fn,
    history_length=10,
    ang_vel_scale=0.25, dof_pos_scale=1.0, dof_vel_scale=0.05,
    zero_dof_vel_mask=ankle_mask,   # TWIST 在观测中把踝关节速度置零
    action_scale=0.5,
    action_clip=10.0,
)
```

`ctrl.clip_reference(player, state)` 从带 `root_pos` 和 `root_quat` 的 `joint` 片段构建
`mimic_target`（见[参考从哪里来](#参考从哪里来)）。在 Horse-stance 片段上与 RoboJuDo 的
`MotionTwistCtrl` 对比了 150 个 tick：根高度、roll / pitch / yaw 和关节完全一致；本体坐标系下的根速度
最多相差 0.014 m/s，因为 RoboJuDo 用高斯滤波（σ = 2 帧）平滑速度，而 player 用的是普通有限差分。
如果需要与 RoboJuDo 一致的速度，请传入已平滑速度的片段。

`mimic_target` 有 31 个值，由参考帧构建（RoboJuDo `MotionTwistCtrl`）：

```python
# root_rot：[x, y, z, w]；root_vel、root_ang_vel：世界坐标系；dof_pos：23 个关节
rpy = euler_xyz(root_rot)                               # roll, pitch, yaw
v_b = quat_rotate_inverse(root_rot, root_vel)
w_b = quat_rotate_inverse(root_rot, root_ang_vel)
dof_25 = np.insert(dof_pos, [19, 23], 0.0)              # 在 19 和 24 处插入置零的腕关节槽位
mimic = np.concatenate([[root_z], rpy, v_b, [w_b[2]], dof_25])   # 33 个值
mimic_target = np.delete(mimic, [27, 32])               # 去掉腕关节槽位 -> 31

ref = Reference(ReferenceKind.MOTION_FRAMES, frames={"mimic_target": mimic_target})
```

观测是 mimic target、当前本体感知，以及之前 10 个 `[mimic_target, proprio]` 块（最旧的在前）：
`11 * (31 + 74) = 1155` 个值。

```python
KP = [100, 100, 100, 150, 40, 40] * 2 + [150] * 3 + [40] * 8
KD = [2, 2, 2, 4, 2, 2] * 2 + [4] * 3 + [5] * 8
```

## `beyondmimic_tracker`

每个发布的 checkpoint 只在一个动作上训练，它的 ONNX 文件对给定的 `time_step` 输出该动作的参考
（`joint_pos`、`joint_vel`、刚体位姿）。profile 把它作为一个行为提供：

| Variant | 行为 | 帧数（50 Hz） | 状态估计器 |
|---|---|---|---|
| `dance_wose`（默认） | `dance` | 6574（131.5 s） | 否 |
| `jump_wose` | `jump` | 165（3.3 s） | 否 |
| `violin` | `violin` | 612（12.2 s） | 是：需要 `state.base_lin_vel` |
| `waltz` | `waltz` | 945（18.9 s） | 是：需要 `state.base_lin_vel` |

帧数是实测的：从某个 `time_step` 起所有参考输出都不再变化，取这个 `time_step` 再加一。RoboJuDo 自己的
配置把 Violin 限制在 500 个 tick，Waltz 限制在 850 个 tick。

```python
import hiwbc

ctrl = hiwbc.load("beyondmimic_tracker/unitree_g1", variant="jump_wose",
                  checkpoint_dir="RoboJuDo/assets/models/g1")
(spec,) = ctrl.behaviors                 # "jump"
ramp_to(spec.start_pos)                  # 动作的第一个姿态，IsaacLab 顺序
source = ctrl.start_behavior("jump")     # reset + RoboJuDo 的预热前向
while not source.done:
    state = read_state()                 # 29 个关节；violin / waltz 还需要 base_lin_vel
    send(ctrl.step(state, source.next(state)))
```

这遵循 RoboJuDo 的 `use_motion_from_model` 模式，每个 tick 做一次前向：观测读取上一次前向返回的参考，
前向在当前帧运行，参考 anchor 的朝向在第一个 tick 转到机器人的朝向。机器人躯干姿态取你传入的
`state.extra["torso_quat"]`，否则用基座姿态经三个腰关节旋转得到。anchor 位置项为零，与 RoboJuDo
`override_robot_anchor_pos` 默认的效果相同。

在 RoboJuDo 的 MuJoCo 场景中检查过：由 hiWBC 驱动物理仿真，同时让 RoboJuDo 计算同一个 tick。四个
动作都跑到结束再加 60 个 tick（Dance：1000 个 tick），没有摔倒，观测最大差异 3.3e-5（Waltz；其他
在 1e-5 及以下），PD 目标最大差异 1.5e-5。剩下的差异来自躯干姿态：RoboJuDo 取自 MuJoCo 的 FK，
hiWBC 由腰关节计算。

要输入你自己的参考，就直接构建控制器：

```python
sess = ort.InferenceSession("beyondmimic/Jump_wose.onnx", providers=["CPUExecutionProvider"])
meta = sess.get_modelmeta().custom_metadata_map   # 关节顺序、默认值、增益（见上文）
clock = {"t": 0}

def forward_fn(obs):
    return sess.run(["actions"], {"obs": obs[None],
                                  "time_step": np.array([[clock["t"]]], np.float32)})[0].squeeze()

ctrl = make_controller(
    "beyondmimic_tracker",
    num_actions=29,
    default_pos=default_pos,                 # 来自 meta，IsaacLab 顺序
    forward_fn=forward_fn,
    without_state_estimator=True,            # *_wose checkpoint；Violin / Waltz 用 False
    action_scale=np.asarray(action_scale),   # 每个关节一个值，来自 meta
)
ctrl.reset()
```

每个 checkpoint 只在一个动作上训练，所以换一个参考很可能会让它摔倒。每个 tick 的参考项（RoboJuDo
`BeyondMimicPolicy._get_command`）：

| 项 | 如何获得 |
|---|---|
| `command`（58） | 参考在 `time_step` 处的 `concat(joint_pos, joint_vel)`。从模型的 `joint_pos` / `joint_vel` 输出读取 |
| `motion_anchor_ori_b`（6） | 参考 anchor（`torso_link`）相对于机器人躯干姿态的姿态，取旋转矩阵的前两列：`R(q_robot_torso⁻¹ * q_ref_anchor)[:, :2].reshape(-1)`。先把参考的 yaw 对齐到机器人的起始朝向 |
| `motion_anchor_pos_b`（3） | 参考 anchor 在机器人躯干坐标系中的位置。仅当 `without_state_estimator=False` 时需要 |
| `base_lin_vel`（3） | 来自状态估计器的本体坐标系线速度。仅当 `without_state_estimator=False` 时需要 |

```python
q_rel = quat_mul(quat_conjugate(robot_torso_quat), ref_anchor_quat)   # [x, y, z, w]
anchor_ori_b = matrix_from_quat(q_rel)[:, :2].reshape(-1)
ref = Reference(ReferenceKind.MOTION_FRAMES,
                frames={"command": command, "motion_anchor_ori_b": anchor_ori_b})
target = ctrl.step(state, ref)
clock["t"] += 1
```

`quat_mul`、`quat_conjugate` 和 `matrix_from_quat` 在 `hiwbc.math_utils` 中。

**预热。** RoboJuDo 在 reset 时对全零观测运行一次前向，这会留下一个非零的“上一步动作”，并启动模型的
动作输出。`start_behavior` 会做这一步。使用你自己的参考时，要与之完全一致，就在 `ctrl.reset()` 之后
自己运行这次前向，并把动作复制到 `ctrl._last_action`；不这样做，最初几个 tick 会略有差异。

观测大小：`*_wose`（无状态估计器）为 154，其他为 160。Waltz checkpoint 与 RoboJuDo 有 3.7e-4 的偏差，
在 1000 个 tick 内没有增大；其他的在 5e-6 以内一致。

## `kungfubot_tracker`

KungfuBot2 有三个输入，所以 `observe` 返回一个 tuple。

```python
sess = ort.InferenceSession("kungfubot2/horse_test_43000.onnx",
                            providers=["CPUExecutionProvider"])

def forward_fn(obs):
    actor_obs, future, history = obs
    return sess.run(["action"], {"actor_obs": actor_obs[None],
                                 "future_motion_targets": future[None],
                                 "prop_history": history[None]})[0].squeeze()

ctrl = make_controller(
    "kungfubot_tracker",
    num_actions=23,
    default_pos=kungfubot_default,           # RoboJuDo g1_kungfubot_policy_cfg.py
    forward_fn=forward_fn,
    history_length=10,
    ang_vel_scale=0.25, dof_pos_scale=1.0, dof_vel_scale=0.05,
    roll_pitch_scale=1.0, actions_scale=1.0, history_scale=1.0,
    action_scale=np.asarray(kungfubot_action_scales),   # 每个关节一个值，同一文件
    action_clip=100.0,
)
```

输入大小：`actor_obs` 877，`future_motion_targets` 600，`prop_history` 740。

参考项来自未来的参考帧（RoboJuDo `MotionKungfuBotCtrl`）：`future_motion_root_height`、
`future_motion_roll_pitch`、`future_motion_base_lin_vel` 和 `future_motion_base_yaw_vel`（两者都在参考
根坐标系中）、`future_motion_dof_pos`、`next_step_ref_motion`，以及 `anchor_ref_rot`，即参考根姿态相对
机器人基座的姿态，取旋转矩阵的两列（6 个值）。每一项乘以它在 `obs_scales` 中的系数
（`horse_test_43000` 全部为 1.0）。

```python
q_rel = quat_mul(quat_conjugate(state.base_quat), ref_root_quat)    # [x, y, z, w]
q_rel = q_rel[[1, 2, 3, 0]]   # 仅 horse_test_43000：训练时用了这个重排
anchor_ref_rot = matrix_from_quat(q_rel)[:, :2].reshape(-1)
```

这个重排复现了 PBHC 训练代码中的一个 bug（[PBHC#68](https://github.com/TeleHuman/PBHC/issues/68)），
`horse_test_43000` 就是带着它训练的；RoboJuDo 用 `compatibility_old_version=True` 开启它。对修复之后
训练的 checkpoint，去掉这一行。

从片段出发时，`clip_reference` 构建全部七项。未来窗口是第 `1, 5, 10, ..., 95` 帧
（`linspace(1, 95, 20)`，截断取整）。`next_step_ref_motion` 中的关键点是膝、踝和肘，加上从肘和躯干偏移
得到的手部和头部点（PHC `unitree_g1_23dof_KungfuBot.yaml`）。在 RoboJuDo 的 horse-stance 片段上与其
`MotionKungfuBotCtrl` 对比，最大差异为 6.7e-3（关节角，在片段末尾，RoboJuDo 的插值在那里的截断方式
不同），其余各项都低于 1.1e-3。hiWBC 的窗口从当前帧开始；RoboJuDo 的窗口滞后一帧，因为它在推进之前
就计算了第 t 个 tick 的窗口。

历史是按项排列、最新的在前：先是全部 10 个过去的动作，然后是全部 10 个角速度，依此类推。在同步运行的
对齐检查中，这个 checkpoint 在第 404 个 tick 摔倒；单独在 RoboJuDo 下运行时在第 405 个 tick 摔倒。

## `protomotions_tracker`（H2H）

```python
model = torch.jit.load("h2h/h2h.pt", map_location="cpu")

ctrl = make_controller(
    "protomotions_tracker",
    num_actions=21,
    default_pos=[-0.1, 0.0, 0.0, 0.3, -0.2, 0.0] * 2 + [0.0] * 9,
    forward_fn=forward_fn,                   # 封装 h2h.pt
    history_length=25,
    use_dof_pos_offset=False,                # H2H 输入的是绝对关节角
    action_scale=0.25,
    action_clip=10.0,
    action_beta=0.8,
)
```

`task_obs` 是遥操作模仿观测：参考关键刚体相对机器人关键刚体的位置（以及速度），表示在机器人的朝向
坐标系中。它需要对机器人的实测姿态做 FK。RoboJuDo 用 `compute_imitation_observations_teleop_max_np`
（`robojudo/policy/utils/utils_h2h.py`）计算它，并把头部目标限制在离机器人水平距离 1 m 以内。

从片段出发时，`clip_reference` 为双手和头部构建 `task_obs`（27 个值）（PHC `unitree_g1_29dof.yaml`
extend_config）。它需要 `state.extra["base_pos"]` 中的机器人世界坐标位置，并且片段的起点放在机器人所在的位置。
机器人自身的点由 `state.dof_pos` 做 FK 得到（29 自由度模型，腕关节为 0）。片段必须带有 29 自由度的关节角。
与 RoboJuDo 对比：参考点 1.5e-5，速度 1.3e-5（默认片段，播放速度 1），200 个随机机器人姿态下
`task_obs` 2.3e-7。RoboJuDo 在淡入之后以半速播放 H2H 片段（`speed_target = 1/2`）；hiWBC 以 1 倍速播放。

```python
KP = [100, 100, 100, 150, 40, 40] * 2 + [200] + [40] * 8
KD = [5, 5, 5, 5, 2, 2] * 2 + [6] + [2] * 8
```

## `supermimic`

```python
from hiwbc.controllers.supermimic import SuperMimicController, goal_features

ctrl = SuperMimicController.from_bundle("supermimic_deploy/assets")   # config.json + policy.onnx
ctrl.reset()

goal = goal_features(joint_pos, joint_vel, root_quat, root_lin_vel_w, root_ang_vel_w,
                     root_z, default_pos=ctrl.default_pos)            # (T, 68)，整个片段
for t in range(len(goal)):
    ref = Reference(ReferenceKind.MOTION_FRAMES, frames={"goal": goal[t:t + 10]})
    target = ctrl.step(read_state(), ref)    # target.kp / kd 取自 bundle
    send(target)
```

bundle 就是 [supermimic_deploy](https://github.com/KyrieYoung926/supermimic_deploy) 仓库里的 `assets/` 目录，
`policy.onnx` 按它的 README（第 2 节）下载。hiWBC 不附带 SuperMimic 权重。supermimic_deploy 采用 MIT 许可，它的
policy 用 [SuperMimic-Reproduced](https://github.com/CuboiLeo/SuperMimic-Reproduced)（MIT）训练；权重的使用条件以
这两个项目为准。

也可以用 `hiwbc.load("supermimic/unitree_g1", checkpoint_dir="supermimic_deploy/assets")`；配合
`joint` 片段时，`ctrl.clip_reference(player, state)` 用 `goal_features` 构建 10 个 goal 帧。

`from_bundle` 会对照 `config.json` 检查关节顺序、历史长度和 ONNX 文件的 sha256。`goal_features`
接受 50 Hz 的参考片段（`root_quat` 为 `[x, y, z, w]`，速度在世界坐标系中），每帧返回一个 68 维 goal；
每个 tick 传入第 `t .. t+9` 帧，接近片段末尾时控制器会重复最后一帧。设 `bound_torque=True`（默认）时，
目标会被重新表示，使机器人的 PD 力矩保持在 bundle 的力矩上限之内。

## `blinkmimic`

[BlinkMimic](https://github.com/fuyichen0725/blinkmimic)（PyTorch、mjlab）通过一个学到的世界模型训练 G1
跟踪 policy，这是 SuperMimic 提出的方法。它的观测和 goal 与 `supermimic` 不同：BlinkMimic 的 policy 不能在
`supermimic` 上跑，反过来也不行。同一个控制器也能跑 XMimic 的 policy，二者接口相同（见 [XMimic](#xmimic)）。

```python
import hiwbc
from hiwbc.references import ClipPlayer

ctrl = hiwbc.load("blinkmimic/unitree_g1",
                  checkpoint="model_2700.pt")                # 你训出的 policy
player = ClipPlayer(clip, ctrl.control_dt, **ctrl.clip_player_options)   # space 为 "joint"
ctrl.check_clip(player)                      # 需要 root_pos、root_quat 和关节
ctrl.reset()
while not player.done:
    state = read_state()
    send(ctrl.step(state, ctrl.clip_reference(player, state)))   # target.kp / kd 已设置
    player.advance()
```

通过 `wbc` layer，同一个 profile 播放以 `MOTION_CLIP` 发送的 hiRobot `MotionClip`
（[机器人指南](robot.zh-CN.md#3-命令)）：

```yaml
# g1_blinkmimic.yaml
defaults: [robots/unitree_g1_mujoco_standing]
layers:
  - type: wbc
    profile: blinkmimic/unitree_g1   # checkpoint 默认为 ~/.cache/hiwbc/blinkmimic/policy.pt
```

```python
from hirobot import Action, ControlMode, make_robot

robot = make_robot("g1_blinkmimic.yaml")
robot.connect()
play = Action(mode=ControlMode.MOTION_CLIP, motion=clip)
robot.send_action(play)
while not robot.get_status().command.done:
    robot.send_action(play)   # 同一个对象：片段接着播放
robot.hold()
```

和所有跟踪器一样，机器人应从接近片段第一帧的姿态开始（见上文的**起始姿态**）。

| Variant | Checkpoint（`checkpoint` 默认值） | 需要 | 说明 |
|---|---|---|---|
| `default` | BlinkMimic `.pt`（`~/.cache/hiwbc/blinkmimic/policy.pt`） | `torch` extra | torch 只用来读文件；网络在 numpy 中以 fp32 运行 |
| `xmimic` | XMimic `ckpt_*.npz`（`~/.cache/hiwbc/xmimic/policy.npz`） | numpy | 见 [XMimic](#xmimic) |
| `onnx` | BlinkMimic `tools/export_onnx.py` 导出的文件，训练进一个动作（`~/.cache/hiwbc/blinkmimic/dance1_subject2.onnx`；传入你的导出文件路径） | `onnx` extra | 把这个动作作为行为播放（`ctrl.behaviors`、`start_behavior`）；不能跟踪其他片段 |

Profile 选项：`checkpoint`（路径）、`precision`（仅 `xmimic`，默认 `"bf16"`）、`providers`（仅 `onnx`，
onnxruntime 的 execution provider，默认 CPU）、`bound_torque`（默认 `True`：重新表示目标，使 PD 力矩不超过每个
关节的力矩上限，与训练时的执行器截断一致）。默认姿态、action scale、增益和力矩上限取自 BlinkMimic 的 G1 执行器
模型（`hiwbc.controllers.blinkmimic` 中的 `G1_*`）。

**获取 checkpoint。** BlinkMimic 不发布训练好的权重，hiWBC 也不附带。用上游仓库训一个，把路径传给
`checkpoint=`：BlinkMimic 在一块 RTX 4090 上约 3 分钟达到训练目标（据其 README）。

**布局**（G1，29 个关节，SDK 顺序，50 Hz）。模型有两个输入：

- `obs_history`（990）：10 个 tick，按帧排列（frame-major），最旧的在前，包括当前 tick；`reset()` 之后
  10 个槽位都填当前条目。每个 tick 99 个值：`q - q0`（29）、`dq`（29）、机体坐标系下的基座角速度（3）、
  投影重力（3）、上一个动作（29），以及参考根姿态相对机器人根的姿态（6：`R^T R_ref` 的前两列）。
- `goal`（730）：接下来的 10 个参考帧 `f+1 .. f+10`，到片段末尾时重复最后一帧。每帧 73 个值：当前参考帧
  朝向坐标系下的根位移（3）和旋转（6D，6），该未来帧自身根坐标系下的根线速度和角速度（3 + 3），关节角
  （29），关节速度 × 0.1（29）。

`frames` 带 `goal` `(10, 73)` 和 `ref_root_quat` `(4,)`，即当前参考根姿态，`[x, y, z, w]`。
`clip_reference` 构建这两项；goal 中的速度是重采样到 50 Hz 的片段上的中心差分，与训练器的算法一致。
模型在输出上加上下一帧的参考关节角（一个存在 checkpoint 里的残差），目标为
`q0 + action_scale * action`，无 EMA，无裁剪。增益按训练器的公式：`kp = armature * (2π · 10 Hz)²`，
`kd = 2 · 2.0 · armature · (2π · 10 Hz)`（阻尼比 2.0），`action_scale = 0.25 · effort / kp`。
XMimic 的布局相同。

`obs_history` 中的姿态项比较机器人和参考在世界坐标系中的姿态，所以片段的朝向必须和机器人一致。片段
播放器在第一个 tick 把片段转到机器人的朝向。在硬件上，IMU 的 yaw 漂移会直接进入这一项。

**`onnx` variant。** BlinkMimic 的 `tools/export_onnx.py` 把一个动作的 goal 固化进计算图，所以导出文件只播放
这个动作，不能跟踪其他片段：

```python
ctrl = hiwbc.load("blinkmimic/unitree_g1", variant="onnx",
                  checkpoint="dance1_subject2.onnx")   # 你的导出文件
(spec,) = ctrl.behaviors                 # 以元数据的 motion 字段命名
ramp_to(spec.start_pos)                  # 动作的第一个姿态，SDK 顺序
source = ctrl.start_behavior(spec.name)  # 无参数；会重置控制器
while not source.done:
    state = read_state()
    send(ctrl.step(state, source.next(state)))
```

- 行为以导出文件元数据的 `motion` 字段命名，没有该字段时用文件名（不含扩展名）。它不接收参数，持续
  `(frames - 1) × 0.02` 秒。
- 播放方式与上游 `tools/play_onnx.py` 一致：每个 tick 一帧，上一个动作从零开始。第一个 tick 时，动作只绕
  yaw 转到机器人的朝向。
- 经 `wbc` layer 时，`robot.run_layer_skill(spec.name)` 启动它，`WbcRobot` 先把机器人缓动到动作的第一个
  姿态（[机器人指南](robot.zh-CN.md#3-命令)）。
- 如果导出文件元数据中的关节名、增益、默认姿态或 action scale 与 BlinkMimic 的 G1 值不同（相对容差 1e-5），
  或者历史长度、goal 帧数、控制周期或观测名与本控制器不同，加载会失败。

**在硬件上。** `blinkmimic` 目前只在仿真中运行过，BlinkMimic 和 XMimic 的 policy 都是。上真机前注意两点：

- 网络在 numpy 中运行。在核数多的机器上，BLAS 线程会让单步耗时出现尖峰；给控制进程设置
  `OMP_NUM_THREADS=1`。
- policy 训练时用的 BlinkMimic G1 执行器模型给踝关节和腰部 roll / pitch 的力矩上限是 50 N·m。hiRobot 的 G1
  body 配置中这些关节是 35 N·m，所以 `bound_torque` 对这些关节截断的力矩高于 body 配置允许的值。

和 BlinkMimic 对拍过，用的是在 LAFAN1 G1 上训的一个 checkpoint：

- **policy 输入**：输入相同时，观测项误差在 3e-7 以内，goal 在 1e-5 以内。hiWBC 自己从 30 fps CSV 构造 goal
  时，长片段的后段最多差 1e-3。原因是 BlinkMimic 在 float32 时间网格上重采样，hiWBC 用的是 float64。
- **动作**：`.pt` runner 与 BlinkMimic 的 fp32 policy 误差在 2.4e-6 以内。`onnx` variant 回放 BlinkMimic
  录下的 6573 拍导出文件 rollout，误差在 4.4e-6 以内。
- **闭环（训练场景）**：在 BlinkMimic 的 MuJoCo 训练场景里，走路、跳舞、摔倒再起身三个片段都没有摔倒，平均关节误差
  与 BlinkMimic 自己的评估相差不到 6e-4 rad。
- **闭环（hiRobot）**：经 `WbcRobot` 在 hiRobot 的自由基座 G1 上，policy 能跑完 20 s 走路和 30 s 跳舞而不摔倒；
  从默认姿态起步或从片段第一帧起步都一样。
- **片段结束后**：和其他 tracker 一样，机器人在片段结束后大约一秒内摔倒（见"常见的坑"）。

BlinkMimic 采用 MIT 许可。它训出的 checkpoint 来自 LAFAN1（CC BY-NC-ND 4.0），所以训好的 policy 只能用于
非商业用途。

## XMimic

[XMimic](https://github.com/nico-bohlinger/xmimic) 是 BlinkMimic 的 JAX 重写（论文：Bohlinger 和 Peters，
"XMimic and XSoma: Learning Humanoid Motion Tracking and Locomotion in Under a Minute"，2026，
[arXiv:2611.20452](https://arxiv.org/abs/2611.20452)）。它的 policy 接口和 BlinkMimic 相同，所以
[`blinkmimic`](#blinkmimic) 控制器通过 `xmimic` variant 运行它；片段、布局、增益和硬件注意事项都一样。

```python
ctrl = hiwbc.load("blinkmimic/unitree_g1", variant="xmimic",
                  checkpoint="ckpt_5800.npz")      # 你训出的 policy
```

```yaml
# g1_xmimic.yaml
defaults: [robots/unitree_g1_mujoco_standing]
layers:
  - type: wbc
    profile: blinkmimic/unitree_g1
    variant: xmimic          # checkpoint 默认为 ~/.cache/hiwbc/xmimic/policy.npz
```

- **Checkpoint。** XMimic 的 `ckpt_*.npz`。XMimic 不发布权重；据其 README，一块 H100 上训练 2.5 分钟，四块上
  0.86 分钟。文件用 numpy 读，不需要 extra。
- **精度。** `precision="bf16"`（默认）像 XMimic 的评估一样把 matmul 输入舍入到 bf16；`"fp32"` 不舍入。
  `self_check` 用 `.npz` 里存的测试动作检查 runner（[API](../api.zh-CN.md#xmimic)）。
- **对拍。** fp32 下 runner 与 XMimic 的 policy 误差在 6.4e-6 以内。bf16 路径没法逐拍对上：同一拍单独算和放进
  batch 里算，XMimic 自己的 bf16 输出就最多差 0.023。checkpoint 里存的测试动作，hiWBC 复现到 1e-5。在 XMimic
  的训练场景里，走路、跳舞、摔倒再起身三个片段都没有摔倒，平均关节误差与 XMimic 自己的 rollout 相差不到
  3e-4 rad；经 `WbcRobot` 能跑完 20 s 走路和 30 s 跳舞而不摔倒。

XMimic 采用 MIT 许可；在 LAFAN1 上训出的 policy 只能用于非商业用途。

## 常见的坑

- **历史顺序。** ASAP 和 KungfuBot2 的历史按项排列、最新的在前。按帧排列（frame-major）的历史也能
  对上输入大小，但 policy 会在约 1 s 内摔倒。
- **起始姿态。** 跟踪器从默认姿态而不是动作第一帧开始，可能在第一秒内摔倒。
- **片段结束后。** 跟踪器没有站立参考，片段或行为结束后 `WbcRobot` 用纯 PD 保持 `default_pos`，这样不会
  保持平衡，自由基座的机器人大约一秒内就会摔倒。要在当前片段结束前发下一个片段。
- **动作时钟漂移。** 使用你自己的 BeyondMimic 参考时，`time_step` 和参考必须对应同一帧；每个 tick
  两者各推进一次。行为会替你做这件事。
- **速度平滑。** 默认情况下 `ClipPlayer` 以 50 Hz 差分求速度，不做平滑。
  `ClipPlayer(..., source_velocity_sigma=2.0)` 以片段自身的帧率差分并做平滑，与 PHC 相同；KungfuBot
  的 profile 通过 `ctrl.clip_player_options` 开启它（在 horse-stance 片段上，50 Hz 差分的结果偏差
  0.16 m/s）。TWIST 和 SuperMimic 使用默认值；它们的 profile 在一次 0.014 m/s 的检查后没有开启它。
- **Anchor 坐标系。** `motion_anchor_ori_b` 和 `anchor_ref_rot` 相对于机器人实测的躯干 / 基座姿态，
  每个 tick 重新计算，而不是相对固定的世界坐标系。

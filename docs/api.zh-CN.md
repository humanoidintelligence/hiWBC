# hiWBC API 参考

[English](./api.md) | 简体中文

每个已发布 policy 的分步设置（以及它训练时用的取值）见[指南](guides/README.zh-CN.md)。

本页列出 hiWBC 开放的接口：接口约定类型、`Controller` 基类、控制器与 profile 注册表、动作片段回放、
行为（behavior），以及每个内置控制器的构造参数和观测布局。[顶层包](#顶层包)下列出的名字都可以用
`from hiwbc import ...` 导入。

hiWBC 计算 `(ControllerState, Reference) -> JointTarget`。控制器不做任何硬件 I/O：调用方读取机器人，
每个控制周期调用一次 `Controller.step`，再把返回的目标发出去。调用方要么是你自己的循环，要么是
[`WbcRobot`](#wbcrobothirobot-layer)，即 hiWBC 注册的 hiRobot layer。

全文约定：

- 数组是 `numpy.ndarray`，在控制器内部转成 `float32`。角度单位为 rad，角速度为 rad/s。
- 四元数为 `[x, y, z, w]`。
- 关节向量（`dof_pos`、`default_pos`、`JointTarget.position`、增益）按控制器的关节顺序排列，也就是
  其 policy 训练时的顺序。按关节名把机器人的关节布局映射到这个顺序，是调用方的事。
- 控制器从不加载模型。每个控制器都接收一个注入的 `forward_fn(obs) -> action`，所以可以用桩函数代替
  checkpoint。[profile](#profile) 负责加载 checkpoint，并用训练时的取值构建控制器。

## 目录

- [顶层包](#顶层包)
- [快速示例](#快速示例)
- [接口约定类型](#接口约定类型)
- [Controller](#controller)
- [注册表](#注册表)
- [Profile](#profile)
- [动作片段](#动作片段)
- [行为](#行为)
- [前向函数](#前向函数)
- [远程推理](#远程推理)
- [WbcRobot：hiRobot layer](#wbcrobothirobot-layer)
- [关节映射](#关节映射)
- [速度命令控制器](#速度命令控制器)
- [动作跟踪控制器](#动作跟踪控制器)
- [SuperMimic](#supermimic)
- [BlinkMimic](#blinkmimic)
- [XMimic](#xmimic)
- [SONIC](#sonic)
- [echo](#echo)
- [数学工具](#数学工具)
- [编写控制器](#编写控制器)

## 顶层包

| 名字 | 类别 | 模块 | 用途 |
|---|---|---|---|
| `ControllerState` | dataclass | `hiwbc.types` | 每个 tick 读取的本体感知 |
| `Reference`, `ReferenceKind` | dataclass / 枚举 | `hiwbc.types` | 本 tick 的命令及其解读方式 |
| `JointTarget` | dataclass | `hiwbc.types` | PD 位置目标，可选每个关节的增益 |
| `BehaviorSpec`, `ReferenceSource` | dataclass / protocol | `hiwbc.types` | 控制器自带的一个行为，以及它返回的逐 tick 参考源 |
| `Controller` | 抽象类 | `hiwbc.controller` | `observe` → `forward` → 共用的后处理 |
| `register_controller`, `make_controller`, `available_controllers` | 函数 | `hiwbc.registry` | 按名字注册的控制器注册表 |
| `load`, `profile_info`, `get_profile`, `available_profiles`, `register_profile`, `Profile` | 函数 / dataclass | `hiwbc.registry` | Profile：某台机器人上带训练取值和 checkpoint 的控制器 |
| `load_onnx_runner`, `load_torchscript_runner` | 函数 | `hiwbc.runtime` | 从 checkpoint 文件构建 `forward_fn` |
| `WbcRobot` | 类 | `hiwbc.robot` | hiRobot layer（`type: wbc`）；它导入 `hirobot`，所以不从顶层重新导出 |
| `make_remote`, `RemoteFn`, `ForwardTimeout`, `RemoteError` | 函数 / 类 | `hiwbc.remote` | 在 `hiwbc.serve` 上运行控制器的模型前向 |
| `ModelServer` | 类 | `hiwbc.serve` | 前向服务（`python -m hiwbc.serve`） |
| `robot_configs` | 函数 | `hiwbc.examples` | hiWBC 自带的机器人配置，名字 -> 路径（[WbcRobot](#wbcrobothirobot-layer)） |
| `__version__` | str | `hiwbc` | 包版本 |

导入 `hiwbc` 会导入 `hiwbc.controllers`，从而注册内置控制器。控制器类在各自的模块里：

| 模块 | 类和辅助函数 |
|---|---|
| `hiwbc.controllers.echo` | `EchoController` |
| `hiwbc.controllers.unitree` | `UnitreeLocoController`, `ObsScales` |
| `hiwbc.controllers.smooth` | `SmoothLocoController`, `SmoothObsScales` |
| `hiwbc.controllers.asap` | `AsapTrackerController`, `AsapObsScales` |
| `hiwbc.controllers.amo` | `AmoLocoController`, `load_amo_runners` |
| `hiwbc.controllers.twist` | `TwistTrackerController` |
| `hiwbc.controllers.beyondmimic` | `BeyondMimicTrackerController` |
| `hiwbc.controllers.protomotions` | `ProtoMotionsTrackerController` |
| `hiwbc.controllers.kungfubot` | `KungfuBotTrackerController` |
| `hiwbc.controllers.supermimic` | `SuperMimicController`, `goal_features`, `load_onnx_runner`, `SUPERMIMIC_JOINT_NAMES` |
| `hiwbc.controllers.blinkmimic` | `BlinkMimicController`, `goal_features`, `load_npz_runner`, `load_pt_runner`, `self_check`, `load_onnx_motion`, `BLINKMIMIC_JOINT_NAMES`, G1 表（`G1_DEFAULT_POS`、`G1_ACTION_SCALE`、`G1_KP`、`G1_KD`、`G1_EFFORT`） |
| `hiwbc.controllers.sonic` | `SonicController`, `SonicRobotConfig`, `SonicObsConfig`, `ENCODER_MODE_IDS`, `load_onnx_runner`, `vr_3point`, `VR_3POINT_NAMES` |
| `hiwbc.controllers.sonic_planner` | `SonicPlanner`, `LocomotionMode`, `velocity_command`, `load_planner_runner` |
| `hiwbc.configs.blinkmimic.unitree_g1` | `check_onnx_metadata` |
| `hiwbc.configs.sonic.unitree_g1` | `SONIC_G1`, `SONIC_VARIANTS`，G1 表（`G1_JOINT_NAMES`、`G1_JOINT_NAMES_MUJOCO`、`G1_DEFAULT_POS`、`G1_ACTION_SCALE`、`G1_KP`、`G1_KD`……） |
| `hiwbc.configs.<controller>.<robot>` | 每个 profile 一个模块（见 [Profile](#profile)） |
| `hiwbc.references` | `ClipPlayer`, `require_clip` |
| `hiwbc.math_utils` | 四元数和投影重力的辅助函数 |

## 快速示例

```python
import numpy as np
from hiwbc import ControllerState, Reference, ReferenceKind, make_controller

n = 12
ctrl = make_controller(
    "unitree_loco",
    num_actions=n,
    default_pos=np.zeros(n),
    forward_fn=lambda obs: np.zeros(n, dtype=np.float32),  # 代替 checkpoint
)
ctrl.reset()
state = ControllerState(
    dof_pos=np.zeros(n), dof_vel=np.zeros(n),
    base_quat=np.array([0.0, 0.0, 0.0, 1.0]),
    base_ang_vel=np.zeros(3),
)
ref = Reference(kind=ReferenceKind.VELOCITY, value=np.array([0.5, 0.0, 0.0]))
target = ctrl.step(state, ref)     # JointTarget
target.position                    # (12,) 绝对 PD 位置目标
```

磁盘上有 RoboJuDo 的 G1 checkpoint 时，profile 会用训练取值和增益构建同一个控制器：

```python
import hiwbc

ctrl = hiwbc.load("unitree_loco/unitree_g1", checkpoint_dir="RoboJuDo/assets/models/g1")
ctrl.joint_names, ctrl.kp          # 控制器的关节顺序和增益
```

## 接口约定类型

`hiwbc.types`。

### `ReferenceKind`

`str` 枚举，指明控制器接收的是哪种命令。

| 成员 | 取值 | 载荷 |
|---|---|---|
| `VELOCITY` | `"velocity"` | `Reference.value = [vx, vy, wyaw]` |
| `MOTION_FRAMES` | `"motion_frames"` | `Reference.frames = {term: array}`，本 tick 的参考动作各项 |
| `LATENT` | `"latent"` | `Reference.value` = 不透明的运动 latent，例如 SONIC token |

### `ControllerState`

| 字段 | 形状 | 说明 |
|---|---|---|
| `dof_pos` | `(num_dofs,)` | 关节位置 |
| `dof_vel` | `(num_dofs,)` | 关节速度 |
| `base_quat` | `(4,)` | 基座姿态，`[x, y, z, w]`，世界坐标系 |
| `base_ang_vel` | `(3,)` | 基座角速度，本体坐标系 |
| `base_lin_vel` | `(3,)` or `None` | 基座线速度，本体坐标系。可选；只有带状态估计器的 checkpoint 的 BeyondMimic 行为会读它 |
| `extra` | `dict` | 自由格式。如果有 `extra["torso_quat"]`，`beyondmimic_tracker` 会读它 |

### `Reference`

| 字段 | 类型 | 说明 |
|---|---|---|
| `kind` | `ReferenceKind` | 决定如何读取其他字段 |
| `value` | array or `None` | `VELOCITY` 命令或 `LATENT` token |
| `frames` | `dict[str, array]` or `None` | `MOTION_FRAMES` 的载荷。键取决于控制器 |
| `extra` | `dict` | 逐 tick 的选项，例如 `play_speed`（ASAP）、`encoder_mode` / `play`（SONIC）、`time_step`（BeyondMimic 行为） |

### `JointTarget`

| 字段 | 形状 | 说明 |
|---|---|---|
| `position` | `(num_actions,)` | 绝对 PD 位置目标 |
| `kp`, `kd` | `(num_actions,)` or `None` | 每个关节的增益，即控制器的 `kp` / `kd`。`None` 表示保留机器人配置的增益。每个 profile 都会设置它们；用 `make_controller` 构建且不传 `kp` / `kd` 的控制器把它们留为 `None` |
| `extra` | `dict` | 控制器特有的输出。`sonic` 把它用的 token 放在 `extra["motion_token"]` |

## Controller

`hiwbc.controller.Controller`，一个抽象基类。

```python
Controller(*, num_actions, default_pos, action_scale=1.0, action_clip=None, action_beta=1.0,
           joint_names=None, obs_joint_names=None, kp=None, kd=None, control_dt=0.02,
           cartesian_targets=None, stand_command=None, velocity_limits=None)
```

| 参数 | 说明 |
|---|---|
| `num_actions` | 动作的长度，也是 `JointTarget.position` 的长度 |
| `default_pos` | `(num_actions,)` 默认姿态，加到缩放后的动作上 |
| `action_scale` | 正标量，或每个关节一个缩放值的 `(num_actions,)` 数组 |
| `action_clip` | `clip(action, -c, c)` 的正边界，`None` 表示不裁剪 |
| `action_beta` | EMA 权重，取值 `(0, 1]`。`1.0` 表示不平滑 |
| `joint_names` | 控制器驱动的关节，按控制器的顺序，使用 hiRobot 的写法（没有 `_joint` 后缀）。只做测试的裸构建中为 `None`；每个 profile 都会设置 |
| `obs_joint_names` | `ControllerState.dof_pos` 携带的关节，与 `joint_names` 不同时才需要（AMO 会读取它不驱动的手臂）。默认：`joint_names` |
| `kp`, `kd` | `(num_actions,)` 非负增益，随每个目标一起返回。`None`：机器人保留自己的增益 |
| `control_dt` | 控制周期，s |
| `cartesian_targets` | 末端执行器名 -> 控制器目标名，用于接收笛卡尔目标的控制器。只有带 planner 的 `sonic` 设置它（`left_hand` -> `left_wrist`，`right_hand` -> `right_wrist`） |
| `stand_command` | 对该 checkpoint 表示"站住"的 `VELOCITY` 命令（不一定是零） |
| `velocity_limits` | `[vx, vy, wz]` `VELOCITY` 命令每个轴测试过的 `(min, max)`，单位同命令。`None`：没给。`sonic`（带 planner）和 `unitree_loco` 的 `default` profile 设为 `((-0.8, 1.0), (-0.5, 0.5), (-1.0, 1.0))` |

以下情况构造函数抛出 `ValueError`：`default_pos`、`action_scale`、`kp` 或 `kd` 形状错误；`action_scale`、
`action_clip` 或 `control_dt` 不为正；`action_beta` 不在 `(0, 1]` 内；`velocity_limits` 不是三对满足 `min <= 0 <= max` 的 `(min, max)`；`joint_names` 长度不对或有重复；
给了 `obs_joint_names` 却没有 `joint_names`。

| 成员 | 说明 |
|---|---|
| `accepts` | 类属性：控制器处理的 `ReferenceKind` 组成的元组。为空表示都接受 |
| `observe(state, reference)` | 抽象方法。构建 policy 输入：一个扁平数组，多输入模型则是数组元组 |
| `forward(obs)` | 抽象方法。运行模型并返回原始动作 |
| `step(state, reference) -> JointTarget` | 一个控制周期（见下文） |
| `reset()` | 清除每个 episode 的状态。基类清除上一个动作，并对每个带 `reset()` 的注入 `*_fn` 调用它（循环模型的隐藏状态）；子类还会清除时钟和历史 |
| `motion_spaces` | 控制器能跟踪的 `MotionClip.space` 取值（`"joint"`、`"smpl"`）；为空表示不接收片段。见[动作片段](#动作片段) |
| `check_clip(player)` | 片段缺少控制器要读的内容时抛出 `ValueError` |
| `clip_reference(player, state) -> Reference` | 从 `ClipPlayer` 得到本 tick 的参考 |
| `stand_reference(state) -> Reference` | 让机器人保持站立的参考：速度控制器是 `stand_command`，`sonic` 是默认姿态（带 planner 时为零 `VELOCITY`）。其他跟踪控制器抛出 `NotImplementedError` |
| `cartesian_rest() -> dict` | 目标名 -> `[x, y, z, qx, qy, qz, qw]`，没有笛卡尔目标命令时控制器跟踪的位姿。默认为 `{}` |
| `behaviors` | 控制器自带的 `list[BehaviorSpec]`。见[行为](#行为) |
| `start_behavior(name, **args) -> ReferenceSource` | 启动一个行为；名字未知时抛出 `KeyError` |
| `profile`, `variant` | 由 `hiwbc.load` 设置：构建它的 profile 和 variant |

`step` 依次执行：

1. 如果 `accepts` 非空且 `reference.kind` 不在其中，抛出 `ValueError`。
2. `obs = observe(state, reference)`。
3. `raw = forward(obs)`，展平。值的个数不是 `num_actions` 时抛出 `ValueError`。
4. 后处理，按 RoboJuDo `get_action` 的顺序：
   1. `action = (1 - action_beta) * last_action + action_beta * raw`
   2. `last_action = action`（在裁剪和缩放之前；观测读取的"上一个动作"项就是它）
   3. 设置了 `action_clip` 时，`action = clip(action, -action_clip, action_clip)`
   4. `action = action * action_scale`
5. 返回 `JointTarget(position=action + default_pos, kp=kp, kd=kd)`。

每个 episode 开始时调用 `reset()`。带时钟（步态相位、动作相位）的控制器在每次 `step` 之后推进时钟，
而不是在 `observe` 内部。

## 注册表

`hiwbc.registry`。按名字管理控制器；为某台机器人构建控制器通常用 [profile](#profile)。

| 函数 | 说明 |
|---|---|
| `register_controller(name)` | 工厂函数 `(**kwargs) -> Controller` 的装饰器。`name` 已被占用时抛出 `ValueError` |
| `make_controller(name, **kwargs) -> Controller` | 用 `kwargs` 调用工厂函数。抛出的 `KeyError` 会列出可用的名字 |
| `available_controllers() -> list[str]` | 排好序的已注册名字 |

内置名字：`echo`、`unitree_loco`、`smooth_loco`、`amo_loco`、`asap_tracker`、`twist_tracker`、
`beyondmimic_tracker`、`protomotions_tracker`、`kungfubot_tracker`、`supermimic`、`blinkmimic`、`sonic`。每个工厂函数都把
关键字参数原样传给类的构造函数。

## Profile

一个 profile 是一台机器人上的一个控制器：`<controller>/<robot>`。它的模块
`hiwbc/configs/<controller>/<robot>.py` 保存 checkpoint 训练时的取值（关节顺序、默认姿态、增益、缩放、
历史长度），并负责加载 checkpoint。`hiwbc.load` 由它构建一个可直接使用的控制器。

```python
import hiwbc

hiwbc.available_profiles()      # ['amo_loco/unitree_g1', ..., 'unitree_loco/unitree_g1']
ctrl = hiwbc.load("beyondmimic_tracker/unitree_g1", variant="jump_wose",
                  checkpoint_dir="RoboJuDo/assets/models/g1")
```

| 函数 | 说明 |
|---|---|
| `load(profile, **options) -> Controller` | 构建控制器。`options` 是 `variant` 加上该 profile 的 `default_options`；其他键抛出 `TypeError`，未知 variant 抛出 `ValueError`。checkpoint 在这里加载，所以文件缺失会立刻报错。`runtime="remote"`、`endpoint`、`remote_timeout_s` 让前向在服务器上运行（[远程推理](#远程推理)） |
| `profile_info(profile, options=None) -> dict` | `load(profile, **options)` 会建出什么，但不加载 checkpoint，也不导入模型运行时：`description`、`variant`、`joint_names`（控制器驱动的关节）、`velocity_limits`（同 `WbcRobot.base_velocity_limits`）、`missing_files`（`load` 要打开但磁盘上没有的 checkpoint 文件；`[]` 表示齐了）。`options` 可以是整个 `wbc` layer 条目：不是该 profile 选项的键会忽略。profile 没描述的字段是 `None`；`sonic/unitree_g1` 和 `unitree_loco/unitree_g1` 全都描述了。未知 profile 抛出 `KeyError`，未知 variant 抛出 `ValueError` |
| `get_profile(name) -> Profile` | 返回 `Profile`（`name`、`variants`、`default_variant`、`default_options`、`description`、`info`）。首次使用时导入其模块 |
| `available_profiles() -> list[str]` | 自带的和已注册的 profile。只列模块名而不导入，所以不会加载任何模型运行时 |
| `register_profile(name, *, build, variants, default_options=None, description="", info=None)` | 从 `configs/` 模块注册一个 profile。`build(variant_data, options) -> Controller`；`info(variant_data, options) -> dict` 给出 `joint_names`、`velocity_limits`、`files`（checkpoint 路径），供 `profile_info` 用 |

有 `"default"` 时 `default_variant` 就是它，否则是列出的第一个。

自带的 profile（都用于 29 自由度的 Unitree G1；控制器可以只驱动其中一部分关节）：

| Profile | Variant | `variant` 之外的选项 | 行为 | 片段空间 |
|---|---|---|---|---|
| `unitree_loco/unitree_g1` | `default`（带步态，12 个腿部关节），`wo_gait`（29 个，IsaacLab 顺序） | `checkpoint_dir`, `device` | | |
| `smooth_loco/unitree_g1` | `default` | `checkpoint_dir`, `device` | | |
| `amo_loco/unitree_g1` | `default` | `checkpoint_dir`, `device` | | |
| `asap_tracker/unitree_g1` | `default`（CR7），`horse_squat`，`dance_0605` | `checkpoint_dir`, `providers` | 内置在 checkpoint 中的动作 | |
| `twist_tracker/unitree_g1` | `default` | `checkpoint_dir`, `device` | | `joint` |
| `beyondmimic_tracker/unitree_g1` | `dance_wose`（默认），`jump_wose`，`violin`，`waltz` | `checkpoint_dir`, `providers` | `dance`、`jump`、`violin`、`waltz`（每个 variant 一个） | |
| `kungfubot_tracker/unitree_g1` | `default` | `checkpoint_dir`, `providers` | | `joint` |
| `protomotions_tracker/unitree_g1` | `default` | `checkpoint_dir`, `device` | | `joint` |
| `supermimic/unitree_g1` | `default` | `checkpoint_dir`, `check_hash`, `providers`, `bound_torque` | | `joint` |
| `blinkmimic/unitree_g1` | `default`（BlinkMimic `.pt`），`xmimic`（XMimic `.npz`），`onnx`（训练进一个动作） | `checkpoint`、`precision`（`xmimic`，默认 `"bf16"`）、`providers`（`onnx`）、`bound_torque`（默认 `True`） | 训练进的动作（`onnx`） | `joint`（`default`、`xmimic`） |
| `sonic/unitree_g1` | `default`, `low_latency`, `sonic_v1_1` | `checkpoint_dir`、`providers`、`encoder`、`encoder_mode`、`planner`（默认 `True`）、`planner_thread`（默认 `False`） | | `joint`、`smpl`（需加载编码器） |
| `echo/default` | `default` | `default_pos`, `joint_names`, `kp`, `kd` | | |

`checkpoint_dir` 的默认值：RoboJuDo 部署的 checkpoint 用 RoboJuDo 的 G1 模型目录，即
`$HIWBC_ROBOJUDO_MODELS` 或 `~/.cache/hiwbc/robojudo/g1`；SONIC 用 `~/.cache/hiwbc/sonic`；SuperMimic 用
`~/.cache/hiwbc/supermimic/assets`。BlinkMimic 接收一个文件 `checkpoint`：`~/.cache/hiwbc/blinkmimic/policy.pt`
（`default`）、`~/.cache/hiwbc/xmimic/policy.npz`（`xmimic`，[XMimic](#xmimic)）或 `~/.cache/hiwbc/blinkmimic/dance1_subject2.onnx`
（`onnx`）。两个项目都不发布权重，所以要用 `checkpoint=` 传入你自己训的 policy 或导出文件的路径
（[BlinkMimic](#blinkmimic)）。

## 动作片段

`hiwbc.references.ClipPlayer` 播放一个 hiRobot `MotionClip`（或任何有相同字段 `space`、`fps`、`joint_names`、
`joint_pos`、`joint_vel`、`smpl_joints`、`root_pos`、`root_quat` 的对象），供 `motion_spaces` 包含该片段
`space` 的控制器使用。

```python
from hiwbc.references import ClipPlayer

player = ClipPlayer(clip, ctrl.control_dt, **ctrl.clip_player_options)
ctrl.check_clip(player)                      # 片段缺少 ctrl 要读的内容时抛出 ValueError
while not player.done:
    state = read_state()
    target = ctrl.step(state, ctrl.clip_reference(player, state))
    send(target)
    player.advance()
```

播放器：

- 把片段从 `fps` 重采样到 `control_dt`：`root_quat` 用 slerp，其余线性插值；
- 在第一次 `clip_reference` 时绕 z 轴旋转片段，使其第一帧的根朝向与机器人基座朝向一致，并把第一帧的根
  位置移到 x, y = 0（SONIC 的 `apply_delta_heading`），或者在 `window(..., base_pos=...)` 传入机器人世界
  位置时移到机器人下方（H2H 就是这样做的）；
- 窗口超出片段末尾时重复最后一帧；
- 片段没有速度时，由 `joint_pos` 差分得到 `joint_vel`，由 `root_pos` / `root_quat` 差分得到根的线速度 /
  角速度（按控制频率做简单有限差分，不平滑；设置 `source_velocity_sigma=s` 时，根速度先按片段自身的帧率
  差分，再在 `s` 个源帧上做高斯平滑，与 PHC 的做法相同）；
- 对用 `player.add_derived(name, fn)` 注册的项，在对齐之后按每个片段自身的帧率计算，再重采样；`window`
  在 `name` 下返回它们（KungfuBot 和 H2H 用这种方式注册本体 FK）；
- 按名字把片段的关节匹配到控制器的关节（去掉 `_joint` 后缀），片段缺少的关节用控制器的默认姿态填充
  （速度为零）。

`player.append(clip)` 把一个空间和字段都相同的片段排在当前片段之后，用于流式播放。`done`、`frame`、
`num_frames`、`duration` 和 `time` 报告进度。

| 控制器 | 空间 | 片段必须包含 | 构建的参考 |
|---|---|---|---|
| `sonic` | `joint`、`smpl`（已加载编码器） | `root_quat`；`joint` 还需要控制器的至少一个关节，对于 `joint` 模式读取根高度的 variant 还需要 `root_pos`（已发布的 variant 都不读） | 从当前帧开始的编码器窗口，`encoder_mode` 设为片段的空间 |
| `twist_tracker` | `joint` | `root_pos`、`root_quat`、控制器的至少一个关节 | 当前帧的 31 维 `mimic_target` |
| `supermimic` | `joint` | `root_pos`、`root_quat`、控制器的至少一个关节 | 由 `goal_features` 得到的 10 个目标帧 |
| `blinkmimic` | `joint`（`default`、`xmimic` variant） | `root_pos`、`root_quat`、控制器的至少一个关节 | `goal`（第 `f+1 .. f+10` 帧）和 `ref_root_quat` |
| `kungfubot_tracker` | `joint`（带 `future_steps`、`kinematics`、`key_bodies`，与 profile 中一致） | `root_pos`、`root_quat`、控制器的至少一个关节 | 全部七项：20 个未来帧、带关键点的下一步、`anchor_ref_rot` |
| `protomotions_tracker` | `joint`（带 `kinematics`、`track_points`，与 H2H profile 中一致） | `root_pos`、`root_quat`、控制器的至少一个关节；每个 tick 的 `state.extra["base_pos"]` | 双手和头部的 `task_obs` |

`kungfubot_tracker` 和 `protomotions_tracker` 需要安装 `hirobot` 来做 FK，`ClipPlayer` 本身不需要。其他控制器
不接收片段。`asap_tracker`、`beyondmimic_tracker` 和 `blinkmimic` 的 ONNX 导出文件的动作在 checkpoint 里，以[行为](#行为)的方式播放。

## 行为

行为（behavior）是控制器无需上层给参考就能做的事，例如训练进 ASAP 或 BeyondMimic checkpoint 的动作。
`WbcRobot` 把它们列为 layer skill（`list_layer_skills` / `run_layer_skill`）。

```python
(spec,) = ctrl.behaviors                    # BehaviorSpec
ramp_to(spec.start_pos)                     # 动作的第一个姿态，控制器顺序
source = ctrl.start_behavior(spec.name)     # 会重置控制器
while not source.done:
    state = read_state()
    send(ctrl.step(state, source.next(state)))
```

`BehaviorSpec` 的字段：`name`、`description`、`parameters`（`start_behavior` 关键字参数的 JSON Schema）、
`start_pos`（开始时机器人必须处于的关节姿态，或 `None`）。`ReferenceSource` 有 `done` 和
`next(state) -> Reference`。`done` 之后切换到另一个控制器或参考；如果继续 step，控制器会一直给最后一帧。

| 控制器 | 行为 | 参数 | 结束条件 |
|---|---|---|---|
| `asap_tracker` | `motion_name`（profile 按 variant 设置） | `play_speed`，取值 `(0, 2]`，默认 1 | 相位到达 1 时 |
| `beyondmimic_tracker` | `motion_name`（`dance`、`jump`、`violin`、`waltz`） | 无 | `motion_frames` 个 tick 之后 |
| `blinkmimic`（`onnx` variant） | 导出文件元数据的 `motion` 字段，没有则用文件名（不含扩展名） | 无 | `frames - 1` 个 tick 之后，即 `(frames - 1) × 0.02` 秒 |

## 前向函数

前向函数接收 `observe` 的返回值，返回原始动作，可以是任何含 `num_actions` 个值的类数组对象。有两个加载器
从 checkpoint 文件构建前向函数：

```python
from hiwbc import load_onnx_runner, load_torchscript_runner

forward_fn = load_torchscript_runner("policy.pt")          # 需要 torch extra
forward_fn = load_onnx_runner("policy.onnx")               # 需要 onnx extra
```

`load_torchscript_runner(path, *, device="cpu")` 加上 batch 维，在 `torch.no_grad()` 下运行模型，返回展平的
输出。元组观测作为多个位置参数分别传入。

`load_onnx_runner(path, *, input_names=None, output_name=None, extra_inputs=None,
providers=None, num_threads=None)` 把每个数组作为 `(1, -1)` float32 输入：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `input_names` | 不在 `extra_inputs` 中的模型输入，按模型顺序 | 由观测提供的输入，按 `observe` 返回的顺序 |
| `output_name` | 第一个输出 | 要返回的输出 |
| `extra_inputs` | `None` | 每次调用都输入的常量，名字 -> 数组 |
| `providers` | `["CPUExecutionProvider"]` | onnxruntime providers |
| `num_threads` | onnxruntime 默认值 | intra-op 和 inter-op 线程数；设为 1 时延迟最稳定 |

例如 `kungfubot_tracker` 返回 `(actor_obs, future_motion_targets, history)`，与其 ONNX 输入顺序一致，所以
`load_onnx_runner(path)` 直接可用。BeyondMimic 的 ONNX 还有一个 `time_step` 输入和多个输出；对于调用方构建的
参考，`time_step` 输入 0 即可（已发布模型的 `actions` 输出不依赖它）：

```python
forward_fn = load_onnx_runner(
    "Dance_wose.onnx", input_names=("obs",), output_name="actions",
    extra_inputs={"time_step": np.zeros((1, 1), np.float32)},
)
```

hiWBC 不固定 torch 或 onnxruntime 的版本。`torch` 和 `onnx` 这两个 extra 会安装它们；核心只需要 numpy，
`import hiwbc` 不会导入其中任何一个。

## 远程推理

`hiwbc.load(profile, runtime="remote", endpoint="tcp://host:port",
remote_timeout_s=0.015, ...)` 照常在本地构建控制器，然后把它的每个模型可调用对象换成 `RemoteFn`，调用运行
同一 profile 的服务器。观测构建、历史和后处理都留在本地，只有前向在远端。两台机器上都必须有 checkpoint。

在服务器上：

```bash
python -m hiwbc.serve --profile sonic/unitree_g1 [--variant default] \
    [--host 127.0.0.1] [--port 5555] [--option key=value ...]
```

`--option` 的值按 Python 字面量解析（`planner=False`、`checkpoint_dir=/data/sonic`）；
`providers=CUDAExecutionProvider` 会变成列表。

| 名字 | 说明 |
|---|---|
| `make_remote(controller, endpoint, *, timeout_s=0.015, planner_timeout_s=0.5, connect=True) -> dict[str, RemoteFn]` | 替换由 `hiwbc.load` 构建的控制器的模型可调用对象（`*_fn` 属性，以及作为 `"planner"` 的 SONIC planner）。除非 `connect=False`，否则逐个连接并检查服务器的 profile 和 variant。遇到循环模型抛出 `ValueError` |
| `RemoteFn(endpoint, *, name, profile, variant, timeout_s)` | `fn(*args)` 运行服务器上名为 `name` 的模型。每次调用（包括重连）都受 `timeout_s` 限制。超时、连接关闭或连接被拒绝时抛出 `ForwardTimeout` 并丢弃 socket；下一次调用会重连。`calls`、`reconnects` 是计数 |
| `ForwardTimeout` | `TimeoutError` 的子类。`WbcRobot` 把它当作一次错过的 tick |
| `RemoteError` | 服务器拒绝了连接（profile、variant 或模型不同），或模型在服务器上抛出异常 |
| `ModelServer(controller, host="127.0.0.1", port=5555)` | 服务本身；`start()` 在线程上运行，`serve_forever()` 阻塞，`stop()` 关闭所有连接。`port=0` 选一个空闲端口（`server.port`） |

- **无状态。** 服务器只运行前向，所以客户端随时可以重连。带 buffer 的 TorchScript 模型（`unitree_loco`
  `default` 的 LSTM）会被 `make_remote` 和 `ModelServer` 拒绝。
- **传输格式。** TCP，每次调用一个请求一个回复。每条消息是
  `[u32 length][u32 header length][JSON header][raw array bytes]`，大端序。数组以 C 顺序的原始字节传输，
  dtype 取自固定列表（`float32`、`float64`、`int64`、`int32`、`bool`）；不做任何 unpickle。消息上限 16 MB。
  第一条消息写明协议版本、profile、variant 和模型；不匹配时服务器拒绝。
- **没有认证。** 只绑定 localhost 或可信的有线网络。

## WbcRobot：hiRobot layer

`hiwbc.robot.WbcRobot(inner, profile=None, *, controller=None, variant=None,
**options)` 是一个 hiRobot `Robot`，在 `inner`（下面一层的机器人：assembly 外面套着安全门）上运行控制器。
hiWBC 通过 `hirobot.drivers` entry point 把它注册为 `wbc` layer，所以 `make_robot` 会从配置项
`layers: [{type: wbc, profile: ...}]` 构建它；该项的其余键是它的关键字参数。[机器人指南](guides/robot.zh-CN.md)
走一遍完整设置并列出各个选项。

| 成员 | 说明 |
|---|---|
| `action_modes` | 由控制器决定：`BASE_VELOCITY`（接受 `VELOCITY`）、`MOTION_LATENT`（`LATENT`）、`MOTION_CLIP`（`motion_spaces`）、`CARTESIAN`（`cartesian_targets` 和 `VELOCITY`）、`BASE_POSE`（`BASE_VELOCITY` 且机器人报告 `base_position`） |
| `direct_joints` | `inner` 中控制器不驱动的关节；在任何模式下都可以用 `Action.joint_names` + `joint_target` 指挥它们。只带这些关节的 `BASE_VELOCITY`、`MOTION_LATENT` 或 `CARTESIAN` action（没有 `base_velocity`、`latent` 或 `ee_targets`）只更新它们，参考和它的超时不变 |
| `base_velocity_limits` | 控制器的 `velocity_limits`：`base_velocity` 每个轴的 `(min, max)`，单位同它的取值。没有 `BASE_VELOCITY` 或没给时为 `None`。不强制 |
| `cartesian_targets`, `motion_spaces` | 来自控制器 |
| `connect()`, `reset()` | 在 `start_seconds` 内插值到 `default_pos`，然后运行控制器。真机的 `inner` 会得到一个以 `1 / control_dt` 运行的线程 |
| `send_action(action)` | 设置命令。仿真中还会运行一个外层周期的 tick，所以重复发送上一条命令就会推进仿真时间 |
| `hold()` | 原地站住 |
| `get_status().command` | hiRobot 的 `CommandProgress`，表示调用方最近一条 `BASE_POSE` / `MOTION_CLIP` / `BASE_VELOCITY` / `MOTION_LATENT` / `CARTESIAN` 命令的进度：`done`（已到达、片段已播完；没有终点的模式为 `None`）、`goal`、`clip_time` / `clip_duration`（整个 `append` 队列）。结束后保留到下一条命令；还没发过命令、`hold()` 之后、layer skill 运行期间为 `None`（[指南](guides/robot.zh-CN.md#5-状态)） |
| `get_status().extra["wbc"]` | `profile`、`variant`、`mode`、片段进度、`goal`、`arrived`、`ee_targets`、`ticks`、`missed_ticks`、`late_ticks`、`step_ms_max` |
| `list_layer_skills()`, `run_layer_skill(name, args=None)`, `stop_layer_skill()` | `stand` 加上控制器的[行为](#行为)，由 layer 运行（hiRobot 的 [layer skill](https://github.com/humanoidintelligence/hiRobot/blob/main/docs/api.md#layers)） |
| `list_robot_skills()`, `run_robot_skill(name, args=None)`, `stop_robot_skill()` | hiRobot 的统一列表：上面这些 layer skill，加上厂商的 `list_onboard_skills()` |

每个 tick 的参考依次取：正在运行的行为、正在播放的片段、比 `timeout_s` 更新的上一条 `BASE_VELOCITY` /
`MOTION_LATENT` / `CARTESIAN` 命令、正在走向的 `BASE_POSE` 目标，否则是 `Controller.stand_reference`。
没有站立参考的控制器在 PD 下保持 `default_pos`。

`hiwbc.robot_configs()` 返回包里自带的 hiRobot 配置 `{名字: Path}`，每个都是自由站立的 MuJoCo G1 加
`wbc` layer：`g1_sonic`（SONIC；需要 `onnx` extra 和 SONIC checkpoint）和 `g1_unitree_loco`
（`unitree_loco` LSTM；需要 `torch` extra 和 RoboJuDo 的 `unitree/policy_lstm_1.pt`）。用
`make_robot(path)` 打开。它们在 `hiwbc/hirobot_configs/robots/`，登记在 hiRobot 的 `hirobot.configs`
entry point 下，所以 hiRobot 也能按名字找到：
`make_robot("g1_sonic")`、`defaults: [robots/g1_sonic]`。hiRobot 内置的同名配置优先。

tick 中出现异常时，layer 进入故障状态并急停整个栈（hiRobot 的 `Layer.estop`：先停最底下的机器人，
Unitree 或 MuJoCo 本体会进入阻尼，kp 为 0；然后 layer 停掉自己的线程）：在 `reset()` 之前 `send_action`
都抛出 `RobotFaultError`。`ForwardTimeout`（远程推理）不属于这类异常：该 tick 重发上一个目标，只有连续超过
`max_missed_ticks` 次（或第一个 tick 就超时）才进入阻尼。本体关节状态比 `stale_s` 旧
（`inner.now() - obs.stamps["body"].t`，默认 0.1 s）属于这类异常，每个 tick 都检查，包括启动插值期间。

layer 在 `hold()`、片段或行为结束后以及命令超时时使用 `Controller.stand_reference`，并把
`Controller.cartesian_rest` 作为 `CARTESIAN` 限速开始和回归的位姿。

## 关节映射

每个控制器使用自己的关节顺序（例如 SONIC 用 IsaacLab 顺序，见
`hiwbc.configs.sonic.unitree_g1.G1_JOINT_NAMES`）。关节名采用 hiRobot 的写法（G1 的关节名没有 `_joint`
后缀），所以可以用 hiRobot 的 `DofMap` 按名字重排向量：

```python
from hirobot import DofMap
from hiwbc.configs.sonic.unitree_g1 import G1_JOINT_NAMES

to_ctrl = DofMap(robot.embodiment.joint_names, G1_JOINT_NAMES)
dof_pos = to_ctrl.remap(obs.joint_pos)                  # 机器人顺序 -> 控制器顺序
q_cmd = to_ctrl.scatter(target.position, default=hold)  # 控制器顺序 -> 机器人顺序
```

控制器可以只用机器人关节的一个子集；`scatter` 用 `default` 填充其余关节。控制器的某个关节不在机器人上时，
`DofMap` 抛出 `ValueError`。

## 速度命令控制器

都接受 `ReferenceKind.VELOCITY`。`unitree_loco` 和 `smooth_loco` 接收形状为 `(3,)` 的 `value`；其他形状或
`value=None` 抛出 `ValueError`。它们维护一个步态时钟 `phase = timestep / (control_hz * cycle_time)`，每次
`step` 推进一个 tick，`reset()` 时重新开始；默认 `action_beta=0.8`、`action_scale=0.25`。
[`amo_loco`](#amo_loco) 接收 3 维或 7 维命令。

### `unitree_loco`

`UnitreeLocoController`，仿照 RoboJuDo 的 `UnitreePolicy`
（[unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym)；对
[unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab) 设 `with_gait=False`）。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `num_actions`, `default_pos`, `forward_fn` | 必填 | |
| `control_hz` | `50` | 控制频率；决定时钟步长 |
| `cycle_time` | `0.8` | 步态周期，s |
| `with_gait` | `True` | 追加步态 sin / cos 项 |
| `obs_scales` | `ObsScales()` | `dof_pos=1.0`、`dof_vel=0.05`、`ang_vel=0.25`、`command=(2.0, 2.0, 0.25)` |
| `max_cmd` | `(0.8, 0.5, 1.57)` | 每个轴的命令乘数 |
| `history_length` | `0` | 每项保留的 tick 数；需要 `with_gait=False`（unitree_rl_lab：`5`） |
| `gravity_scale` | `1.0` | 投影重力的乘数 |
| `action_scale`, `action_clip`, `action_beta` | `0.25`, `None`, `0.8` | 见 [Controller](#controller) |

观测是一个含 `9 + 3 * num_actions (+ 2)` 个值的扁平向量：

| # | 项 | 宽度 |
|---|---|---|
| 1 | `base_ang_vel * ang_vel` | 3 |
| 2 | 投影重力 | 3 |
| 3 | `value * command * max_cmd` | 3 |
| 4 | `(dof_pos - default_pos) * dof_pos` | n |
| 5 | `dof_vel * dof_vel` | n |
| 6 | 上一个动作 | n |
| 7 | `sin(2π·phase)`、`cos(2π·phase)`（仅在 `with_gait` 时） | 2 |

`history_length > 0` 时（RoboJuDo 的 `UnitreeWoGaitPolicy`），第 1 到 6 项各自保留 `history_length` 个
tick（包括当前 tick），按项优先（term-major）排列，最旧的在前：先是全部角速度帧，再是全部重力帧，依此类推。
缓冲区在构造时和 `reset()` 时填零。

### `smooth_loco`

`SmoothLocoController`，仿照 RoboJuDo 的 `SmoothPolicy`
（[Smooth Humanoid Locomotion](https://github.com/zixuan417/smooth-humanoid-locomotion)）。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `num_actions`, `default_pos`, `forward_fn` | 必填 | |
| `control_hz`, `cycle_time` | `50`, `0.8` | 同 `unitree_loco` |
| `n_priv_latent` | `0` | 填零的特权 latent 槽位宽度 |
| `history_length` | `0` | 追加的过去本体感知块数 |
| `obs_scales` | `SmoothObsScales()` | `ang_vel=0.25`、`dof_vel=0.05` |
| `action_scale`, `action_clip`, `action_beta` | `0.25`, `None`, `0.8` | |

本体感知块（`10 + 3 * num_actions` 个值）：

| # | 项 | 宽度 |
|---|---|---|
| 1 | `sin(2π·phase)`、`cos(2π·phase)` | 2 |
| 2 | `value`（不缩放） | 3 |
| 3 | `base_ang_vel * ang_vel` | 3 |
| 4 | 基座 roll、pitch | 2 |
| 5 | `dof_pos - default_pos` | n |
| 6 | `dof_vel * dof_vel` | n |
| 7 | 上一个动作 | n |

观测 = `proprio`、`zeros(n_priv_latent)`，然后是 `history_length` 个过去的本体感知块，最旧的在前。历史只包含
更早的 tick；它在构造时和 `reset()` 时填零。

### `amo_loco`

`AmoLocoController`（`hiwbc.controllers.amo`），仿照 RoboJuDo 的 `AMOPolicy`
（[AMO](https://github.com/OpenTeleVision/AMO)）。policy 观测全部躯体关节，但只驱动前 `num_actions` 个
（G1：观测 23 个，驱动 15 个）。一个 adapter MLP 把高度 / 躯干命令和手臂角度映射为观测的一部分，模型还有
第二个输入。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `num_actions`, `default_pos` | 必填 | 驱动的关节及其默认姿态 |
| `obs_default_pos` | 必填 | 所有观测关节的默认姿态。前 `num_actions` 项必须等于 `default_pos` |
| `forward_fn` | 必填 | `(obs, extra_history) -> action` |
| `adapter_fn` | 必填 | `(4 + n_upper,) -> (n_adapter_out,)`，输入和输出都已归一化 |
| `adapter_input_mean`, `adapter_input_std`, `adapter_output_mean`, `adapter_output_std` | 必填 | adapter 的归一化统计量 |
| `control_hz` | `50` | |
| `ang_vel_scale`, `dof_vel_scale` | `0.25`, `0.05` | |
| `action_scale` | `0.25` | 无 EMA，缩放后不裁剪 |
| `raw_action_clip` | `40.0` | 对模型原始输出的裁剪 |
| `default_command` | `(0, 0, 0, 0.75, 0, 0, 0)` | 3 维命令时使用其后四项 |
| `gait_freq` | `1.3` | 步态时钟频率，Hz |
| `n_priv` | `3` | 填零的特权槽位 |
| `history_length`, `extra_history_length` | `10`, `25` | 观测 / 第二个输入中的本体感知帧数 |

`load_amo_runners(policy_file, adapter_file, norm_file, *, device="cpu")` 以关键字参数形式返回
`forward_fn`、`adapter_fn` 和四个统计量（需要 `torch` extra）。

`value` 有 7 项 `[c0, c1, c2, height, torso_yaw, torso_pitch, torso_roll]`，或 3 项（其余取自
`default_command`）。`c1` 驱动 yaw 项（`dyaw = -c1`）；`c0` 和 `c2` 进入 demo 项。

本体感知帧（`proprio_size` 个值）：`base_ang_vel * ang_vel_scale`（3），roll 和 pitch（2），`dyaw` 的
`sin`、`cos`（2），`dof_pos - obs_default_pos`（n），`dof_vel * dof_vel_scale`（n），扩展到全部观测关节的
上一个动作（n；不驱动的项为 `offset / action_scale`），步态 `sin`（2），adapter 输出。

`observe` 返回 `(obs, extra_history)`：

- `obs`：本体感知，demo（`n_upper + 9`：手臂角度、`c0`、`c2`、0、躯干 yaw / pitch / roll、三次
  `0.75 + height`），`zeros(n_priv)`，然后是之前 `history_length` 个本体感知帧，最旧的在前。
- `extra_history`：最近 `extra_history_length` 个本体感知帧，最旧的在前，包括当前帧。

两个缓冲区都在构造时和 `reset()` 时填零。步态时钟从 `[0.25, 0.25]` 开始；`c0..c2` 全部小于 0.1 时，它把两只
脚锁定为同相，否则让它们转为反相。

## 动作跟踪控制器

都接受 `ReferenceKind.MOTION_FRAMES`。`action_beta` 默认为 `1.0`（无 EMA），所以"上一个动作"项是上一个 tick
的模型原始输出；H2H student 用 `0.8`。`action_scale`（标量或每个关节一个）默认为 `1.0`，`action_clip` 默认为
`None`。

跟踪控制器读取已经为当前 tick 算好的参考项。`twist_tracker`、`supermimic`、`kungfubot_tracker` 和
`protomotions_tracker`（以及 `sonic`）可以用 `clip_reference` 从动作片段计算这些项（[动作片段](#动作片段)；
KungfuBot 和 H2H 需要其 profile 设置好的 FK），`asap_tracker` 和 `beyondmimic_tracker` 把自己的动作作为
[行为](#行为)播放。除此之外，回放和朝向对齐由调用方负责。每个 `frames` 值在使用前展平。缺少某个键时抛出
指明该键的 `ValueError`。

四个跟踪控制器（`hiwbc.controllers.twist`、`beyondmimic`、`protomotions`、`kungfubot`）共用这些参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `num_actions`, `default_pos`, `forward_fn` | 必填 | |
| `history_length` | `0` | 保留的过去块数。在构造时和 `reset()` 时填零 |
| `action_scale`, `action_clip`, `action_beta` | `1.0`, `None`, `1.0` | |

### `asap_tracker`

`AsapTrackerController`，仿照 RoboJuDo 的 `AsapPolicy`（[ASAP](https://github.com/LeCAR-Lab/ASAP)）。动作
内置在 checkpoint 中，所以不读 `frames`；控制器自己推进相位。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `num_actions`, `default_pos`, `forward_fn` | 必填 | |
| `motion_length_s` | 必填 | 内置动作的长度，s |
| `control_hz` | `50` | |
| `use_history` | `False` | 插入历史项 |
| `history_length` | `4` | 过去块数，最新的在前 |
| `obs_scales` | `AsapObsScales()` | 全为 `1.0`：`actions`、`base_ang_vel`、`dof_pos`、`dof_vel`、`projected_gravity`、`ref_motion_phase`、`history` |
| `action_scale`, `action_clip` | `1.0`, `None` | |

观测：上一个动作（n），`base_ang_vel`（3），`dof_pos - default_pos`（n），`dof_vel`（n），历史
（`history_length * (3n + 7)`，仅在 `use_history` 时），投影重力（3），相位（1）。历史包含更早的 tick，
各项为上一个动作、角速度、关节位置、关节速度、重力、相位，按项优先排列，最新的在前：先是全部
`history_length` 个上一个动作，再是全部角速度，依此类推。它在构造时和 `reset()` 时填零。

`phase = timestep / (control_hz * motion_length_s)`。每次 `step` 之后，timestep 推进
`reference.extra.get("play_speed", 1.0)`。相位到达 `1.0` 后不再推进，`motion_done` 变为 `True`。`reset()`
让动作重新开始。

### `twist_tracker`

`TwistTrackerController`，仿照 RoboJuDo 的 `TwistPolicy`（[TWIST](https://github.com/YanjieZe/TWIST)）。

额外参数：`ang_vel_scale`、`dof_pos_scale`、`dof_vel_scale`（都是 `1.0`），`zero_dof_vel_mask`（`None`；
一个 `(n,)` 数组，在观测中关节速度被置零的关节处为 `1`）。

| `frames` 键 | 内容 |
|---|---|
| `mimic_target` | 上半身模仿目标 |

观测 = `mimic_target`、本体感知，然后是 `history_length` 个过去的 `[mimic_target, proprio]` 块，最旧的在前。
本体感知为 `base_ang_vel * ang_vel_scale`（3），roll 和 pitch（2），`(dof_pos - default_pos) * dof_pos_scale`
（n），`dof_vel * dof_vel_scale`（n），上一个动作（n）。TWIST 训练时踝关节速度置零；传入在踝关节处为 `1` 的
`zero_dof_vel_mask`（G1 23 自由度：索引 4、5、10、11）。

### `beyondmimic_tracker`

`BeyondMimicTrackerController`，仿照 RoboJuDo 的 `BeyondMimicPolicy`
（[BeyondMimic](https://github.com/HybridRobotics/whole_body_tracking)）。

额外参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `without_state_estimator` | `False` | 用于训练时没有 anchor 位置和基座线速度项的 checkpoint（`*_wose`） |
| `motion_fn` | `None` | `motion_fn(obs, time_step) -> dict`，含模型的 `actions`、`joint_pos`、`joint_vel` 和 `body_quat_w`（`[w, x, y, z]`）输出。启用行为 |
| `motion_name` | `None` | 行为名。有 `motion_fn` 时必填 |
| `motion_frames` | `None` | 内置动作的帧数。有 `motion_fn` 时必填 |
| `anchor_body_index` | `None` | anchor body（`torso_link`）在 `body_quat_w` 中的索引。有 `motion_fn` 时必填 |

| `frames` 键 | 内容 | 是否需要 |
|---|---|---|
| `command` | 参考关节命令 | 总是 |
| `motion_anchor_pos_b` | 机器人 anchor 坐标系下的参考 anchor 位置 | 除非 `without_state_estimator` |
| `motion_anchor_ori_b` | 机器人 anchor 坐标系下的参考 anchor 姿态 | 总是 |
| `base_lin_vel` | 基座线速度（来自状态估计器） | 除非 `without_state_estimator` |

观测：`command`，[`motion_anchor_pos_b`]，`motion_anchor_ori_b`，[`base_lin_vel`]，`base_ang_vel`（3），
`dof_pos - default_pos`（n），`dof_vel`（n），上一个动作（n）。方括号中的项在 `without_state_estimator` 时
去掉。没有历史，没有观测缩放。`obs_dim` 是观测长度（29 个关节时，无状态估计器为 154，有则为 160）。

**行为。** 每个已发布的 checkpoint 只在一个动作上训练，其 ONNX 文件对给定的 `time_step` 输出该动作的参考。
有 `motion_fn` 时，控制器按 RoboJuDo `use_motion_from_model` 模式的方式播放它：

1. `start_behavior` 重置控制器，并在第 0 帧对全零观测运行一次前向。其 `actions` 成为初始的上一个动作，其参考
   成为第一帧。`BehaviorSpec.start_pos` 是该帧的 `joint_pos`。
2. 每个 tick，观测读取上一次前向给出的参考，前向在当前帧（`reference.extra["time_step"]`）运行，所以第 0 和
   第 1 个 tick 都读第 0 帧。
3. 参考 anchor 的朝向在第一个 tick 只按 yaw 转到机器人基座的朝向上。
4. `motion_anchor_ori_b` 是参考 anchor 相对机器人躯干的姿态。躯干姿态在给出时取 `state.extra["torso_quat"]`，
   否则是 `base_quat` 再经 `waist_yaw`、`waist_roll` 和 `waist_pitch` 关节角旋转（G1 从骨盆到躯干的链）。
5. 有状态估计器时，`motion_anchor_pos_b` 为零（取机器人 anchor 位置等于参考的 anchor 位置，即 RoboJuDo 的
   `override_robot_anchor_pos`），`base_lin_vel` 来自 `state.base_lin_vel`；它为 `None` 时 `next` 抛出
   `ValueError`。

参考源在 `motion_frames` 个 tick 之后 `done`；继续 step 时重复最后一帧。不带 `time_step` 的参考照旧走
`forward_fn`。

### `protomotions_tracker`

`ProtoMotionsTrackerController`，仿照 RoboJuDo 的 `ProtoMotionsTrackerPolicy`
（[ProtoMotions](https://github.com/NVlabs/ProtoMotions)）。

额外参数：`dof_vel_scale`、`ang_vel_scale`（`1.0`），`use_dof_pos_offset`（`False`：输入绝对 `dof_pos`；
`True`：输入 `dof_pos - default_pos`）。

| `frames` 键 | 内容 |
|---|---|
| `task_obs` | 模仿观测，在上游计算（需要本体 FK） |

观测：`dof_pos` 或其偏移（n），`dof_vel * dof_vel_scale`（n），`base_ang_vel * ang_vel_scale`（3），投影重力
（3），`task_obs`，上一个动作（n），然后是历史，最新的在前。每个历史块是不缩放的绝对 `dof_pos`、`dof_vel`、
`base_ang_vel`、重力、上一个动作。

### `kungfubot_tracker`

`KungfuBotTrackerController`，仿照 RoboJuDo 的 `KungfuBotGeneralPolicy`
（[KungfuBot / PBHC](https://github.com/TeleHuman/PBHC)）。模型有三个输入，所以 `observe` 返回一个元组，
`forward_fn` 接收它。

额外参数：`ang_vel_scale`、`dof_pos_scale`、`dof_vel_scale`、`roll_pitch_scale`、`actions_scale`、
`history_scale`（都是 `1.0`）。

| `frames` 键 | 内容 |
|---|---|
| `anchor_ref_rot` | 参考 anchor 旋转 |
| `next_step_ref_motion` | 下一步的参考动作 |
| `future_motion_base_lin_vel`, `future_motion_base_yaw_vel`, `future_motion_dof_pos`, `future_motion_roll_pitch`, `future_motion_root_height` | 未来动作目标 |

`observe` 返回 `(actor_obs, future_motion_targets, history)`：

- `actor_obs`：上一个动作、`anchor_ref_rot`、角速度、`dof_pos - default_pos`、关节速度、历史、
  `next_step_ref_motion`、roll 和 pitch。这是 policy 按名字排序后的顺序；每个本体感知项乘以其缩放。
- `future_motion_targets`：五个 `future_motion_*` 项，按表中列出的顺序。
- `history`：`history_length` 个过去条目，每条为上一个动作、角速度、绝对 `dof_pos`、关节速度、roll 和 pitch
  （各自缩放），按项优先排列，最新的在前，然后乘以 `history_scale`。`reset()` 后填零。

## SuperMimic

`SuperMimicController`（`hiwbc.controllers.supermimic`），注册名为 `supermimic`。这是面向 29 自由度 Unitree G1
的非官方复现：它遵循 [supermimic_deploy](https://github.com/KyrieYoung926/supermimic_deploy) 的推理代码，
该仓库的 ONNX policy 用 SuperMimic-Reproduced 训练。其他复现使用不同的观测布局，不在支持范围内。

```python
from hiwbc.controllers.supermimic import SuperMimicController

ctrl = SuperMimicController.from_bundle("supermimic_deploy/assets")
# 或：make_controller("supermimic", assets_dir="supermimic_deploy/assets")
```

`from_bundle(assets_dir, *, check_hash=True, providers=None, **kwargs)` 读取资源包中的 `config.json` 和
`policy.onnx`。关节顺序与 `SUPERMIMIC_JOINT_NAMES` 不同、历史或未来帧数不是 10、或 ONNX 的 sha256 与配置不符时，
抛出 `ValueError`。需要 `onnx` extra。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `forward_fn` | 必填 | `((930,), (680,)) -> (29,)`；可用 `load_onnx_runner(path)` 构建 |
| `default_pos` | 必填 | `q0`，SDK 关节顺序 |
| `action_scale` | 必填 | 每个关节一个 |
| `kp`, `kd` | `None` | 随每个目标一起返回 |
| `effort` | `None` | 每个关节的力矩上限 |
| `bound_torque` | `True` | 重新表达目标，使 `kp * (target - q) - kd * dq` 不超过 `effort`。需要 `kp`、`kd`、`effort` |

`accepts = (MOTION_FRAMES,)`。`frames["goal"]` 形状为 `(T, 68)`，第 0 帧是 50 Hz 下的当前 tick；使用第
`0..9` 帧，`T < 10` 时重复最后一帧。`goal_features(joint_pos, joint_vel, root_quat, root_lin_vel_w,
root_ang_vel_w, root_z, default_pos)` 从 50 Hz 片段计算 68 维目标：`q - q0`（29），`dq`（29），投影重力（3），
根坐标系下的根角速度（3），朝向坐标系下的根线速度（3），根高度（1）。

模型有两个输入：

- `obs_history`（930）：10 个 tick 的 `[q - q0, dq, base ang vel, projected gravity, last action]`（每个 93），
  按项优先排列，最旧的在前，包括当前 tick。`reset()` 后的第一个 tick，10 个槽位都填当前条目。
- `goal`（680）：10 个目标帧，按帧优先（frame-major）排列。

动作映射为 `q0 + action_scale * action`，无 EMA，无裁剪。

## BlinkMimic

`BlinkMimicController`（`hiwbc.controllers.blinkmimic`），注册名为 `blinkmimic`。它在 29 自由度 Unitree G1 上
（SDK 关节顺序，50 Hz）运行用 [BlinkMimic](https://github.com/fuyichen0725/blinkmimic)（PyTorch）训出的 policy。
BlinkMimic 沿用 SuperMimic 的方法，但观测和 goal 与 [`supermimic`](#supermimic) 不同。XMimic 的 policy 接口相同，
也在这个控制器上运行（见 [XMimic](#xmimic)）。

```python
import hiwbc
from hiwbc.controllers.blinkmimic import BlinkMimicController, load_pt_runner

ctrl = hiwbc.load("blinkmimic/unitree_g1", checkpoint="model_2700.pt")
# 或手动构建：
from hiwbc.controllers.blinkmimic import G1_ACTION_SCALE, G1_DEFAULT_POS, G1_EFFORT, G1_KD, G1_KP

fwd = load_pt_runner("model_2700.pt", default_pos=G1_DEFAULT_POS, action_scale=G1_ACTION_SCALE)
ctrl = BlinkMimicController(forward_fn=fwd, default_pos=G1_DEFAULT_POS,
                            action_scale=G1_ACTION_SCALE, kp=G1_KP, kd=G1_KD, effort=G1_EFFORT)
```

| 参数 | 默认值 | 说明 |
|---|---|---|
| `forward_fn` | `None` | `forward_fn((obs_history (10, 99), goal (730,))) -> (29,)`；可用 `load_pt_runner`（XMimic 用 `load_npz_runner`）构建。单动作导出文件用 `None`，此时只播放它的行为。`forward_fn`、`motion_fn` 至少给一个 |
| `default_pos` | 必填 | `q0`，SDK 关节顺序 |
| `action_scale` | 必填 | 每个关节一个，`0.25 * effort / kp` |
| `kp`, `kd` | `None` | 随每个目标一起返回 |
| `effort` | `None` | 每个关节的力矩上限 |
| `bound_torque` | `True` | 重新表达目标，使 `kp * (target - q) - kd * dq` 不超过 `effort`，与训练时执行器截断力矩一致。需要 `kp`、`kd`、`effort` |
| `control_dt` | `0.02` | 控制周期，单位秒 |
| `motion_fn` | `None` | 单动作导出文件，`motion_fn(obs_history (10, 99), time_step) -> dict`，含 `actions`、`joint_pos` 和 `next_anchor_quat_w`（`[w, x, y, z]`）；可用 `load_onnx_motion` 构建。启用行为 `motion_name` |
| `motion_name` | `None` | 训练进的动作的行为名。有 `motion_fn` 时必填 |
| `motion_frames` | `None` | 训练进的动作的帧数（50 Hz），至少 2。有 `motion_fn` 时必填 |
| `anchor_quat_init` | `None` | 动作第 0 帧的根姿态，`[w, x, y, z]`。有 `motion_fn` 时必填 |

Runner（两者都是无状态的 `forward_fn((obs_history, goal)) -> action`，所以 `runtime="remote"` 可用）：

| 函数 | 读取 | 说明 |
|---|---|---|
| `load_npz_runner(path, *, default_pos, action_scale, precision="bf16")` | XMimic `ckpt_*.npz` | 见 [XMimic](#xmimic) |
| `load_pt_runner(path, *, default_pos=None, action_scale=None)` | BlinkMimic `.pt` | 用 torch 读文件（`torch` extra），然后在 numpy 中以 fp32 运行网络。残差用 checkpoint 自己的 `q0` 和 scale；若给了 `default_pos` / `action_scale`，必须与之一致 |

`goal_features(root_pos, root_quat, joint_pos, *, frames=10, dt=0.02)` 由连续的 50 Hz 参考行
`f, f+1, ..., f+11`（接近片段末尾时更少，至少 2 行）构建 `(10, 73)` 的 goal，与训练器的 `Motion.goal` 一致。

网络用 checkpoint 里的滑动均值和方差归一化两个输入，编码 goal，再和历史一起解码，最后加上下一帧的参考
关节角 `(q_ref - q0) / action_scale`（存在 checkpoint 里的残差）。目标为 `q0 + action_scale * action`，
无 EMA，无裁剪。

`accepts = (MOTION_FRAMES,)`。`frames`：

- `goal` `(10, 73)`：50 Hz 下的参考帧 `f+1 .. f+10`，到片段末尾时重复最后一帧。每帧：在当前参考帧 `f` 的
  朝向坐标系下的根位移（3）和旋转（6D，6）；该未来帧自身根坐标系下的根线速度和角速度（3 + 3）；关节角（29）；
  关节速度 × 0.1（29）。速度是中心差分。
- `ref_root_quat` `(4,)`：参考帧 `f` 的根姿态，世界坐标系，`[x, y, z, w]`。

模型有两个输入：

- `obs_history`（990）：10 个 tick 的 `[q - q0, dq, base ang vel（机体坐标系）, projected gravity,
  previous action, 6D(R^T R_ref)]`（每个 99），按帧优先（frame-major）排列，最旧的在前，包括当前 tick。
  6D 项取旋转矩阵的前两列，按行展平。`reset()` 后的第一个 tick，10 个槽位都填当前条目。
- `goal`（730）：10 个目标帧，按帧优先排列。

`R_ref` 是世界坐标系中的姿态，所以开始时片段必须转到机器人的朝向；`ClipPlayer` 会这样做。

profile 的 `onnx` variant 加载 BlinkMimic `tools/export_onnx.py` 导出的、训练进一个动作的文件，把这个动作
作为[行为](#行为)播放。需要 `onnx` extra。`load_onnx_motion(path, providers=None)` 返回
`(motion_fn, metadata)`；导出文件的输入是 `(obs_history, time_step)`，goal 在文件内部，所以不能跟踪其他片段
（`check_clip` 会报错）。`check_onnx_metadata(metadata, source)`（`hiwbc.configs.blinkmimic.unitree_g1`）在导出
文件的关节名、增益、默认姿态或 action scale 与 `G1_*` 的值不同（相对容差 1e-5），或历史长度、goal 帧数、
控制周期、观测名或 anchor body（pelvis）与本控制器不同时抛出 `ValueError`；profile 加载时会调用它。

这个行为按上游 `tools/play_onnx.py` 的方式播放动作：`time_step` 依次为 `0, 1, ..., frames - 2`，每个 50 Hz
帧一个 tick，所以行为持续 `(frames - 1) × 0.02` 秒。6D 项在第 0 个 tick 用导出文件的 `anchor_quat_w_init`，
之后用上一个 tick 的 `next_anchor_quat_w`。上一个动作从零开始。第一个 tick 时，动作只绕 yaw 转到机器人基座的
朝向。`spec.start_pos` 是动作的第一个姿态；`WbcRobot` 开始前先缓动到这个姿态。forward 失败的 tick 不会推进
动作。

## XMimic

[XMimic](https://github.com/nico-bohlinger/xmimic)（JAX；Bohlinger 和 Peters，
[arXiv:2611.20452](https://arxiv.org/abs/2611.20452)）是 BlinkMimic 的重写，policy 接口相同。它的 policy 经
profile 的 `xmimic` variant 在 [`BlinkMimicController`](#blinkmimic) 上运行；上面的参数、frames、布局和增益都照用。

```python
import hiwbc
from hiwbc.controllers.blinkmimic import G1_ACTION_SCALE, G1_DEFAULT_POS, load_npz_runner, self_check

ctrl = hiwbc.load("blinkmimic/unitree_g1", variant="xmimic", checkpoint="ckpt_5800.npz")
# 或只建 runner：
fwd = load_npz_runner("ckpt_5800.npz", default_pos=G1_DEFAULT_POS, action_scale=G1_ACTION_SCALE)
assert self_check(fwd) < 1e-4
```

`load_npz_runner(path, *, default_pos, action_scale, precision="bf16")` 只用 numpy 读 XMimic 的 `ckpt_*.npz`，
返回无状态的 `forward_fn((obs_history, goal)) -> action`。`precision="bf16"` 像 XMimic 评估时一样把 matmul 输入
舍入到 bf16；`"fp32"` 不舍入。profile 选项 `precision` 原样传给它。

`self_check(runner, test_vectors=None) -> float` 在 XMimic `.npz` 自带的测试向量（或你传入的
`(obs_history, goal, action)`）上运行 runner，返回相对 `max(1, |action|)` 的最大动作误差。`precision="bf16"`
的 runner 与 XMimic 的向量相差约 1e-7；`"fp32"` 的 runner 最多相差约 1e-2。

## SONIC

`SonicController`（`hiwbc.controllers.sonic`），注册名为 `sonic`。它用 NVIDIA 的
[GEAR-SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl) 驱动 29 自由度 Unitree G1，由两个模型组成：

- **编码器**把参考映射为 64 维运动 token；
- **解码器**把 token 加上 10 个 tick 的本体感知历史映射为 29 个关节动作。

观测布局遵循 NVIDIA 的 C++ 部署（`gear_sonic_deploy`）和每个已发布 checkpoint 的 `observation_config.yaml`。
hiWBC 不附带权重。

已在 MuJoCo 中用已发布的 `default`、`low_latency` 和 `sonic_v1_1` checkpoint 与 C++ 部署的逐行移植版对照检查：
编码器输入、token、解码器输入和 PD 目标的差异都在 1e-5 以内。完整设置见 [SONIC 指南](guides/sonic.zh-CN.md)。

### 构建

profile 从 checkpoint 文件夹构建它：

```python
ctrl = hiwbc.load("sonic/unitree_g1", variant="default")   # checkpoint_dir: ~/.cache/hiwbc/sonic
```

`encoder=False` 跳过编码器（只支持 `LATENT`）；`encoder_mode` 设置初始模式。也可以直接构建：

```python
from hiwbc.configs.sonic.unitree_g1 import SONIC_G1, SONIC_VARIANTS
from hiwbc.controllers.sonic import SonicController

ctrl = SonicController.from_onnx(
    "model_decoder.onnx",
    "model_encoder.onnx",     # 可选；只有 MOTION_FRAMES 需要
    robot=SONIC_G1,
    obs=SONIC_VARIANTS["default"],
    encoder_mode="joint",
)
```

`from_onnx(decoder_path, encoder_path=None, *, providers=None, **kwargs)` 用 `load_onnx_runner` 包装每个文件，
并把 `kwargs` 传给构造函数。需要 `onnx` extra。`load_onnx_runner(path, *, providers=None)` 返回一个
`forward_fn`，把 `(1, dim)` float32 张量输入模型唯一的输入；`providers` 默认为 `["CPUExecutionProvider"]`。

| 参数 | 默认值 | 说明 |
|---|---|---|
| `decoder_fn` | 必填 | `(decoder_dim,) -> (n,)` |
| `encoder_fn` | `None` | `(encoder_dim,) -> (64,)`。`MOTION_FRAMES` 需要 |
| `robot` | 必填 | 一个 `SonicRobotConfig`；29 自由度 G1 用 `SONIC_G1` |
| `obs` | 必填 | 一个 `SonicObsConfig`；已发布的是 `SONIC_VARIANTS[...]` |
| `encoder_mode` | `"joint"` | `joint`、`teleop` 或 `smpl`；可以逐 tick 覆盖 |
| `stand_root_z` | `0.793` | `stand_reference` 使用的根高度，m |
| `planner` | `None` | 一个 `SonicPlanner`。有它时控制器还接受 `VELOCITY` |
| `cartesian_targets` | `None` | 末端执行器名 -> VR 三点名（`left_wrist`、`right_wrist`、`head`）；planner 打开时 profile 设置两个手腕 |

加载了编码器时 `motion_spaces` 为 `("joint", "smpl")`，没有则为 `()`。`clip_reference` 把 `encoder_mode` 设为
片段的空间，并传入编码器读取的窗口（[动作片段](#动作片段)）。`stand_reference` 在 `stand_root_z` 高度保持默认
姿态，面向机器人当前朝向，使用 `joint` 模式且 `play=False`。

`SonicRobotConfig(joint_names, default_pos, action_scale, kp, kd,
lower_body_index, wrist_index=())` 保存所有与机器人相关的内容，按控制器的关节顺序。`kp` / `kd` 随每个目标一起
返回。`lower_body_index` 和 `wrist_index` 选出 teleop 和 SMPL 编码器项读取的关节；没有手腕的机器人把
`wrist_index` 留空，此时用到手腕项的 checkpoint 会在构造时抛出 `ValueError`。观测中的关节项按配置决定大小，
所以控制器中没有任何地方假定 29 个关节。

不做 EMA，也不裁剪。对 `SONIC_G1`，关节顺序是 IsaacLab 顺序 `G1_JOINT_NAMES`；它与 `G1_JOINT_NAMES_MUJOCO`
中 G1 的硬件（MuJoCo）顺序不同，所以要在上游按名字重映射关节。`joint` 就是部署中的 `g1` 编码器模式（id 0），
改名是因为它的含义是"机器人关节空间的动作"，而不是指某台特定机器人。

### Variant

| Variant | Checkpoint | 编码器维度 | 解码器维度 |
|---|---|---|---|
| `default` | `policy/release/observation_config.yaml` | 1762 | 994 |
| `low_latency` | 低延迟版本 | 1247 | 994 |
| `sonic_v1_1` | SONIC v1.1 | 1751 | 994 |

`ctrl.encoder_dim` 和 `ctrl.decoder_dim` 给出所选 variant 的尺寸。项名沿用部署中的写法：
`<term>_<N>frame_step<S>` 表示 N 帧、相邻两帧相隔 S 个 tick。`default` 每隔 5 个 tick 取一个未来参考帧，
`low_latency` 每个 tick 取一个；`sonic_v1_1` 在机器人的朝向坐标系（只含 yaw）而不是完整基座坐标系中表达 anchor
姿态。

`SonicObsConfig(encoder_terms, encoder_modes, decoder_terms=..., token_dim=64)` 描述一个 checkpoint。
`encoder_modes` 把模式名映射到 `(mode_id, required_terms)`。要支持新的 checkpoint，按其
`observation_config.yaml` 逐项构建一个 `SonicObsConfig`，作为 `obs` 传入。

### 参考

`accepts = (LATENT, MOTION_FRAMES)`，有 planner 时再加 `VELOCITY`。

**`VELOCITY`**（仅限 planner）：`value` 是本体坐标系下的 `[vx, vy, wz]`；`[0, 0, 0]` 表示站住。planner 把它变成
`joint` 编码器使用的未来帧。`extra["ee_targets"]`（`{vr point name: [x, y, z, qx, qy, qz, qw]}`，根坐标系）切换
到 `teleop` 模式并使用这些 VR 点；缺少的点取默认姿态。见
[SONIC 指南](guides/sonic.zh-CN.md#7-行走速度命令与-planner)。

**`LATENT`**：`value` 是 64 维 token，例如 VLA 的运动 token。不运行编码器。

**`MOTION_FRAMES`**：`frames` 保存一个未来参考帧窗口，采样率为 50 Hz 控制频率，第 0 帧是当前 tick。控制器对
每一项取第 `0, S, 2S, ...` 帧，窗口不够长时重复最后一帧；也接受单帧（没有最前面的时间轴）。动作回放、把片段
对齐到机器人朝向以及 FK 都在上游完成。

| 模式 | `frames` 键 |
|---|---|
| `joint` | `joint_pos` (T, n), `joint_vel` (T, n), `root_quat` (T, 4) |
| `teleop` | `joint_pos`、`joint_vel`（下半身关节在内部选出）、`root_quat`、`vr_3point_pos` (3, 3)、`vr_3point_quat` (3, 4) |
| `smpl` | `smpl_joints` (T, 24, 3)（根局部坐标）、`root_quat`、`joint_pos`（手腕关节在内部选出） |

- 关节数组按 `robot.joint_names` 顺序（`SONIC_G1` 为 IsaacLab 顺序）。
- `root_quat` 是世界坐标系下的参考根姿态 `[x, y, z, w]`，已对齐到机器人朝向。编码器看到的是它相对机器人基座
  （或朝向）坐标系的 6D 旋转。
- `vr_3point_*` 是左手腕、右手腕和头部，在参考根坐标系中。四元数为 `[x, y, z, w]`；控制器把它们转换为编码器
  训练时使用的 `[w, x, y, z]`。

`reference.extra` 选项：

| 键 | 默认值 | 作用 |
|---|---|---|
| `encoder_mode` | 构造时的取值 | 本 tick 的模式 |
| `play` | `True` | `False` 时每次采样都取第 0 帧，并把速度置零（部署中的暂停状态） |

当前模式不用的项填零。模式需要而 `frames` 缺少的键会抛出 `KeyError`。部署在这种情况下则会静默退回到另一种模式。

### 解码器输入和输出

解码器输入是 token，后接 10 个 tick 的历史（最旧的在前）：基座角速度（3）、`dof_pos - default_pos`（29）、
`dof_vel`（29）、上一个动作（29）和投影重力（3），共 `64 + 10 * 93 = 994` 个值。每次 `step` 在推理之前记录
一条，所以其中的上一个动作是上一个 tick 的解码器原始输出。历史填满之前用零填充，重力除外：为与部署一致，
重力用 `[0, 0, 1]` 填充。

每次 `step` 返回 `JointTarget(position, kp, kd)`，`extra["motion_token"]` 设为所用的 token。`ctrl.token` 保存
同一个数组。`reset()` 清除历史和 token。

### 行走 planner

`hiwbc.controllers.sonic_planner.SonicPlanner(planner_fn, *, qpos_joint_index,
lower_body_index, default_height=0.78874, control_dt=0.02, look_ahead=2,
seed=1234, background=False)` 移植了 NVIDIA C++ 部署中的 planner（规划器）路径（`localmotion_kplanner.hpp`）。
`sonic/unitree_g1` profile 会构建它；只有用于其他机器人时才需要自己构建。

| 名字 | 说明 |
|---|---|
| `load_planner_runner(path, *, providers=None)` | 从 `planner_sonic.onnx` 得到 `planner_fn`；文件不是 V2 planner 时抛出 `ValueError` |
| `start(joint_pos)` | 由测得的关节生成第一个规划，处于空闲状态。`VELOCITY` 参考在另一种参考之后到来时，控制器会调用它 |
| `set_command(mode, speed, movement_yaw, facing_yaw, height=-1.0)` | 请求一个规划（`LocomotionMode`，planner 坐标系下的 yaw 角）；在下一次 10 Hz 检查时生效 |
| `velocity_command(planner, velocity, base_quat_wxyz, *, dt)` | `[vx, vy, wz]` -> `set_command`，与手柄的做法相同（见指南） |
| `window(length, base_quat_wxyz) -> dict` | 从当前帧起接下来 `length` 帧的 `joint_pos`、`joint_vel`、`root_quat`（`[x, y, z, w]`）、`root_z`；`start` 之后的第一次调用会把动作的朝向对齐到机器人 |
| `advance(joint_pos)` | policy 运行之后的一个 tick：混合已完成的规划，检查是否要重新规划，播放下一帧 |
| `replans`, `command`, `frame` | 计数器和当前生效的命令 |

`background=True`（profile 中的 `planner_thread=True`）在工作线程上运行每次重新规划；结果混合进来之前，当前
动作继续播放。抛出 `TimeoutError`（远程推理）的重新规划会在下一次检查时重试。

## echo

`EchoController`（`hiwbc.controllers.echo`），注册名为 `echo`。它接受所有 `ReferenceKind`，忽略观测，返回
`default_pos` 加上可选的固定 `residual`。它不需要模型，用于测试连线。

```python
make_controller("echo", num_actions=29, default_pos=pose, residual=None)
```

## 数学工具

`hiwbc.math_utils`，纯 numpy。所有四元数都是 `[x, y, z, w]`。

| 函数 | 返回值 |
|---|---|
| `gravity_orientation(q)` | 基座坐标系下的投影重力（`unitree_loco`） |
| `quat_rotate_inverse(q, v)` | 用 `q` 的逆旋转 `v`；`v = [0, 0, -1]` 时即投影重力（跟踪控制器、SONIC） |
| `quat_to_euler(q)` | `[roll, pitch, yaw]`，rad |
| `matrix_from_quat(q)` | `(3, 3)` 旋转矩阵 |
| `quat_mul(a, b)` | Hamilton 积 `a * b` |
| `quat_conjugate(q)` | 共轭（单位四元数的逆） |
| `subtract_frame_transforms_rot(q01, q02)` | 相对旋转 `q01⁻¹ * q02` |

## 编写控制器

1. 继承 `Controller`。设置 `accepts`，并把 `num_actions`、`default_pos` 和后处理参数传给 `super().__init__`。
   训练时没有动作平滑的 policy 用 `action_beta=1.0`。
2. 实现 `observe`。严格按 policy 训练时的顺序拼接各项；多输入模型返回元组。上一个动作从 `self._last_action`
   读取。
3. 实现 `forward`，调用注入的 `forward_fn`。不要在控制器内部加载 checkpoint。
4. 如果控制器维护时钟或历史，重写 `reset()`（调用 `super().reset()`），并在 `step` 中 `super().step(...)` 之后
   推进时钟。
5. 在 `hiwbc/controllers/` 中注册工厂函数，并在 `hiwbc/controllers/__init__.py` 中导入其模块：

   ```python
   @register_controller("my_tracker")
   def _build(**kwargs):
       return MyTrackerController(**kwargs)
   ```

6. 为它运行的每台机器人添加一个 profile：`hiwbc/configs/<controller>/<robot>.py` 调用 `register_profile`，
   其 `build(variant, options)` 加载 checkpoint，并传入 `joint_names`、`kp`、`kd` 和训练取值。
7. 可选：设置 `motion_spaces` 并实现 `check_clip` / `clip_reference` 以接收动作片段，或者为 checkpoint 中的动作
   实现 `behaviors` / `start_behavior`。
8. 添加一个测试：通过桩 `forward_fn` 输入已知的状态值，逐项断言观测布局。

> 观测顺序关乎安全。即使输入的项顺序或缩放错了，policy 仍然会输出动作，用这些无意义的动作去指挥机器人。
> 新控制器在上真机之前先在仿真中检查。

# Getting started

English | [简体中文](./getting_started.zh-CN.md)

This guide covers what every controller needs, whichever policy it runs: the
control loop, building `ControllerState` from the robot, mapping joint order,
wrapping a checkpoint as a forward function, PD gains, and first power-on. The
per-family guides ([locomotion](locomotion.md),
[motion tracking](motion_tracking.md), [SONIC](sonic.md)) build on it. To let
hiRobot run the loop instead of writing it yourself, see
[Running as a hiRobot robot](robot.md).

## 1. Install

```bash
git clone https://github.com/humanoidintelligence/hiWBC.git && cd hiWBC
pip install -e .            # core: numpy + hiRobot (the controllers import only numpy)
pip install -e '.[torch]'   # TorchScript checkpoints (.pt)
pip install -e '.[onnx]'    # ONNX checkpoints (.onnx)
python -m pytest            # stub models, no GPU or weights needed
```

## 2. The control loop

A controller is a pure step: `(ControllerState, Reference) -> JointTarget`. It
never talks to hardware. The caller, usually hiRobot, runs this loop at the
policy's control rate (50 Hz for every built-in controller):

```python
import time
import numpy as np
import hiwbc
from hiwbc import ControllerState, Reference, ReferenceKind

ctrl = hiwbc.load("unitree_loco/unitree_g1")   # or make_controller(...), see the family guides
ctrl.reset()                                    # once per episode

DT = 0.02                                       # 50 Hz
next_tick = time.perf_counter()
while running:
    raw = robot.read()                          # your robot interface
    state = ControllerState(
        dof_pos=raw.q[policy_idx],              # policy joint order (section 4)
        dof_vel=raw.dq[policy_idx],
        base_quat=raw.imu_quat_xyzw,            # [x, y, z, w]
        base_ang_vel=raw.imu_gyro,              # body frame, rad/s
    )
    ref = Reference(ReferenceKind.VELOCITY, value=np.array([0.5, 0.0, 0.0]))
    target = ctrl.step(state, ref)

    q_cmd = default_full_body.copy()            # joints the policy does not drive
    q_cmd[policy_idx] = target.position         # back to robot order
    kp[policy_idx], kd[policy_idx] = target.kp, target.kd   # gains, when the controller has them (section 6)
    robot.send_position(q_cmd, kp=kp, kd=kd)    # the robot's PD loop tracks it

    next_tick += DT
    time.sleep(max(0.0, next_tick - time.perf_counter()))
```

Each `step` does `observe` → `forward` → post-processing (EMA, clip, scale,
plus default pose), and returns an absolute PD position target. Controllers that
keep a clock or a history (gait phase, motion phase, proprio history) advance it
inside `step`, so call `step` exactly once per tick and `reset()` at the start
of every episode.

## 3. Building `ControllerState`

| Field | What to put in it | Common mistake |
|---|---|---|
| `dof_pos`, `dof_vel` | Measured joint positions / velocities, in the **policy's** joint order | Passing the robot's native order |
| `base_quat` | Pelvis (IMU) orientation in the world frame, **`[x, y, z, w]`** | Unitree SDK and MuJoCo give `[w, x, y, z]`: reorder with `q[[1, 2, 3, 0]]` |
| `base_ang_vel` | Pelvis angular velocity in the **body** frame, rad/s | Using a world-frame value. The IMU gyro is already body frame |
| `base_lin_vel` | Body-frame linear velocity, only if a state estimator provides it | Read only by the `beyondmimic_tracker` checkpoints that use a state estimator (`violin`, `waltz`) |
| `extra` | Optional extras. `torso_quat` (`[x, y, z, w]`): the torso orientation, if you measure it (BeyondMimic; otherwise computed from the waist joints) | |

On a Unitree G1 with the IMU on the pelvis, `base_quat` and `base_ang_vel` come
straight from `low_state.imu_state` (reorder the quaternion). In MuJoCo, they are
`qpos[3:7]` (reordered) and `qvel[3:6]` of the free joint.

## 4. Joint order

Every policy was trained with its own joint order and joint subset, which
usually differ from the robot's. `dof_pos`, `dof_vel`, `default_pos` and
`JointTarget.position` are all in the policy's order. A profile-built controller
lists it in `ctrl.joint_names` (hiRobot names, no `_joint` suffix), and the
joints `dof_pos` must carry in `ctrl.obs_joint_names`. Map by joint name, once,
at start-up:

```python
robot_joints = [...]       # the robot's order, e.g. Unitree SDK / MuJoCo order
policy_joints = ctrl.joint_names   # the order the policy was trained with
policy_idx = np.array([robot_joints.index(n) for n in policy_joints])

state_q = raw.q[policy_idx]              # robot -> policy
q_cmd[policy_idx] = target.position      # policy -> robot
```

Joints the policy does not drive (for example the wrists under a 23-DoF policy)
hold a fixed pose: send them their default position with the gains the policy's
config gives them.

The orders the built-in examples use are in each family guide. Two traps:

- For the 29-DoF G1, Unitree SDK / MuJoCo order (`left_hip_pitch, left_hip_roll,
  ...`) differs from IsaacLab order (`left_hip_pitch, right_hip_pitch,
  waist_yaw, ...`). IsaacLab-trained policies (BeyondMimic, unitree_rl_lab,
  SONIC) use IsaacLab order.
- AMO observes 23 joints but drives only the first 15. The actuated joints must
  lead the observed ones.

## 5. Forward functions

Controllers never load a checkpoint. They take a `forward_fn(obs) -> action`,
so tests can pass a stub and you can use any runtime. A profile
(`hiwbc.load`) builds the forward function for you; the rest of this section is
for building a controller by hand.

TorchScript (`.pt`):

```python
import numpy as np
import torch

model = torch.jit.load("policy.pt", map_location="cpu").eval()

@torch.no_grad()
def forward_fn(obs):
    return model(torch.from_numpy(np.asarray(obs, np.float32))[None]).numpy().squeeze()
```

ONNX (`.onnx`), single input:

```python
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession("policy.onnx", providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0].name

def forward_fn(obs):
    return sess.run(None, {inp: np.asarray(obs, np.float32)[None]})[0].squeeze()
```

Check a checkpoint's input names and sizes before wiring it up; a size mismatch
is the first sign of a wrong observation layout:

```python
print([(i.name, i.shape) for i in sess.get_inputs()])
print([o.name for o in sess.get_outputs()])
```

> Recurrent TorchScript policies (for example `policy_lstm_1.pt`) keep their
> hidden state inside the model object. Use one model object per controller.
> `hiwbc.load_torchscript_runner` snapshots the buffers and `ctrl.reset()`
> restores them; a hand-written `forward_fn` must reset the hidden state itself.

Multi-input models (`kungfubot_tracker`, `supermimic`, `amo_loco`) receive a
tuple; the family guides show the mapping to model inputs.

## 6. PD gains

`JointTarget.position` is a position target for the robot's joint PD loop. The
gains belong to the policy: use the stiffness and damping the policy was trained
with. A controller built by a profile returns them in `JointTarget.kp` / `kd`
(from RoboJuDo's configs, the ONNX metadata, or the SONIC / SuperMimic
configs); send them with the target. A controller built with `make_controller`
without `kp` / `kd` returns `None`, and you configure the gains on the robot
side (each family guide lists them).

Wrong gains are a common reason a correctly wired policy falls: a policy trained
at kp 100 on the hips will not stand at kp 40.

## 7. First power-on

1. **Simulation first.** Run the full loop in MuJoCo with the same joint
   mapping, gains and control rate you will use on hardware.
2. **Hang or support the robot**, with the area clear and someone at the e-stop.
3. **Ramp to the default pose** from the measured pose over 2 to 3 s, with low
   gains rising to the policy's gains. Never let the first command jump to a
   target. hiRobot's `reset()` does this.
4. **Then call `ctrl.reset()` and start stepping.** The controller's histories
   and clocks start from that moment; a stale history from an earlier run makes
   the first actions wrong.
5. Stand still first (zero velocity command, or a motion's first frame), then
   lower the robot to the ground.

## 8. Common pitfalls

| Symptom | Likely cause |
|---|---|
| `forward returned shape (k,), expected (n,)` | `num_actions` does not match the checkpoint |
| ONNX "invalid dimensions" on the input | Wrong observation size: check `history_length`, `use_history`, `n_priv_latent`, the `frames` terms, `without_state_estimator` |
| Robot falls within 1 s, policy looks fine in its own repo | Joint order, quaternion order (`wxyz` vs `xyzw`), gains, or a history layout mismatch |
| Robot leans steadily one way | `default_pos` not in policy joint order, or a joint sign flipped in the mapping |
| Jerky first steps | `reset()` not called before the first `step`, or the robot did not start from the default pose |
| A tracker drifts after a few seconds | Reference terms computed in the wrong frame upstream (anchor, heading alignment) |

# Contributing to hiWBC

hiWBC puts learned whole-body controllers behind one contract. Most
contributions add a controller for a released policy, a profile for another
robot or checkpoint, or a parity check of an existing controller. This page
says what each kind of change needs so it can be reviewed quickly.

## Set up

hiWBC depends on [hiRobot](https://github.com/humanoidintelligence/hiRobot).
Install it first (an editable checkout works), then:

```bash
git clone https://github.com/humanoidintelligence/hiWBC.git && cd hiWBC
pip install -e '.[dev]'          # add '.[onnx]' / '.[torch]' to run real checkpoints
python -m pytest                 # run from the repo root; no GPU, weights or robot needed
ruff check src tests             # line length 100, target py39
```

Every controller takes an injected `forward_fn(obs) -> action`, and the tests
pass a stub, so the full suite runs on any machine.

## What a change needs

| Change | Add | Don't touch |
|---|---|---|
| New checkpoint of a supported policy | A variant in its profile, `src/hiwbc/configs/<controller>/<robot>.py`, with the values that checkpoint was trained with; a line in the guide | Controller code, unless the observation changed |
| Supported controller on a new robot | A profile `src/hiwbc/configs/<controller>/<robot>.py`: joint names, default pose, gains, scales | Controller code |
| New controller | One file in `src/hiwbc/controllers/`, a profile, a test that pins the observation layout by value, a row in the README's controller table and a section in the matching guide | `WbcRobot`, other controllers, hiRobot |
| Core (`types.py`, `controller.py`, `registry.py`, `robot.py`, `remote.py`) | Open an issue first. Every controller and the `wbc` layer depend on these | |

The README's [Add a new controller](README.md#add-a-new-controller) has the
steps. In short:

- Subclass `Controller`, set `accepts`, and build the observation in `observe`.
  Leave EMA, clip, scale and the default-pose offset to the base class; a
  policy trained without smoothing uses `action_beta=1.0`.
- Never load a checkpoint inside a controller. The profile's `build` turns the
  file into a `forward_fn` with `load_onnx_runner` / `load_torchscript_runner`.
- Import only numpy at module level. `torch` and `onnxruntime` are imported
  inside the function that loads a checkpoint (`tests/test_runtime.py` checks
  that `import hiwbc` stays numpy-only).
- Quaternions are `[x, y, z, w]`. Convert at the boundary if the policy was
  trained with `[w, x, y, z]`.
- Body-frame transforms and forward kinematics belong to the caller or hiRobot;
  a tracker consumes per-tick terms from `reference.frames`.
- A released controller or profile name never changes meaning. An incompatible
  checkpoint gets a new variant or a new name.

## Observation order and parity

Observation field order is safety-critical. A wrong order or a wrong history
layout (term-major vs frame-major, newest vs oldest first) often still fits the
input size, and the robot falls within about a second. So:

- Follow the policy's own deployment code (or the reference deployment it was
  checked against) term for term. Name that code, with a commit, in the pull
  request.
- Pin the layout by value in a test, not only its shape:
  `tests/test_history_layout.py` shows how.
- For a new controller or a change to an observation, history or
  post-processing, run it against the reference deployment on the same inputs
  and report the largest difference per tick (the built-in controllers match
  to about 1e-5). Say in the pull request how you ran it.

## Working on real robots

CI cannot run hardware. We tested the policies RoboJuDo deploys on a real robot
earlier, and their controllers match RoboJuDo in MuJoCo; this release,
including `sonic`, `supermimic` and the `wbc` layer, has been run in simulation
only. A pull
request that changes what reaches the motors (post-processing, gains,
`WbcRobot`) must say how it was checked. Run in simulation first, through
hiRobot's MuJoCo robot with the `wbc` layer, then follow hiRobot's setup guide:
read-only, then the robot on a gantry at low gains, with the e-stop in reach.
Don't put lab IP addresses, network interfaces or local paths into shared
files.

## Pull requests

- One topic per pull request. Keep refactors separate from behavior changes.
- `python -m pytest` and `ruff check src tests` pass.
- Update the docs your change makes wrong: the README tables, `docs/api.md`,
  the guide for that controller family.
- Commit messages: a short summary line, then what changed and why, with the
  test count.
- Don't commit checkpoints. Profiles load them from a directory given by the
  `checkpoint_dir` option; say in the guide where the weights come from and
  under which license.

## Reporting problems

Open a GitHub issue with the controller and profile (`<controller>/<robot>`,
variant), where the checkpoint came from, the hiWBC and hiRobot commits, and the
full error. For a robot that falls or shakes, say whether it was in simulation
or on hardware and what it did.

## License

hiWBC is licensed under [Apache-2.0](LICENSE). By contributing you agree that
your contribution is licensed under the same terms.

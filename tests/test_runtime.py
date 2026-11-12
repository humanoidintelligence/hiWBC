"""Tests for hiwbc.runtime: model loaders, joint names, and the numpy-only import."""

import subprocess
import sys

import numpy as np
import pytest

from hiwbc import load_onnx_runner, load_torchscript_runner
from hiwbc.configs.sonic.unitree_g1 import (
    G1_JOINT_NAMES,
    G1_JOINT_NAMES_MUJOCO,
    ISAACLAB_TO_MUJOCO_INDEX,
)


def test_import_hiwbc_loads_no_model_runtime():
    code = (
        "import sys, hiwbc, hiwbc.runtime, hiwbc.controllers.sonic, hiwbc.controllers.supermimic, "
        "hiwbc.configs.sonic.unitree_g1, "
        "hiwbc.controllers.amo; "
        "print(','.join(m for m in ('torch', 'onnxruntime') if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""


# ----- joint names: hiRobot's DofMap does the reordering ------------------


def test_g1_names_match_hirobot_and_dofmap_gives_sonic_permutation():
    """SONIC's G1 joint names are hiRobot's, so DofMap reproduces the deployment's permutation."""
    hirobot = pytest.importorskip("hirobot")

    emb = hirobot.load_embodiment("unitree_g1_mujoco")
    assert tuple(emb.joint_names) == G1_JOINT_NAMES_MUJOCO
    dm = hirobot.DofMap(emb.joint_names, G1_JOINT_NAMES)
    np.testing.assert_array_equal(dm.remap(np.arange(len(emb.joint_names))), ISAACLAB_TO_MUJOCO_INDEX)


def test_supermimic_names_are_hirobot_names():
    from hiwbc.controllers.supermimic import SUPERMIMIC_JOINT_NAMES

    assert SUPERMIMIC_JOINT_NAMES == G1_JOINT_NAMES_MUJOCO


# ----- loaders (build tiny models on the fly) -------------------------------


def _tiny_onnx(path, inputs, out_dim=3):
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    nodes, vis = [], []
    names = []
    for name, dim in inputs:
        vis.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, dim]))
        names.append(name)
    # sum of each input's first out_dim entries
    nodes.append(helper.make_node("Concat", names, ["cat"], axis=1))
    starts = helper.make_tensor("s", TensorProto.INT64, [1], [0])
    ends = helper.make_tensor("e", TensorProto.INT64, [1], [out_dim])
    axes = helper.make_tensor("a", TensorProto.INT64, [1], [1])
    nodes.append(helper.make_node("Slice", ["cat", "s", "e", "a"], ["action"]))
    graph = helper.make_graph(
        nodes, "g", vis, [helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, out_dim])],
        initializer=[starts, ends, axes],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return path


def test_load_onnx_runner_single_and_multi_input(tmp_path):
    pytest.importorskip("onnxruntime")
    p1 = _tiny_onnx(tmp_path / "one.onnx", [("obs", 5)])
    f = load_onnx_runner(p1)
    np.testing.assert_allclose(f(np.arange(5.0)), [0, 1, 2])

    p2 = _tiny_onnx(tmp_path / "two.onnx", [("a", 2), ("b", 4), ("t", 1)])
    g = load_onnx_runner(p2, input_names=("b", "a"), extra_inputs={"t": np.zeros((1, 1))})
    # concat order is the model's (a, b, t): a=[9, 8], b=[...]
    np.testing.assert_allclose(g((np.arange(4.0), np.array([9.0, 8.0]))), [9, 8, 0])
    with pytest.raises(ValueError, match="expected 2 input"):
        g(np.zeros(4))
    with pytest.raises(ValueError, match="no model input"):
        load_onnx_runner(p2, input_names=("zz",))


def test_load_torchscript_runner(tmp_path):
    torch = pytest.importorskip("torch")

    class Two(torch.nn.Module):
        def forward(self, x, y):
            return x[:, :2] + y[:, :2]

    p = tmp_path / "m.pt"
    torch.jit.script(Two()).save(str(p))
    f = load_torchscript_runner(p)
    np.testing.assert_allclose(f((np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0]))), [11, 22])

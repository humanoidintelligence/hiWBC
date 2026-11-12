"""Model loaders: turn a checkpoint file into the ``forward_fn`` a controller takes.

Joint reordering between a robot and a controller is hiRobot's ``DofMap``; hiWBC
no longer has its own.

``torch`` and ``onnxruntime`` are imported inside the loader that needs them, so
``import hiwbc`` stays numpy-only.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

__all__ = ["load_onnx_runner", "load_torchscript_runner"]


def load_onnx_runner(
    path: str,
    *,
    input_names: Sequence[str] | None = None,
    output_name: str | None = None,
    extra_inputs: dict[str, np.ndarray] | None = None,
    providers: Sequence[str] | None = None,
    num_threads: int | None = None,
) -> Callable:
    """Build a ``forward_fn`` from an ONNX file.

    The returned function takes one array, or a tuple of arrays for a model with
    several inputs (matched to ``input_names`` in order, default: the model's own
    input order). Each array is cast to float32 and reshaped to ``(1, -1)``. It
    returns the first output, or ``output_name``, flattened.

    Args:
        path: The ``.onnx`` file.
        input_names: Model inputs fed from the controller's observation, in the
            order ``observe`` returns them. Default: all inputs not in
            ``extra_inputs``, in model order.
        output_name: Output to return. Default: the model's first output.
        extra_inputs: Constant inputs fed on every call (e.g. BeyondMimic's
            ``time_step``), name -> array.
        providers: onnxruntime execution providers. Default: CPU.
        num_threads: If set, intra- and inter-op thread counts. One thread gives
            the steadiest latency for these small models.

    Requires the ``onnx`` extra (``onnxruntime``).
    """
    import onnxruntime as ort

    opts = ort.SessionOptions()
    if num_threads is not None:
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = num_threads
    session = ort.InferenceSession(
        str(path), sess_options=opts,
        providers=list(providers) if providers else ["CPUExecutionProvider"],
    )
    const = {k: np.asarray(v, np.float32) for k, v in (extra_inputs or {}).items()}
    model_inputs = [i.name for i in session.get_inputs()]
    names = list(input_names) if input_names else [n for n in model_inputs if n not in const]
    unknown = [n for n in [*names, *const] if n not in model_inputs]
    if unknown:
        raise ValueError(f"{path}: no model input named {unknown}; inputs are {model_inputs}")
    out = output_name or session.get_outputs()[0].name

    def run(obs) -> np.ndarray:
        parts = obs if isinstance(obs, tuple) else (obs,)
        if len(parts) != len(names):
            raise ValueError(f"expected {len(names)} input(s) {names}, got {len(parts)}")
        feed = {n: np.asarray(x, np.float32).reshape(1, -1) for n, x in zip(names, parts)}
        feed.update(const)
        return session.run([out], feed)[0].reshape(-1)

    return run


def load_torchscript_runner(path: str, *, device: str = "cpu") -> Callable:
    """Build a ``forward_fn`` from a TorchScript (``torch.jit``) file.

    The returned function takes one array, or a tuple of arrays passed as
    separate positional arguments, adds a batch dimension, runs the model under
    ``torch.no_grad()`` and returns the output flattened. Its ``reset()``
    restores the model's buffers to their as-loaded values (the hidden state of
    a recurrent export); ``Controller.reset()`` calls it.

    Requires the ``torch`` extra.
    """
    import torch

    model = torch.jit.load(str(path), map_location=device).eval()
    # Recurrent exports (e.g. unitree's policy_lstm_1.pt) keep their hidden state in
    # buffers that the forward pass overwrites. Snapshot them so reset() restores the
    # as-loaded state, the same as reloading the file.
    initial = {name: buf.detach().clone() for name, buf in model.named_buffers()}

    def run(obs) -> np.ndarray:
        parts = obs if isinstance(obs, tuple) else (obs,)
        args = [torch.from_numpy(np.asarray(x, np.float32).reshape(1, -1)).to(device) for x in parts]
        with torch.no_grad():
            y = model(*args)
        return y.detach().cpu().numpy().reshape(-1)

    def reset() -> None:
        with torch.no_grad():
            for name, buf in model.named_buffers():
                buf.copy_(initial[name])

    run.reset = reset
    run.stateful = bool(initial)
    return run


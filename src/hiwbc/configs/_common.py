"""Helpers shared by profile modules: checkpoint paths and option handling."""

from __future__ import annotations

import os
from pathlib import Path

#: Default root for RoboJuDo-released G1 checkpoints. Users override per machine with
#: the profile's ``checkpoint_dir`` option or the ``HIWBC_ROBOJUDO_MODELS`` variable.
ROBOJUDO_MODELS = os.environ.get("HIWBC_ROBOJUDO_MODELS", "~/.cache/hiwbc/robojudo/g1")


def checkpoint(directory: str | os.PathLike, *parts: str) -> Path:
    """``directory / parts``, with ``~`` expanded; raise if the file is missing."""
    path = Path(directory).expanduser().joinpath(*parts)
    if not path.is_file():
        raise FileNotFoundError(
            f"checkpoint not found: {path} (set the profile's checkpoint_dir option)"
        )
    return path


def strip_joint(names) -> tuple[str, ...]:
    """RoboJuDo / IsaacLab names (``left_hip_pitch_joint``) -> hiRobot names (``left_hip_pitch``)."""
    return tuple(n.removesuffix("_joint") for n in names)

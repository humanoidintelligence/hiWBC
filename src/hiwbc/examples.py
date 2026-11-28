"""Robot configs shipped with hiWBC: hiRobot configs with a ``wbc`` layer."""

from __future__ import annotations

from pathlib import Path

_DIR = Path(__file__).with_name("hirobot_configs") / "robots"


def robot_configs() -> dict[str, Path]:
    """Name -> path of each robot config shipped with hiWBC, sorted by name.

    Each is a hiRobot config file with a ``wbc`` layer, for ``make_robot(path)``.
    A hiRobot that reads the ``hirobot.configs`` entry points also finds them
    by name (``make_robot("g1_sonic")``).
    Opening one needs the extras and checkpoints its header comment lists.
    """
    return {p.stem: p for p in sorted(_DIR.glob("*.yaml"))}

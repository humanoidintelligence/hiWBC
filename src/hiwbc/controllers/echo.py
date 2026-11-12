"""A dependency-free controller for exercising the contract.

``EchoController`` needs no checkpoint, torch, or hardware. It returns the
default pose (optionally plus a fixed residual), so the step pipeline,
registry, and hiRobot integration can run end to end with no model. Analogous
to hiVLA's ``echo`` backend and hiAgent's ``echo`` planner.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..controller import Controller
from ..registry import register_controller
from ..types import ControllerState, Reference, ReferenceKind


class EchoController(Controller):
    """Holds the default pose; echoes a fixed residual if given one."""

    accepts = (ReferenceKind.VELOCITY, ReferenceKind.MOTION_FRAMES, ReferenceKind.LATENT)

    def __init__(
        self,
        *,
        num_actions: int,
        default_pos: Sequence[float],
        residual: Sequence[float] | None = None,
        **base,
    ) -> None:
        super().__init__(num_actions=num_actions, default_pos=default_pos, **base)
        if residual is None:
            self._residual = np.zeros(num_actions, dtype=np.float32)
        else:
            self._residual = np.asarray(residual, dtype=np.float32)
            if self._residual.shape != (num_actions,):
                raise ValueError(
                    f"residual has shape {self._residual.shape}, expected ({num_actions},)"
                )

    def observe(self, state: ControllerState, reference: Reference) -> np.ndarray:
        # The echo controller ignores the observation; return an empty vector.
        return np.zeros(0, dtype=np.float32)

    def forward(self, obs: np.ndarray) -> np.ndarray:
        return self._residual


@register_controller("echo")
def _build_echo(**kwargs) -> EchoController:
    return EchoController(**kwargs)

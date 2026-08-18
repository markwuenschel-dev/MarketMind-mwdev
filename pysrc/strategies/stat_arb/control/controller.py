from __future__ import annotations

from typing import Protocol


class StatArbController(Protocol):
    """Future controller contract. Live Phase I-D runtime must not depend on this."""

    def control_name(self) -> str:
        """Return the future controller name."""


def build_controller() -> None:
    raise NotImplementedError(
        "controller.py is a Phase I-Db stub. "
        "Invariant: control-layer execution is deferred beyond the live Phase I-D pairs runtime."
    )

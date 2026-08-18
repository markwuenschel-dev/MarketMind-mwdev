"""Kill switch: block new orders on reconciliation or risk failures."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class KillSwitchState:
    """Mutable kill-switch gate used by the execution shell."""

    block_new_orders: bool = False
    reason: str = ""
    _history: list[str] = field(default_factory=list, repr=False)

    def engage(self, reason: str) -> None:
        if not reason:
            raise ValueError("kill switch engage requires a non-empty reason")
        self.block_new_orders = True
        self.reason = reason
        self._history.append(reason)

    def release(self) -> None:
        self.block_new_orders = False
        self.reason = ""

    def allows_new_orders(self) -> bool:
        return not self.block_new_orders


__all__ = ["KillSwitchState"]

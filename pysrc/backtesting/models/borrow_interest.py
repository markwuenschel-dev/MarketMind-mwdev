from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BorrowInterestConfig:
    borrow_interest_bps: float = 0.0


class BorrowInterestModel:
    def __init__(self, config: BorrowInterestConfig | None = None) -> None:
        self.config = config or BorrowInterestConfig()

    def to_execution_assumptions(self) -> dict[str, float]:
        return {"borrow_interest_bps": self.config.borrow_interest_bps}

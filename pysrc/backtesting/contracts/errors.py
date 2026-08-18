from __future__ import annotations

from dataclasses import dataclass


class BacktestingContractError(RuntimeError):
    """Base error for backtesting substrate contract failures."""


class DeterminismTierMissingError(BacktestingContractError):
    """Raised when a backtest plan omits a required determinism tier."""


class PitUnsafeInputError(BacktestingContractError):
    """Raised when a PIT-required plan receives a non-PIT-safe input."""


class OptionalDependencyMissingError(BacktestingContractError):
    """Raised when an optional backend is invoked without its dependency."""


class NotImplementedLaneError(BacktestingContractError):
    """Raised when a scaffolded engine or lane cannot execute yet."""


@dataclass(frozen=True)
class UnknownIdError(BacktestingContractError):
    """Actionable unknown-id error for registry resolution."""

    component_kind: str
    requested_id: str
    available_ids: tuple[str, ...]
    hint: str

    def __str__(self) -> str:
        available = ", ".join(self.available_ids) if self.available_ids else "<none>"
        return (
            f"Unknown {self.component_kind} id {self.requested_id!r}. "
            f"Available ids: {available}. {self.hint}"
        )

"""BudgetManager: tracks and enforces trial budget and timeout limits."""

from __future__ import annotations

import time


class BudgetExhaustedError(RuntimeError):
    """Raised when the trial budget or timeout is exceeded."""


class BudgetManager:
    """Tracks consumed budget against declared limits."""

    def __init__(self, max_trials: int, timeout_seconds: int | None) -> None:
        self._max_trials = max_trials
        self._timeout_seconds = timeout_seconds
        self._trials_used = 0
        self._start_time = time.monotonic()

    def consume(self, n: int = 1) -> None:
        """Consume n trials; raise BudgetExhaustedError if limit exceeded."""
        self._trials_used += n
        if self._trials_used > self._max_trials:
            raise BudgetExhaustedError(
                f"Trial budget exhausted: {self._trials_used} > {self._max_trials}"
            )
        if self._timeout_seconds is not None:
            elapsed = time.monotonic() - self._start_time
            if elapsed > self._timeout_seconds:
                raise BudgetExhaustedError(
                    f"Timeout exceeded: {elapsed:.1f}s > {self._timeout_seconds}s"
                )

    @property
    def remaining(self) -> int:
        """Return the number of remaining trials in the budget."""
        return max(0, self._max_trials - self._trials_used)


__all__ = ["BudgetExhaustedError", "BudgetManager"]

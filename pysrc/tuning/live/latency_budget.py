"""LatencyBudget: enforces per-inference latency constraints."""

from __future__ import annotations

import time


class LatencyBudgetExceededError(RuntimeError):
    """Raised when an operation exceeds its declared latency budget."""


class LatencyBudget:
    """Context manager that raises LatencyBudgetExceededError if budget_ms is exceeded."""

    def __init__(self, budget_ms: int) -> None:
        if budget_ms < 1:
            raise ValueError("budget_ms must be >= 1")
        self._budget_ns = budget_ms * 1_000_000
        self._start: int = 0

    def __enter__(self) -> LatencyBudget:
        self._start = time.monotonic_ns()
        return self

    def __exit__(self, *_: object) -> None:
        elapsed = time.monotonic_ns() - self._start
        if elapsed > self._budget_ns:
            raise LatencyBudgetExceededError(
                f"Operation took {elapsed / 1e6:.2f}ms, "
                f"exceeding budget of {self._budget_ns / 1e6:.0f}ms"
            )

    def elapsed_ms(self) -> float:
        """Return elapsed milliseconds since __enter__."""
        return (time.monotonic_ns() - self._start) / 1e6


__all__ = ["LatencyBudgetExceededError", "LatencyBudget"]

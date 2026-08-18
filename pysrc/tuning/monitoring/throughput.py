"""ThroughputMonitor: tracks trial and task throughput rates."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class ThroughputReport:
    """Throughput statistics over the observation window."""

    trials_per_second: float
    tasks_per_second: float
    observation_seconds: float


class ThroughputMonitor:
    """Tracks trial and task completion counts over a wall-clock window."""

    def __init__(self) -> None:
        self._trials = 0
        self._tasks = 0
        self._start = time.monotonic()

    def record_trial(self, n: int = 1) -> None:
        """Increment the completed trial count."""
        self._trials += n

    def record_task(self, n: int = 1) -> None:
        """Increment the completed task count."""
        self._tasks += n

    def report(self) -> ThroughputReport:
        """Return throughput rates over elapsed wall-clock time."""
        elapsed = max(time.monotonic() - self._start, 1e-9)
        return ThroughputReport(
            trials_per_second=self._trials / elapsed,
            tasks_per_second=self._tasks / elapsed,
            observation_seconds=elapsed,
        )


__all__ = ["ThroughputReport", "ThroughputMonitor"]

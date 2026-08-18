"""LatencyMonitor: records and reports inference/execution latency histograms."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LatencyReport:
    """Summary latency statistics in milliseconds."""

    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float


class LatencyMonitor:
    """Accumulates latency samples and produces summary reports."""

    def __init__(self) -> None:
        self._samples: list[float] = []

    def record(self, latency_ms: float) -> None:
        """Record a single latency sample in milliseconds."""
        self._samples.append(latency_ms)

    def report(self) -> LatencyReport:
        """Return summary latency statistics over all recorded samples."""
        if not self._samples:
            return LatencyReport(0.0, 0.0, 0.0, 0.0)
        s = sorted(self._samples)
        n = len(s)

        def pct(p: float) -> float:
            idx = min(int(p * n), n - 1)
            return s[idx]

        return LatencyReport(
            p50_ms=pct(0.50),
            p95_ms=pct(0.95),
            p99_ms=pct(0.99),
            max_ms=s[-1],
        )


__all__ = ["LatencyReport", "LatencyMonitor"]

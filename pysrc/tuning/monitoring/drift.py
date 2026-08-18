"""DriftMonitor: tracks feature and prediction drift for live strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class DriftReport:
    """Immutable snapshot of drift metrics at a point in time."""

    job_id: str
    feature_hash: str
    drift_score: float
    threshold: float
    is_drifting: bool
    observed_at: datetime
    details: dict[str, float] = field(default_factory=dict)


class DriftMonitor:
    """Accumulates drift observations and emits DriftReports."""

    def __init__(self, job_id: str, threshold: float) -> None:
        self._job_id = job_id
        self._threshold = threshold

    def observe(
        self,
        drift_score: float,
        feature_hash: str,
        now: datetime,
        details: dict[str, float] | None = None,
    ) -> DriftReport:
        """Record a drift observation and return a DriftReport."""
        return DriftReport(
            job_id=self._job_id,
            feature_hash=feature_hash,
            drift_score=drift_score,
            threshold=self._threshold,
            is_drifting=drift_score >= self._threshold,
            observed_at=now,
            details=details or {},
        )


__all__ = ["DriftReport", "DriftMonitor"]

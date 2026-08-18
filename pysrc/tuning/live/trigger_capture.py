"""TriggerCapture: detects drift events and emits retrain triggers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class DriftTrigger:
    """An immutable record of a detected drift event."""

    job_id: str
    drift_score: float
    threshold: float
    detected_at: datetime
    feature_hash: str


class TriggerCapture:
    """Watches drift scores and emits DriftTriggers when the threshold is crossed."""

    def __init__(self, threshold: float, job_id: str) -> None:
        self._threshold = threshold
        self._job_id = job_id

    def evaluate(
        self,
        drift_score: float,
        feature_hash: str,
        now: datetime,
    ) -> DriftTrigger | None:
        """Return a DriftTrigger if drift_score exceeds threshold, else None."""
        if drift_score >= self._threshold:
            return DriftTrigger(
                job_id=self._job_id,
                drift_score=drift_score,
                threshold=self._threshold,
                detected_at=now,
                feature_hash=feature_hash,
            )
        return None


__all__ = ["DriftTrigger", "TriggerCapture"]

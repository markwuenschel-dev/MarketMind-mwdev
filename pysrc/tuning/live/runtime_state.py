"""RuntimeState: mutable snapshot of the live system's active model and configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class RuntimeState:
    """Mutable runtime state for the active live strategy."""

    job_id: str
    active_artifact_hash: str
    activated_at: datetime
    shadow_candidate_id: str | None = None
    last_drift_score: float = 0.0
    last_retrain_at: datetime | None = None
    tags: dict[str, str] = field(default_factory=dict)

    def update_drift(self, score: float) -> None:
        """Update the last observed drift score."""
        self.last_drift_score = score

    def record_retrain(self, now: datetime) -> None:
        """Record the timestamp of the most recent retrain."""
        self.last_retrain_at = now


__all__ = ["RuntimeState"]

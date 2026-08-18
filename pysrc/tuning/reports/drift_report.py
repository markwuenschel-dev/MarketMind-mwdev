"""DriftReport: live drift metrics rendered from monitoring artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class DriftReport:
    """Summary of drift conditions observed over a monitoring window."""

    job_id: str
    feature_hash: str
    mean_drift_score: float
    max_drift_score: float
    n_triggers: int
    window_start: datetime
    window_end: datetime
    details: dict[str, float] = field(default_factory=dict)


def render_drift_report(artifact: dict[str, Any]) -> DriftReport:
    """Construct a DriftReport from a raw artifact payload dict."""
    return DriftReport(
        job_id=artifact["job_id"],
        feature_hash=artifact["feature_hash"],
        mean_drift_score=float(artifact["mean_drift_score"]),
        max_drift_score=float(artifact["max_drift_score"]),
        n_triggers=int(artifact.get("n_triggers", 0)),
        window_start=datetime.fromisoformat(artifact["window_start"]),
        window_end=datetime.fromisoformat(artifact["window_end"]),
        details=dict(artifact.get("details", {})),
    )


__all__ = ["DriftReport", "render_drift_report"]

"""RobustnessReport: stability and regime-robustness metrics rendered from artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RobustnessReport:
    """Stability and robustness summary for a candidate."""

    job_id: str
    candidate_id: str
    fold_score_variance: float
    stability_score: float
    pbo_score: float | None
    regime_breakdown: dict[str, float]


def render_robustness_report(artifact: dict[str, Any]) -> RobustnessReport:
    """Construct a RobustnessReport from a raw artifact payload dict."""
    return RobustnessReport(
        job_id=artifact["job_id"],
        candidate_id=artifact["candidate_id"],
        fold_score_variance=float(artifact.get("fold_score_variance", 0.0)),
        stability_score=float(artifact.get("stability_score", 1.0)),
        pbo_score=float(artifact["pbo_score"]) if "pbo_score" in artifact else None,
        regime_breakdown=dict(artifact.get("regime_breakdown", {})),
    )


__all__ = ["RobustnessReport", "render_robustness_report"]

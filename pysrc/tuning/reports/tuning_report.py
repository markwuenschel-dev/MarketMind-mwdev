"""TuningReport: summary of a completed tuning run rendered from artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TuningReport:
    """Human-readable summary of a tuning job."""

    job_id: str
    spec_hash: str
    best_candidate_id: str
    best_score: float
    n_trials: int
    n_folds: int
    algorithm: str
    gate_passed: bool
    tags: dict[str, str] = field(default_factory=dict)


def render_tuning_report(artifact: dict[str, Any]) -> TuningReport:
    """Construct a TuningReport from a raw artifact payload dict."""
    return TuningReport(
        job_id=artifact["job_id"],
        spec_hash=artifact["spec_hash"],
        best_candidate_id=artifact["best_candidate_id"],
        best_score=float(artifact["best_score"]),
        n_trials=int(artifact["n_trials"]),
        n_folds=int(artifact.get("n_folds", 0)),
        algorithm=artifact.get("algorithm", "unknown"),
        gate_passed=bool(artifact.get("gate_passed", False)),
        tags=dict(artifact.get("tags", {})),
    )


__all__ = ["TuningReport", "render_tuning_report"]

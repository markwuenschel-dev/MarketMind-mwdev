"""ValidationReport: per-fold cross-validation results rendered from artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FoldResult:
    """Result metrics for a single validation fold."""

    fold_index: int
    sharpe: float
    max_drawdown: float
    cost_stress_bps: float


@dataclass(frozen=True)
class ValidationReport:
    """Aggregated validation results across all folds."""

    job_id: str
    candidate_id: str
    method: str
    fold_results: tuple[FoldResult, ...]
    mean_sharpe: float
    mean_drawdown: float


def render_validation_report(artifact: dict[str, Any]) -> ValidationReport:
    """Construct a ValidationReport from a raw artifact payload dict."""
    folds = tuple(
        FoldResult(
            fold_index=int(f["fold_index"]),
            sharpe=float(f["sharpe"]),
            max_drawdown=float(f["max_drawdown"]),
            cost_stress_bps=float(f.get("cost_stress_bps", 0.0)),
        )
        for f in artifact.get("fold_results", [])
    )
    sharpes = [f.sharpe for f in folds]
    drawdowns = [f.max_drawdown for f in folds]
    return ValidationReport(
        job_id=artifact["job_id"],
        candidate_id=artifact["candidate_id"],
        method=artifact.get("method", "unknown"),
        fold_results=folds,
        mean_sharpe=sum(sharpes) / len(sharpes) if sharpes else 0.0,
        mean_drawdown=sum(drawdowns) / len(drawdowns) if drawdowns else 0.0,
    )


__all__ = ["FoldResult", "ValidationReport", "render_validation_report"]

"""GateReport: per-gate pass/fail breakdown rendered from artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GateReport:
    """Per-gate pass/fail summary for a candidate."""

    job_id: str
    candidate_id: str
    overall_passed: bool
    gate_scores: dict[str, float]
    gate_passed: dict[str, bool]
    dsr: float
    t_stat: float


def render_gate_report(artifact: dict[str, Any]) -> GateReport:
    """Construct a GateReport from a raw artifact payload dict."""
    return GateReport(
        job_id=artifact["job_id"],
        candidate_id=artifact["candidate_id"],
        overall_passed=bool(artifact["overall_passed"]),
        gate_scores=dict(artifact.get("gate_scores", {})),
        gate_passed=dict(artifact.get("gate_passed", {})),
        dsr=float(artifact.get("dsr", 0.0)),
        t_stat=float(artifact.get("t_stat", 0.0)),
    )


__all__ = ["GateReport", "render_gate_report"]

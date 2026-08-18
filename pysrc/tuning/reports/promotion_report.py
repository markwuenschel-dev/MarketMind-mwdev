"""PromotionReport: record of a promotion event rendered from artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class PromotionReport:
    """Summary of a completed promotion event."""

    job_id: str
    candidate_id: str
    promotion_id: str
    mode: str
    approved_by: str | None
    promoted_at: datetime
    artifact_hash: str


def render_promotion_report(artifact: dict[str, Any]) -> PromotionReport:
    """Construct a PromotionReport from a raw artifact payload dict."""
    return PromotionReport(
        job_id=artifact["job_id"],
        candidate_id=artifact["candidate_id"],
        promotion_id=artifact["promotion_id"],
        mode=artifact.get("mode", "unknown"),
        approved_by=artifact.get("approved_by"),
        promoted_at=datetime.fromisoformat(artifact["promoted_at"]),
        artifact_hash=artifact["artifact_hash"],
    )


__all__ = ["PromotionReport", "render_promotion_report"]

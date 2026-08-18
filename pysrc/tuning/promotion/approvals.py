"""Approvals: record and enforce human-in-the-loop gates before promotion."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ApprovalRecord:
    """Immutable record of a promotion approval decision."""

    job_id: str
    candidate_id: str
    approved: bool
    approver_id: str
    approved_at: datetime
    notes: str = ""


def require_approval(
    job_id: str,
    candidate_id: str,
    context: dict[str, object],
) -> ApprovalRecord:
    """Block until a human approval is recorded; raises if approval is denied."""
    raise NotImplementedError(
        "require_approval must be wired to an approval store or webhook handler"
    )


__all__ = ["ApprovalRecord", "require_approval"]

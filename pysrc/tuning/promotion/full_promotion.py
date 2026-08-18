"""Full promotion: replace the live strategy with the promoted candidate."""

from __future__ import annotations

from typing import Any


def run_full_promotion(
    candidate_id: str,
    job_id: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    """Execute full promotion; return a promotion receipt dict."""
    raise NotImplementedError(
        "run_full_promotion must be wired to live_checkpoint_switch and artifact writer"
    )


__all__ = ["run_full_promotion"]

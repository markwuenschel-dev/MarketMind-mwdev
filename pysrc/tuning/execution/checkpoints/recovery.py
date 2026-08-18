"""Recovery: resume a partially-completed job from its last checkpoint."""

from __future__ import annotations

from typing import Any


def recover_job(job_id: str, context: dict[str, Any]) -> dict[str, Any]:
    """Attempt to resume a job from its last persisted checkpoint.

    Returns a recovery context dict; raises RuntimeError if recovery is impossible.
    """
    raise NotImplementedError(
        f"recover_job for '{job_id}' must be wired to CandidateStore and ModelStore"
    )


__all__ = ["recover_job"]

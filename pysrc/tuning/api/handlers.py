"""Translates incoming API calls into orchestration commands.

Layer rule: Shell only — no computation; orchestration is imported lazily.
"""

from __future__ import annotations

from .schemas import (
    PromotionRequest,
    PromotionResponse,
    SearchStatusResponse,
    TuningJobRequest,
    TuningJobResponse,
)

__all__ = [
    "submit_tuning_job",
    "get_job_status",
    "submit_promotion",
]


def submit_tuning_job(req: TuningJobRequest) -> TuningJobResponse:
    """Accept a TuningJobRequest and dispatch to the orchestration layer."""
    raise NotImplementedError(
        "submit_tuning_job is not yet implemented; wire this handler to the tuning orchestrator."
    )


def get_job_status(job_id: str) -> SearchStatusResponse:
    """Return the current status and trial progress for the given job_id."""
    raise NotImplementedError(
        f"get_job_status is not yet implemented for job_id={job_id!r}; "
        "wire this handler to the run registry."
    )


def submit_promotion(req: PromotionRequest) -> PromotionResponse:
    """Accept a PromotionRequest and dispatch to the promotion service."""
    raise NotImplementedError(
        "submit_promotion is not yet implemented; wire this handler to the promotion orchestrator."
    )

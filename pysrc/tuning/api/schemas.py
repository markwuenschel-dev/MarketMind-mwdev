"""Pydantic request/response boundary models for the tuning API.

Layer rule: API boundary only — no domain logic.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

__all__ = [
    "TuningJobRequest",
    "TuningJobResponse",
    "PromotionRequest",
    "PromotionResponse",
    "SearchStatusResponse",
    "GateResultResponse",
]


class TuningJobRequest(BaseModel):
    """Incoming request to submit a new tuning job."""

    model_config = ConfigDict(frozen=True)

    job_id: str
    search_space_ref: str
    objective_ref: str
    validation_ref: str
    promotion_ref: str | None = None
    max_trials: int = 100
    timeout_seconds: int | None = None
    tags: dict[str, str] = {}


class TuningJobResponse(BaseModel):
    """Response returned after a tuning job is submitted."""

    model_config = ConfigDict(frozen=True)

    job_id: str
    status: str  # "submitted" | "running" | "complete" | "failed"
    run_id: str | None = None


class PromotionRequest(BaseModel):
    """Request to promote a candidate from a completed tuning job."""

    model_config = ConfigDict(frozen=True)

    job_id: str
    candidate_id: str
    mode: str  # "shadow" | "capped_blend" | "full"


class PromotionResponse(BaseModel):
    """Response returned after a promotion request is accepted."""

    model_config = ConfigDict(frozen=True)

    job_id: str
    candidate_id: str
    promotion_id: str
    status: str


class SearchStatusResponse(BaseModel):
    """Snapshot of the current search progress for a given job."""

    model_config = ConfigDict(frozen=True)

    job_id: str
    status: str
    trials_complete: int
    best_score: float | None = None


class GateResultResponse(BaseModel):
    """Statistical gate evaluation result for a promotion candidate."""

    model_config = ConfigDict(frozen=True)

    job_id: str
    candidate_id: str
    passed: bool
    gate_scores: dict[str, float]

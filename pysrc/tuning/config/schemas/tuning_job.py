"""TuningJobConfig: top-level declarative spec for a tuning job."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["TuningJobConfig"]


class TuningJobConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    job_id: str
    version: str = "1.0.0"
    search_space_ref: str
    objective_ref: str
    validation_ref: str
    promotion_ref: str | None = None
    max_trials: int = Field(default=100, ge=1)
    timeout_seconds: int | None = None
    determinism_tier: Literal["d0", "d1", "d2", "d3"] = "d1"
    tags: dict[str, str] = Field(default_factory=dict)
    crisis_holdout: bool = True

"""TuningJobSpec: validated, frozen specification for a tuning job."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class TuningJobSpec:
    """Validated, immutable specification derived from TuningJobConfig."""

    job_id: str
    version: str
    search_space_ref: str
    objective_ref: str
    validation_ref: str
    spec_hash: str  # cas.v1:b3-256:<hex>
    max_trials: int
    determinism_tier: Literal["d0", "d1", "d2", "d3"]
    promotion_ref: str | None = None
    timeout_seconds: int | None = None
    crisis_holdout: bool = True
    tags: dict[str, str] = field(default_factory=dict)


__all__ = ["TuningJobSpec"]

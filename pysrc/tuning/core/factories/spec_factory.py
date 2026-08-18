"""SpecFactory: build Spec objects from raw config dicts."""

from __future__ import annotations

from typing import Any

from pysrc.tuning.core.specs.tuning_job_spec import TuningJobSpec

__all__ = ["build_tuning_job_spec"]


def build_tuning_job_spec(raw: dict[str, Any], spec_hash: str) -> TuningJobSpec:
    """Build a validated TuningJobSpec from a raw config dict + pre-computed hash."""
    return TuningJobSpec(
        job_id=raw["job_id"],
        version=raw.get("version", "1.0.0"),
        search_space_ref=raw["search_space_ref"],
        objective_ref=raw["objective_ref"],
        validation_ref=raw["validation_ref"],
        spec_hash=spec_hash,
        max_trials=int(raw.get("max_trials", 100)),
        determinism_tier=raw.get("determinism_tier", "d1"),
        promotion_ref=raw.get("promotion_ref"),
        timeout_seconds=raw.get("timeout_seconds"),
        crisis_holdout=bool(raw.get("crisis_holdout", True)),
        tags=dict(raw.get("tags", {})),
    )

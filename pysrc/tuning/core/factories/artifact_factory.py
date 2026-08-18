"""ArtifactFactory: build artifact payload dicts from tuning results."""

from __future__ import annotations

from typing import Any


def build_trial_artifact(
    job_id: str,
    candidate_id: str,
    params: dict[str, Any],
    scores: dict[str, float],
    spec_hash: str,
    determinism_tier: str,
) -> dict[str, Any]:
    """Construct a serialisable artifact payload for a single trial result."""
    return {
        "schema": "tuning.trial.v1",
        "job_id": job_id,
        "candidate_id": candidate_id,
        "spec_hash": spec_hash,
        "determinism_tier": determinism_tier,
        "params": params,
        "scores": scores,
    }


__all__ = ["build_trial_artifact"]

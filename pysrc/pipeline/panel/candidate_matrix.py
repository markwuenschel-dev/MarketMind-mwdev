"""Panel-owned model candidate defaults.

This is a model-training input, not a router comparator matrix.
"""

from __future__ import annotations

from pysrc.contracts.candidate_spec import CandidateSpec
from pysrc.pipeline.contracts.p2 import (
    CandidateMatrix,
    P2Config,
    RunMetadata,
    RunPhase,
)


def load_panel_candidate_matrix(config: P2Config) -> CandidateMatrix:
    """Build the declared panel model candidates without an artifact fallback."""
    candidates = [
        CandidateSpec(
            candidate_id=f"panel__{family}__forward_return",
            model_family=family,
            router_target="child_utility_regression",
            decision_rule="free_routing",
            input_surface="pipeline_indicator_panel",
            feature_allowlist="full_indicator_universe_v1",
            split_policy="w4a_fold_split",
            status="active",
            feature_policy="full_indicator_universe_v1",
        )
        for family in config.panel_model_families
    ]
    return CandidateMatrix(
        meta=RunMetadata(phase=RunPhase.MATRIX, random_seed=config.random_seed),
        candidates=candidates,
        summary={"input_surface": "pipeline_indicator_panel", "candidate_count": len(candidates)},
    )


__all__ = ["load_panel_candidate_matrix"]

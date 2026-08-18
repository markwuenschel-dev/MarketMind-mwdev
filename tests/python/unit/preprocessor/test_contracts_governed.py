from __future__ import annotations

from dataclasses import replace

import pytest

from pysrc.core.errors import PreprocessingError
from pysrc.pipeline.contracts.governance import GovernanceDecision
from pysrc.preprocessor.contracts.executor import (
    CapabilityFacts,
    ExecutionEvidence,
    GovernedExecutionCacheKey,
    GovernedExecutionSpec,
    reject_if_governance_required,
    validate_governed_execution,
)
from pysrc.preprocessor.contracts.plan import CanonicalOp, MaterializationSpec, PreprocessingPlan
from pysrc.preprocessor.contracts.state import FitStateArtifact, PreprocessingStateManifest


@pytest.mark.determinism("d0")
def test_plan_identity_is_deterministic_and_not_object_identity(deterministic_seed: int) -> None:
    op = CanonicalOp(
        name="feature.sma", params={"column": "close", "window": 5}, provides=("sma_5",)
    )
    left = PreprocessingPlan(version="1.0", ops=(op,), group_by=("symbol",))
    right = PreprocessingPlan(
        version="1.0",
        ops=(
            CanonicalOp(
                name="feature.sma", params={"window": 5, "column": "close"}, provides=("sma_5",)
            ),
        ),
        group_by=("symbol",),
    )

    assert left.plan_id == right.plan_id
    assert str(left.plan_id).startswith("cas.v1:b3-256:")


@pytest.mark.determinism("d0")
def test_state_manifest_identity_is_deterministic_and_version_checked(
    deterministic_seed: int,
) -> None:
    state = PreprocessingStateManifest(
        schema_version="1.0",
        plan_version="1.0",
        artifacts=(FitStateArtifact(name="scaler", payload={"center": [1, 2], "scale": [3, 4]}),),
        lineage={"source": "unit-test"},
    )

    assert str(state.state_id).startswith("cas.v1:b3-256:")

    with pytest.raises(ValueError, match="plan_version"):
        replace(state, plan_version="2.0")


@pytest.mark.determinism("d0")
def test_governed_execution_cache_key_is_content_addressed(deterministic_seed: int) -> None:
    plan = PreprocessingPlan(
        version="1.0",
        ops=(
            CanonicalOp(name="feature.returns", params={"column": "close"}, provides=("returns",)),
        ),
        group_by=("symbol",),
    )
    state = PreprocessingStateManifest(schema_version="1.0", plan_version="1.0")
    caps = CapabilityFacts(has_cudf=False, has_polars_gpu=False, backend="polars")
    decision = GovernanceDecision.admit("content-addressed")

    key_a = GovernedExecutionCacheKey.from_inputs(
        plan=plan, state=state, capabilities=caps, governance=decision
    )
    key_b = GovernedExecutionCacheKey.from_inputs(
        plan=replace(plan),
        state=replace(state),
        capabilities=CapabilityFacts(has_cudf=False, has_polars_gpu=False, backend="polars"),
        governance=GovernanceDecision.admit("content-addressed"),
    )

    assert key_a.value == key_b.value
    assert key_a.value.startswith("cas.v1:b3-256:")


@pytest.mark.determinism("d0")
def test_governed_execution_rejects_implicit_materialization_and_retry(
    deterministic_seed: int,
) -> None:
    plan = PreprocessingPlan(
        version="1.0",
        ops=(
            CanonicalOp(
                name="feature.rsi", params={"column": "close", "window": 14}, provides=("rsi_14",)
            ),
        ),
        group_by=("symbol",),
        materialization=MaterializationSpec(format="polars", schema_signature="ohlcv+rsi"),
    )
    state = PreprocessingStateManifest(schema_version="1.0", plan_version="1.0")
    caps = CapabilityFacts(has_cudf=True, has_polars_gpu=False, backend="cudf")
    decision = GovernanceDecision.admit("strict execution")

    with pytest.raises(PreprocessingError, match="materialization"):
        validate_governed_execution(
            GovernedExecutionSpec(
                plan=plan,
                state=state,
                capabilities=caps,
                governance=decision,
                actual_materialization="pandas",
                attempted_retry=True,
                schema_signature="ohlcv+rsi",
                evidence=ExecutionEvidence(events=("fallback",)),
            )
        )


@pytest.mark.determinism("d0")
def test_governed_execution_rejects_compatibility_fallback_request(deterministic_seed: int) -> None:
    with pytest.raises(PreprocessingError, match="governed path"):
        reject_if_governance_required(governed=True, fallback_name="direct_csv")

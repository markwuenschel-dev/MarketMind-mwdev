from __future__ import annotations

from pysrc.pipeline.contracts.governance import GovernanceDecision
from pysrc.preprocessor.contracts.executor import (
    CapabilityFacts,
    ExecutionEvidence,
    GovernedExecutionSpec,
)
from pysrc.preprocessor.contracts.plan import PreprocessingPlan
from pysrc.preprocessor.contracts.state import PreprocessingStateManifest


def build_governed_execution_spec(
    *,
    plan: PreprocessingPlan,
    state: PreprocessingStateManifest,
    backend: str,
    actual_materialization: str,
    schema_signature: str,
    has_cudf: bool = False,
    has_polars_gpu: bool = False,
    governance: GovernanceDecision | None = None,
    evidence: ExecutionEvidence | None = None,
) -> GovernedExecutionSpec:
    return GovernedExecutionSpec(
        plan=plan,
        state=state,
        capabilities=CapabilityFacts(
            has_cudf=has_cudf,
            has_polars_gpu=has_polars_gpu,
            backend=backend,
        ),
        governance=governance or GovernanceDecision.admit("tuning adapter execution"),
        actual_materialization=actual_materialization,
        schema_signature=schema_signature,
        evidence=evidence or ExecutionEvidence(),
    )

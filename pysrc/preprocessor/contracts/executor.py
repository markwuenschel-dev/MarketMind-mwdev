from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from pysrc.core.errors import PreprocessingError
from pysrc.ops.hashing_contract import HashingContract, HashRef
from pysrc.pipeline.contracts.governance import GovernanceDecision, require_governed_admissible
from pysrc.preprocessor.contracts.plan import PreprocessingPlan
from pysrc.preprocessor.contracts.state import PreprocessingStateManifest


@dataclass(frozen=True, slots=True)
class CapabilityFacts:
    has_cudf: bool
    has_polars_gpu: bool
    backend: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "has_cudf": self.has_cudf,
            "has_polars_gpu": self.has_polars_gpu,
            "backend": self.backend,
        }


@dataclass(frozen=True, slots=True)
class ExecutionEvidence:
    events: tuple[str, ...] = field(default_factory=tuple)
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(self, "metrics", dict(self.metrics))

    def to_payload(self) -> dict[str, Any]:
        return {"events": list(self.events), "metrics": dict(self.metrics)}


@dataclass(frozen=True, slots=True)
class GovernedExecutionSpec:
    plan: PreprocessingPlan
    state: PreprocessingStateManifest
    capabilities: CapabilityFacts
    governance: GovernanceDecision
    actual_materialization: str
    schema_signature: str
    evidence: ExecutionEvidence = field(default_factory=ExecutionEvidence)
    attempted_retry: bool = False


@dataclass(frozen=True, slots=True)
class GovernedExecutionCacheKey:
    value: str

    @classmethod
    def from_inputs(
        cls,
        *,
        plan: PreprocessingPlan,
        state: PreprocessingStateManifest,
        capabilities: CapabilityFacts,
        governance: GovernanceDecision,
    ) -> GovernedExecutionCacheKey:
        payload = {
            "plan_id": str(plan.plan_id),
            "state_id": str(state.state_id),
            "capabilities": capabilities.to_payload(),
            "governance": {
                "admissibility": governance.admissibility.value,
                "reasons": list(governance.reasons),
                "failure_class": governance.failure_class.value
                if governance.failure_class
                else None,
                "downgrade_policy": governance.downgrade_policy.value,
                "materialization_policy": governance.materialization_policy.value,
                "retry_policy": governance.retry_policy.value,
                "governed": governance.governed,
            },
        }
        HashingContract.check_banned_values(payload)
        digest = HashingContract.hash_for_identity(
            HashingContract.canonicalize_json(payload).encode("utf-8")
        ).hex()
        ref = HashRef(domain="cas.v1", algo="b3-256", hex_digest=digest)
        return cls(value=str(ref))


def reject_if_governance_required(*, governed: bool, fallback_name: str) -> None:
    if governed:
        raise PreprocessingError(f"governed path rejected compatibility fallback: {fallback_name}")


def validate_governed_execution(spec: GovernedExecutionSpec) -> GovernedExecutionSpec:
    try:
        require_governed_admissible(spec.governance)
    except ValueError as exc:
        raise PreprocessingError(str(exc)) from exc
    declared = spec.plan.materialization
    if spec.actual_materialization != declared.format:
        raise PreprocessingError(
            "governed path rejected implicit materialization change "
            f"from {declared.format} to {spec.actual_materialization}"
        )
    if declared.schema_signature and spec.schema_signature != declared.schema_signature:
        raise PreprocessingError(
            "governed path rejected schema drift "
            f"from {declared.schema_signature} to {spec.schema_signature}"
        )
    if spec.attempted_retry:
        raise PreprocessingError("governed path rejected retry under changed semantics")
    if any(event in {"fallback", "compatibility"} for event in spec.evidence.events):
        raise PreprocessingError("governed path rejected compatibility evidence")
    return spec

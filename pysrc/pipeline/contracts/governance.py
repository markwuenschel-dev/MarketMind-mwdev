from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Admissibility(StrEnum):
    ADMIT = "admit"
    REJECT = "reject"


class FailureClass(StrEnum):
    CONTRACT_VIOLATION = "contract_violation"
    CAPABILITY_REJECTION = "capability_rejection"
    RETRYABLE_OPERATIONAL_FAILURE = "retryable_operational_failure"
    NON_GOVERNED_COMPATIBILITY_PATH = "non_governed_compatibility_path"


class DowngradePolicy(StrEnum):
    NONE_ALLOWED = "none_allowed"
    EXPLICIT_APPROVAL_REQUIRED = "explicit_approval_required"


class MaterializationPolicy(StrEnum):
    EXACT_DECLARED_ONLY = "exact_declared_only"
    EXPLICIT_CHANGE_REQUIRED = "explicit_change_required"


class RetryPolicyClass(StrEnum):
    NONE = "none"
    SAME_SEMANTICS_ONLY = "same_semantics_only"


@dataclass(frozen=True, slots=True)
class GovernanceDecision:
    admissibility: Admissibility
    reasons: tuple[str, ...]
    failure_class: FailureClass | None = None
    downgrade_policy: DowngradePolicy = DowngradePolicy.NONE_ALLOWED
    materialization_policy: MaterializationPolicy = MaterializationPolicy.EXACT_DECLARED_ONLY
    retry_policy: RetryPolicyClass = RetryPolicyClass.NONE
    governed: bool = True

    def __post_init__(self) -> None:
        if not self.reasons:
            raise ValueError("GovernanceDecision requires at least one reason")

    @classmethod
    def admit(cls, reason: str, *, governed: bool = True) -> GovernanceDecision:
        return cls(
            admissibility=Admissibility.ADMIT,
            reasons=(reason,),
            governed=governed,
        )

    @classmethod
    def reject(
        cls,
        reason: str,
        *,
        failure_class: FailureClass = FailureClass.CONTRACT_VIOLATION,
        governed: bool = True,
    ) -> GovernanceDecision:
        return cls(
            admissibility=Admissibility.REJECT,
            reasons=(reason,),
            failure_class=failure_class,
            governed=governed,
        )


def validate_fail_closed_defaults(decision: GovernanceDecision) -> GovernanceDecision:
    if decision.downgrade_policy is not DowngradePolicy.NONE_ALLOWED:
        raise ValueError("Governed defaults must reject downgrade policy changes")
    if decision.materialization_policy is not MaterializationPolicy.EXACT_DECLARED_ONLY:
        raise ValueError("Governed defaults must reject implicit materialization changes")
    if decision.retry_policy is not RetryPolicyClass.NONE:
        raise ValueError("Governed defaults must reject implicit retries")
    if decision.failure_class is FailureClass.NON_GOVERNED_COMPATIBILITY_PATH:
        raise ValueError("Governed defaults forbid compatibility fallback paths")
    return decision


def require_governed_admissible(decision: GovernanceDecision) -> GovernanceDecision:
    validate_fail_closed_defaults(decision)
    if decision.admissibility is not Admissibility.ADMIT:
        raise ValueError("Governed execution requires an admissible governance decision")
    return decision

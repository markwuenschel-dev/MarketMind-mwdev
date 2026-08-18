from __future__ import annotations

import pytest

from pysrc.pipeline.contracts.governance import (
    Admissibility,
    DowngradePolicy,
    FailureClass,
    GovernanceDecision,
    MaterializationPolicy,
    RetryPolicyClass,
    require_governed_admissible,
    validate_fail_closed_defaults,
)


@pytest.mark.determinism("d0")
def test_governance_reject_defaults_are_fail_closed(deterministic_seed: int) -> None:
    decision = GovernanceDecision.reject("silent downgrade blocked")

    assert decision.admissibility is Admissibility.REJECT
    assert decision.failure_class is FailureClass.CONTRACT_VIOLATION
    assert decision.downgrade_policy is DowngradePolicy.NONE_ALLOWED
    assert decision.materialization_policy is MaterializationPolicy.EXACT_DECLARED_ONLY
    assert decision.retry_policy is RetryPolicyClass.NONE
    validate_fail_closed_defaults(decision)


@pytest.mark.determinism("d0")
def test_governance_admit_defaults_remain_strict(deterministic_seed: int) -> None:
    decision = GovernanceDecision.admit("capabilities satisfied")

    require_governed_admissible(decision)
    assert decision.downgrade_policy is DowngradePolicy.NONE_ALLOWED
    assert decision.materialization_policy is MaterializationPolicy.EXACT_DECLARED_ONLY
    assert decision.retry_policy is RetryPolicyClass.NONE


@pytest.mark.determinism("d0")
def test_governance_rejects_non_governed_compatibility_paths(deterministic_seed: int) -> None:
    decision = GovernanceDecision(
        admissibility=Admissibility.ADMIT,
        reasons=("compat fallback requested",),
        failure_class=FailureClass.NON_GOVERNED_COMPATIBILITY_PATH,
        downgrade_policy=DowngradePolicy.NONE_ALLOWED,
        materialization_policy=MaterializationPolicy.EXACT_DECLARED_ONLY,
        retry_policy=RetryPolicyClass.NONE,
        governed=True,
    )

    with pytest.raises(ValueError, match="compatibility"):
        validate_fail_closed_defaults(decision)

"""Screening taxonomy: stages, statuses, reason codes, and REASON_CODE_TO_FAMILY.

Contract: reason_family is always derived from REASON_CODE_TO_FAMILY; callers never
pass reason_family. Builders and producers look up via REASON_CODE_TO_FAMILY[reason_code].
"""

from __future__ import annotations

from enum import StrEnum


class ScreeningStage(StrEnum):
    """Stage in the screening funnel."""

    INTAKE = "INTAKE"
    LANE_0 = "LANE_0"
    LANE_1 = "LANE_1"
    LANE_2 = "LANE_2"
    PROMOTION = "PROMOTION"


class ScreeningStatus(StrEnum):
    """Outcome of a stage."""

    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    ERROR = "ERROR"
    SKIPPED = "SKIPPED"


class ReasonFamily(StrEnum):
    """Family for grouping reason codes."""

    SPEC = "SPEC"
    DATA = "DATA"
    DUPLICATE = "DUPLICATE"
    INVARIANT = "INVARIANT"
    STAT_VALIDITY = "STAT_VALIDITY"
    COST = "COST"
    STABILITY = "STABILITY"
    PROMOTION = "PROMOTION"
    SYSTEM = "SYSTEM"


class ReasonCode(StrEnum):
    """Tier 1: Intake / pre-gate rejection codes."""

    SPEC_INVALID = "SPEC_INVALID"
    SCHEMA_VALIDATION_FAIL = "SCHEMA_VALIDATION_FAIL"
    DUPLICATE_SPEC_HASH = "DUPLICATE_SPEC_HASH"
    DATA_UNAVAILABLE = "DATA_UNAVAILABLE"
    UNSUPPORTED_INPUT_DOMAIN = "UNSUPPORTED_INPUT_DOMAIN"
    PROVENANCE_REFERENCE_MISSING = "PROVENANCE_REFERENCE_MISSING"
    RESOURCE_BUDGET_EXCEEDED = "RESOURCE_BUDGET_EXCEEDED"
    INVARIANT_PRECHECK_FAIL = "INVARIANT_PRECHECK_FAIL"
    # Tier 2: Evaluation / gate failure codes
    IC_BELOW_THRESHOLD = "IC_BELOW_THRESHOLD"
    DSR_P_ABOVE_CUTOFF = "DSR_P_ABOVE_CUTOFF"
    PBO_ABOVE_CUTOFF = "PBO_ABOVE_CUTOFF"
    HARVEY_T_BELOW_CUTOFF = "HARVEY_T_BELOW_CUTOFF"
    COST_MODEL_FAIL = "COST_MODEL_FAIL"
    LEAKAGE_INVARIANT_VIOLATION = "LEAKAGE_INVARIANT_VIOLATION"
    FEATURE_STABILITY_FAIL = "FEATURE_STABILITY_FAIL"
    REGIME_COVERAGE_FAIL = "REGIME_COVERAGE_FAIL"
    ANTI_GOODHART_FAIL = "ANTI_GOODHART_FAIL"
    BASELINE_REGRESSION = "BASELINE_REGRESSION"
    PROMOTION_VETO = "PROMOTION_VETO"


# Contract: reason_family is always derived from this mapping; callers never pass reason_family.
REASON_CODE_TO_FAMILY: dict[ReasonCode, ReasonFamily] = {
    # Tier 1
    ReasonCode.SPEC_INVALID: ReasonFamily.SPEC,
    ReasonCode.SCHEMA_VALIDATION_FAIL: ReasonFamily.SPEC,
    ReasonCode.DUPLICATE_SPEC_HASH: ReasonFamily.DUPLICATE,
    ReasonCode.DATA_UNAVAILABLE: ReasonFamily.DATA,
    ReasonCode.UNSUPPORTED_INPUT_DOMAIN: ReasonFamily.DATA,
    ReasonCode.PROVENANCE_REFERENCE_MISSING: ReasonFamily.DATA,
    ReasonCode.RESOURCE_BUDGET_EXCEEDED: ReasonFamily.SYSTEM,
    ReasonCode.INVARIANT_PRECHECK_FAIL: ReasonFamily.INVARIANT,
    # Tier 2
    ReasonCode.IC_BELOW_THRESHOLD: ReasonFamily.STAT_VALIDITY,
    ReasonCode.DSR_P_ABOVE_CUTOFF: ReasonFamily.STAT_VALIDITY,
    ReasonCode.PBO_ABOVE_CUTOFF: ReasonFamily.STAT_VALIDITY,
    ReasonCode.HARVEY_T_BELOW_CUTOFF: ReasonFamily.STAT_VALIDITY,
    ReasonCode.COST_MODEL_FAIL: ReasonFamily.COST,
    ReasonCode.LEAKAGE_INVARIANT_VIOLATION: ReasonFamily.INVARIANT,
    ReasonCode.FEATURE_STABILITY_FAIL: ReasonFamily.STABILITY,
    ReasonCode.REGIME_COVERAGE_FAIL: ReasonFamily.STAT_VALIDITY,
    ReasonCode.ANTI_GOODHART_FAIL: ReasonFamily.STABILITY,
    ReasonCode.BASELINE_REGRESSION: ReasonFamily.STAT_VALIDITY,
    ReasonCode.PROMOTION_VETO: ReasonFamily.PROMOTION,
}

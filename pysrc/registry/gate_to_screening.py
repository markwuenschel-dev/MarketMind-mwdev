"""Map marketmind_gate ValidationResult (gate_id + status) to ScreeningStage + ReasonCode.

Phase I: explicit mapping for files_exist, json_valid, sharpe_threshold, max_drawdown.
Pass and fail paths use the same stage per gate so funnel analytics are correct
(INTAKE twice, LANE_0 twice, not LANE_0 four times).
"""

from __future__ import annotations

from pysrc.registry.screening_taxonomy import ReasonCode, ScreeningStage

# Stage per gate_id — used for both pass and fail so ACCEPTED/REJECTED are recorded at the right stage.
GATE_STAGE_MAP: dict[str, ScreeningStage] = {
    "files_exist": ScreeningStage.INTAKE,
    "json_valid": ScreeningStage.INTAKE,
    "sharpe_threshold": ScreeningStage.LANE_0,
    "max_drawdown": ScreeningStage.LANE_0,
}

# Reason code when gate fails (max_drawdown -> FEATURE_STABILITY_FAIL as risk/stability, not IC).
GATE_FAIL_REASON_MAP: dict[str, ReasonCode] = {
    "files_exist": ReasonCode.DATA_UNAVAILABLE,
    "json_valid": ReasonCode.SPEC_INVALID,
    "sharpe_threshold": ReasonCode.IC_BELOW_THRESHOLD,
    "max_drawdown": ReasonCode.FEATURE_STABILITY_FAIL,
}


def gate_result_to_stage_and_code(
    gate_id: str,
    passed: bool,
    reason: str | None = None,
) -> tuple[ScreeningStage, ReasonCode | None]:
    """Map a single gate result to (ScreeningStage, ReasonCode | None).

    When passed is True, returns (stage, None) so caller records ACCEPTED at the correct stage.
    When passed is False, returns (stage, code) for the rejection.
    Both paths use GATE_STAGE_MAP so intake vs lane is correct for funnel analytics.
    """
    _ = reason  # Optional: could refine code from reason text in Phase II
    stage = GATE_STAGE_MAP.get(gate_id, ScreeningStage.LANE_0)
    if passed:
        return (stage, None)
    code = GATE_FAIL_REASON_MAP.get(gate_id, ReasonCode.IC_BELOW_THRESHOLD)
    return (stage, code)

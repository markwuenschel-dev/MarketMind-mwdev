"""Tests for gate_result_to_stage_and_code mapping."""

from __future__ import annotations

import pytest

from pysrc.registry.gate_to_screening import gate_result_to_stage_and_code
from pysrc.registry.screening_taxonomy import ReasonCode, ScreeningStage


@pytest.mark.determinism("d0")
def test_files_exist_fail_maps_to_intake_data_unavailable() -> None:
    stage, code = gate_result_to_stage_and_code("files_exist", False)
    assert stage == ScreeningStage.INTAKE
    assert code == ReasonCode.DATA_UNAVAILABLE


@pytest.mark.determinism("d0")
def test_json_valid_fail_maps_to_intake_spec_invalid() -> None:
    stage, code = gate_result_to_stage_and_code("json_valid", False)
    assert stage == ScreeningStage.INTAKE
    assert code == ReasonCode.SPEC_INVALID


@pytest.mark.determinism("d0")
def test_sharpe_threshold_fail_maps_to_lane0_ic_below() -> None:
    stage, code = gate_result_to_stage_and_code("sharpe_threshold", False)
    assert stage == ScreeningStage.LANE_0
    assert code == ReasonCode.IC_BELOW_THRESHOLD


@pytest.mark.determinism("d0")
def test_max_drawdown_fail_maps_to_lane0_feature_stability_fail() -> None:
    stage, code = gate_result_to_stage_and_code("max_drawdown", False)
    assert stage == ScreeningStage.LANE_0
    assert code == ReasonCode.FEATURE_STABILITY_FAIL


@pytest.mark.determinism("d0")
def test_passed_returns_none_reason_code_and_correct_stage() -> None:
    stage, code = gate_result_to_stage_and_code("sharpe_threshold", True)
    assert stage == ScreeningStage.LANE_0
    assert code is None


@pytest.mark.determinism("d0")
def test_passed_intake_gates_return_intake_stage() -> None:
    """Passing files_exist/json_valid recorded as INTAKE ACCEPTED, not LANE_0."""
    for gate_id in ("files_exist", "json_valid"):
        stage, code = gate_result_to_stage_and_code(gate_id, True)
        assert stage == ScreeningStage.INTAKE
        assert code is None

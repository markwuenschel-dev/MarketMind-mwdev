"""MLN-03 confidence_scalar contract."""

from __future__ import annotations

import pytest

from pysrc.core.errors import DataPreconditionError
from pysrc.meta_learning.confidence_contract import (
    apply_confidence_attenuation,
    insufficient_confidence_calibration_block,
    is_routing_enabled,
    synthetic_confidence_calibration_pass_block,
    validate_confidence_calibration_artifact_block,
    validate_confidence_scalar,
)


@pytest.mark.determinism("d0")
def test_validate_confidence_scalar_bounds() -> None:
    assert validate_confidence_scalar(0.0) == 0.0
    assert validate_confidence_scalar(1.0) == 1.0
    assert validate_confidence_scalar(0.5) == 0.5
    with pytest.raises(DataPreconditionError):
        validate_confidence_scalar(-0.01)
    with pytest.raises(DataPreconditionError):
        validate_confidence_scalar(1.01)
    with pytest.raises(DataPreconditionError):
        validate_confidence_scalar(float("nan"))


@pytest.mark.determinism("d0")
def test_apply_confidence_attenuation_zero_abstains() -> None:
    assert apply_confidence_attenuation(base_position=100.0, confidence_scalar=0.0) == 0.0


@pytest.mark.determinism("d0")
def test_apply_confidence_attenuation_one_preserves() -> None:
    assert apply_confidence_attenuation(base_position=-50.0, confidence_scalar=1.0) == -50.0


@pytest.mark.determinism("d0")
def test_apply_confidence_attenuation_reduces_long_and_short() -> None:
    assert apply_confidence_attenuation(base_position=100.0, confidence_scalar=0.25) == 25.0
    assert apply_confidence_attenuation(base_position=-100.0, confidence_scalar=0.25) == -25.0


@pytest.mark.determinism("d0")
def test_routing_disabled_by_default() -> None:
    assert is_routing_enabled() is False
    assert is_routing_enabled(pilot_explicit_opt_in=True) is False
    assert is_routing_enabled(reject_set_negative_evidence_after_costs=True) is False
    assert (
        is_routing_enabled(
            pilot_explicit_opt_in=True,
            reject_set_negative_evidence_after_costs=True,
        )
        is True
    )


@pytest.mark.determinism("d0")
def test_insufficient_calibration_block_validates() -> None:
    block = insufficient_confidence_calibration_block(reason="unit test")
    validate_confidence_calibration_artifact_block(block)
    assert block["routing_pilot"]["enabled"] is False


@pytest.mark.determinism("d0")
def test_synthetic_pass_block_validates() -> None:
    block = synthetic_confidence_calibration_pass_block(ece_value=0.07)
    validate_confidence_calibration_artifact_block(block)
    assert block["reporting_gate"] == "PASS"


@pytest.mark.determinism("d0")
def test_validate_confidence_calibration_rejects_bad_schema() -> None:
    bad = dict(insufficient_confidence_calibration_block(reason="x"))
    bad["schema_version"] = "wrong"
    with pytest.raises(DataPreconditionError):
        validate_confidence_calibration_artifact_block(bad)


@pytest.mark.determinism("d0")
def test_validate_confidence_calibration_rejects_routing_enabled_without_contract() -> None:
    bad = dict(insufficient_confidence_calibration_block(reason="x"))
    bad["routing_pilot"] = {"enabled": True}
    with pytest.raises(DataPreconditionError):
        validate_confidence_calibration_artifact_block(bad)

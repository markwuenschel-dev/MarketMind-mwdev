"""Tests for the governed execution-assumptions lane."""

from __future__ import annotations

import re

import pytest

from pysrc.meta.execution_assumptions_config import (
    GOVERNED_SCHEMA_VERSION,
    ExecutionAssumptionsConfig,
    ExecutionParityDeclaration,
)
from pysrc.meta.execution_assumptions_emitter import (
    GOVERNED_CONTENT_HASH_ALGORITHM,
    GOVERNED_CONTENT_HASH_CANONICALIZATION,
    build_governed_execution_assumptions_document,
    emit_governed_execution_assumptions,
    validate_governed_execution_assumptions_document,
)
from pysrc.meta.execution_assumptions_errors import (
    ExecutionAssumptionsFieldError,
    ExecutionAssumptionsHashError,
    ExecutionAssumptionsParityError,
)


def _config(**overrides: object) -> ExecutionAssumptionsConfig:
    parity = ExecutionParityDeclaration(
        cost_assumptions_match_baseline=True,
        split_assumptions_match_baseline=True,
        data_assumptions_match_baseline=True,
        parity_note="synthetic parity declaration",
    )
    payload: dict[str, object] = {
        "run_id": "run.sha256:1111111111111111111111111111111111111111111111111111111111111111",
        "cost_bps": 5.0,
        "slippage_model": "flat",
        "slippage_bps_estimate": 2.0,
        "borrow_rate_annual_bps": 0.0,
        "latency_model": "T+1 bar",
        "fill_assumption": "next_open",
        "parity": parity,
    }
    payload.update(overrides)
    return ExecutionAssumptionsConfig(**payload)  # type: ignore[arg-type]


@pytest.mark.determinism("d1")
def test_schema_version_present(deterministic_seed: int) -> None:
    _ = deterministic_seed
    report = emit_governed_execution_assumptions(_config())
    doc = report.document
    assert report.schema_version == GOVERNED_SCHEMA_VERSION
    assert doc["schema_version"] == GOVERNED_SCHEMA_VERSION


@pytest.mark.determinism("d1")
def test_all_required_fields_emitted(deterministic_seed: int) -> None:
    _ = deterministic_seed
    doc = build_governed_execution_assumptions_document(_config())
    assert set(doc) == {
        "schema_version",
        "run_id",
        "cost_bps",
        "slippage_model",
        "slippage_bps_estimate",
        "borrow_rate_annual_bps",
        "latency_model",
        "fill_assumption",
        "parity",
        "content_hash",
    }
    assert doc["content_hash"]["algorithm"] == GOVERNED_CONTENT_HASH_ALGORITHM
    assert doc["content_hash"]["canonicalization"] == GOVERNED_CONTENT_HASH_CANONICALIZATION
    assert doc["parity"] == _config().parity.to_json_obj()


@pytest.mark.determinism("d1")
def test_negative_cost_bps_raises(deterministic_seed: int) -> None:
    _ = deterministic_seed
    with pytest.raises(ExecutionAssumptionsFieldError):
        emit_governed_execution_assumptions(_config(cost_bps=-1.0))


@pytest.mark.determinism("d1")
def test_empty_slippage_model_raises(deterministic_seed: int) -> None:
    _ = deterministic_seed
    with pytest.raises(ExecutionAssumptionsFieldError):
        emit_governed_execution_assumptions(_config(slippage_model=""))


@pytest.mark.determinism("d1")
def test_cost_parity_false_raises(deterministic_seed: int) -> None:
    _ = deterministic_seed
    parity = ExecutionParityDeclaration(False, True, True, "synthetic parity declaration")
    with pytest.raises(ExecutionAssumptionsParityError):
        emit_governed_execution_assumptions(_config(parity=parity))


@pytest.mark.determinism("d1")
def test_split_parity_false_raises(deterministic_seed: int) -> None:
    _ = deterministic_seed
    parity = ExecutionParityDeclaration(True, False, True, "synthetic parity declaration")
    with pytest.raises(ExecutionAssumptionsParityError):
        emit_governed_execution_assumptions(_config(parity=parity))


@pytest.mark.determinism("d1")
def test_data_parity_false_raises(deterministic_seed: int) -> None:
    _ = deterministic_seed
    parity = ExecutionParityDeclaration(True, True, False, "synthetic parity declaration")
    with pytest.raises(ExecutionAssumptionsParityError):
        emit_governed_execution_assumptions(_config(parity=parity))


@pytest.mark.determinism("d1")
def test_borrow_rate_none_is_valid(deterministic_seed: int) -> None:
    _ = deterministic_seed
    doc = emit_governed_execution_assumptions(
        _config(borrow_rate_annual_bps=None)
    ).to_json_document()
    assert doc["borrow_rate_annual_bps"] is None


@pytest.mark.determinism("d1")
def test_content_hash_determinism(deterministic_seed: int) -> None:
    _ = deterministic_seed
    a = emit_governed_execution_assumptions(_config())
    b = emit_governed_execution_assumptions(_config())
    assert a.content_hash == b.content_hash
    assert a.document == b.document


@pytest.mark.determinism("d1")
def test_content_hash_format(deterministic_seed: int) -> None:
    _ = deterministic_seed
    report = emit_governed_execution_assumptions(_config())
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", report.content_hash)


@pytest.mark.determinism("d1")
def test_tampered_hash_raises(deterministic_seed: int) -> None:
    _ = deterministic_seed
    doc = emit_governed_execution_assumptions(_config()).document
    doc["content_hash"]["value"] = "sha256:" + ("0" * 64)
    with pytest.raises(ExecutionAssumptionsHashError):
        validate_governed_execution_assumptions_document(doc)

"""Integration tests for the governed execution-assumptions lane."""

from __future__ import annotations

from pathlib import Path

import pytest

from pysrc.meta.execution_assumptions_config import (
    ExecutionAssumptionsConfig,
    ExecutionParityDeclaration,
)
from pysrc.meta.execution_assumptions_emitter import emit_governed_execution_assumptions
from pysrc.meta.execution_assumptions_errors import ExecutionAssumptionsParityError
from pysrc.meta.execution_assumptions_io import (
    load_execution_assumptions,
    write_execution_assumptions,
)
from pysrc.meta.reptile_k_sweep_errors import ArtifactImmutabilityError


def _config() -> ExecutionAssumptionsConfig:
    return ExecutionAssumptionsConfig(
        run_id="run.sha256:2222222222222222222222222222222222222222222222222222222222222222",
        cost_bps=5.0,
        slippage_model="flat",
        slippage_bps_estimate=2.0,
        borrow_rate_annual_bps=None,
        latency_model="T+1 bar",
        fill_assumption="next_open",
        parity=ExecutionParityDeclaration(
            cost_assumptions_match_baseline=True,
            split_assumptions_match_baseline=True,
            data_assumptions_match_baseline=True,
            parity_note="integration parity declaration",
        ),
    )


@pytest.mark.determinism("d1")
def test_governed_execution_assumptions_round_trip_and_write_guard(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    report = emit_governed_execution_assumptions(_config())
    output_path = write_execution_assumptions(report, tmp_path)
    loaded = load_execution_assumptions(output_path)
    assert output_path.name == "execution_assumptions.json"
    assert loaded == report.document
    assert loaded["parity"]["parity_note"] == "integration parity declaration"
    with pytest.raises(ArtifactImmutabilityError):
        write_execution_assumptions(report, tmp_path)


@pytest.mark.determinism("d1")
def test_governed_execution_assumptions_rejects_false_parity(deterministic_seed: int) -> None:
    _ = deterministic_seed
    bad = ExecutionAssumptionsConfig(
        run_id="run.sha256:3333333333333333333333333333333333333333333333333333333333333333",
        cost_bps=5.0,
        slippage_model="flat",
        slippage_bps_estimate=2.0,
        borrow_rate_annual_bps=0.0,
        latency_model="T+1 bar",
        fill_assumption="next_open",
        parity=ExecutionParityDeclaration(
            cost_assumptions_match_baseline=True,
            split_assumptions_match_baseline=False,
            data_assumptions_match_baseline=True,
            parity_note="integration parity declaration",
        ),
    )
    with pytest.raises(ExecutionAssumptionsParityError):
        emit_governed_execution_assumptions(bad)

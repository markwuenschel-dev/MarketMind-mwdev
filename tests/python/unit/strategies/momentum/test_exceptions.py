from __future__ import annotations

import pytest

from pysrc.strategies.momentum.exceptions import (
    ConvergenceError,
    CostGateRejection,
    exception_metadata,
)
from pysrc.strategies.pipeline_strategy import MaterializationError

pytestmark = [pytest.mark.unit, pytest.mark.determinism("d1")]


def test_convergence_error_is_materialization_error() -> None:
    err = ConvergenceError("failed", n_iterations=10, asset_id="AAPL")
    assert isinstance(err, MaterializationError)
    assert err.n_iterations == 10
    assert err.asset_id == "AAPL"


def test_cost_gate_rejection_fields_are_exposed() -> None:
    err = CostGateRejection(
        "cost gate rejected",
        variant="xsec",
        run_id="run-1",
        reason_code="TURNOVER_LIMIT",
    )
    assert err.variant == "xsec"
    assert err.run_id == "run-1"
    assert err.reason_code == "TURNOVER_LIMIT"


def test_exception_metadata_collects_known_fields() -> None:
    err = ConvergenceError("failed", n_iterations=10, asset_id="AAPL")
    assert exception_metadata(err) == {
        "message": "failed",
        "n_iterations": 10,
        "asset_id": "AAPL",
    }


def test_exception_metadata_ignores_unknown_fields() -> None:
    assert exception_metadata(Exception("plain")) == {}

from __future__ import annotations

import json

import pytest

from pysrc.artifact_registry.artifact_store import BundleBacktestArtifactStore
from pysrc.artifact_registry.bundle_writer import BundleWriter
from pysrc.backtesting.contracts.types import BacktestResult
from pysrc.backtesting.validation.statistical.validator import StatisticalValidator


@pytest.mark.determinism("d1")
def test_stat_validator_emits_v1_report(tmp_path) -> None:
    store = BundleBacktestArtifactStore(BundleWriter(tmp_path / "bundle"))
    result = BacktestResult(metrics={"sharpe_ratio": 1.0})

    report = StatisticalValidator().validate(result, {"returns": [0.01] * 12}, store)
    payload = json.loads(
        (tmp_path / "bundle" / "stat_validity_report.json").read_text(encoding="utf-8")
    )

    assert report.status.value in {"PASS", "FAIL", "WARN"}
    assert set(payload) == {
        "schema_version",
        "sharpe_ratio",
        "dsr",
        "min_trl",
        "bootstrap_ci",
        "pbo",
        "gate_result",
    }
    assert payload["schema_version"] == "v1"

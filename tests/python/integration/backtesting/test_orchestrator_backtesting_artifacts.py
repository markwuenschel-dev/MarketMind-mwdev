from __future__ import annotations

import json

import pytest

from pysrc.pipeline.orchestrator import OrchestratorConfig, run_orchestration


@pytest.mark.determinism("d1")
def test_canonical_orchestrator_emits_additive_backtesting_artifacts(tmp_path) -> None:
    csv_path = tmp_path / "prices.csv"
    csv_path.write_text(
        "date,close,symbol\n2024-01-01,100,SPY\n2024-01-02,101,SPY\n2024-01-03,102,SPY\n2024-01-04,103,SPY\n",
        encoding="utf-8",
    )

    exit_code, payload = run_orchestration(
        OrchestratorConfig(
            input_path=csv_path, bundle_dir=tmp_path / "bundle", fast_sma=2, slow_sma=3
        )
    )

    assert exit_code in {0, 1}
    assert payload["success"] is True
    assert (tmp_path / "bundle" / "plan.json").exists()
    assert (tmp_path / "bundle" / "execution_assumptions.json").exists()
    assert (tmp_path / "bundle" / "stat_validity_report.json").exists()
    # Phase I-E: screening_report.json present for every strategy evaluation run
    screening_path = tmp_path / "bundle" / "screening_report.json"
    assert screening_path.exists(), "screening_report.json must be emitted by orchestrator"
    screening = json.loads(screening_path.read_text(encoding="utf-8"))
    assert "schema_version" in screening
    assert "candidates" in screening
    assert "summary" in screening
    assert screening["summary"]["total_candidates"] >= 1
    result = json.loads((tmp_path / "bundle" / "backtest_result.json").read_text(encoding="utf-8"))
    assert "result" in result

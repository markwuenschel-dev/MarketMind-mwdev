"""Gate 7 promotion bundle and finish-line tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pysrc.cli.gate import ExitCode, validate_bundle
from pysrc.pipeline.candidate_portfolios.promotion_bundle import (
    assemble_promotion_bundle,
    build_splits_manifest_from_run,
)
from pysrc.pipeline.candidate_portfolios.promotion_stats import (
    build_crisis_holdout_report,
    build_promotion_model_ledger,
)
from pysrc.pipeline.p2_config_loader import PortfolioSpec


def _write_gate6_report(reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "gate_pass": True,
        "stat_validity_report": {
            "schema_version": "v1",
            "sharpe_ratio": 2.0,
            "dsr": {
                "value": 1.5,
                "p_value": 0.01,
                "n_trials": 3,
                "skewness": 0.0,
                "excess_kurtosis": 0.0,
                "gate_result": "PASS",
            },
            "min_trl": {
                "years_needed": 1.0,
                "years_available": 3.0,
                "target_confidence": 0.95,
                "gate_result": "PASS",
            },
            "bootstrap_ci": {
                "lower_95": 0.5,
                "upper_95": 3.0,
                "lower_99": 0.2,
                "upper_99": 3.5,
                "n_resamples": 100,
                "block_size": 5,
                "gate_result": "PASS",
            },
            "pbo": {"value": 0.1, "gate_result": "PASS"},
            "gate_result": "PASS",
        },
    }
    (reports_dir / "gate6_promotion_report.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_gate4_robustness(reports_dir: Path) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "upgraded_outcome": "D",
        "rationale": "fixture",
        "by_fold": {
            "fold_0": {
                "xgboost": {"net_sharpe": 2.0, "cumulative_log_return": 1.0, "n_days": 10},
                "ridge": {"net_sharpe": 0.5, "cumulative_log_return": 0.1, "n_days": 10},
                "equal_blend": {"net_sharpe": 1.5, "cumulative_log_return": 0.5, "n_days": 10},
            },
            "fold_1": {
                "xgboost": {"net_sharpe": 2.5, "cumulative_log_return": 1.2, "n_days": 10},
                "ridge": {"net_sharpe": 0.4, "cumulative_log_return": 0.0, "n_days": 10},
                "equal_blend": {"net_sharpe": 1.8, "cumulative_log_return": 0.6, "n_days": 10},
            },
        },
    }
    (reports_dir / "gate4_robustness_report.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_predictions(run_dir: Path) -> None:
    rows: list[dict[str, object]] = []
    for fold_id, dates in (
        ("fold_0", ("2024-01-02", "2024-01-03")),
        ("fold_1", ("2024-02-02", "2024-02-03")),
    ):
        for date in dates:
            rows.append(
                {
                    "date": date,
                    "instrument": "AAA",
                    "model_id": "xgboost",
                    "prediction": 0.1,
                    "fold_id": fold_id,
                    "split": "test",
                }
            )
    pred_dir = run_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(pred_dir / "model_prediction_panel.parquet", index=False)


@pytest.mark.determinism("d1")
def test_build_promotion_model_ledger_includes_all_models(
    deterministic_seed: int, tmp_path: Path
) -> None:
    _ = deterministic_seed
    run_dir = tmp_path / "run"
    reports = run_dir / "reports"
    _write_gate4_robustness(reports)
    ledger = build_promotion_model_ledger(run_dir, selected_model="xgboost")

    assert ledger["selected_model"] == "xgboost"
    assert "xgboost" in ledger["models"]
    assert ledger["models"]["xgboost"]["promotion_status"] == "selected"
    assert ledger["models"]["equal_blend"]["promotion_status"] == "comparator"
    assert len(ledger["ranking_by_pooled_sharpe"]) >= 2


@pytest.mark.determinism("d1")
def test_crisis_holdout_excludes_reserved_windows(deterministic_seed: int) -> None:
    _ = deterministic_seed
    dates = pd.date_range("2008-08-01", periods=400, freq="B").astype(str).tolist()
    returns = np.random.default_rng(1).normal(0.001, 0.01, size=len(dates))
    report = build_crisis_holdout_report(returns, dates)

    assert report["excluded_day_count"] > 0
    assert report["holdout_day_count"] < len(dates)
    assert "holdout_net_sharpe" in report


@pytest.mark.determinism("d1")
def test_splits_manifest_from_predictions(deterministic_seed: int, tmp_path: Path) -> None:
    _ = deterministic_seed
    run_dir = tmp_path / "run"
    _write_predictions(run_dir)
    manifest = build_splits_manifest_from_run(run_dir)

    assert manifest["split_method"] == "walk_forward"
    assert len(manifest["splits"]) == 2
    assert manifest["splits"][0]["train_end"] < manifest["splits"][0]["test_start"]


@pytest.mark.determinism("d1")
def test_assemble_bundle_passes_mm_gate(deterministic_seed: int, tmp_path: Path) -> None:
    _ = deterministic_seed
    run_dir = tmp_path / "run"
    reports = run_dir / "reports"
    _write_gate6_report(reports)
    _write_gate4_robustness(reports)
    _write_predictions(run_dir)

    panel_path = tmp_path / "panel.parquet"
    pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03", "2024-02-02", "2024-02-03"],
            "instrument": ["AAA"] * 4,
            "forward_return_1d": [0.01, 0.02, -0.01, 0.015],
        }
    ).to_parquet(panel_path, index=False)

    spec = PortfolioSpec(top_k=1, single_name_cap=1.0, cost_bps=10.0, capacity_constraints=False)
    ledger = build_promotion_model_ledger(run_dir, selected_model="xgboost")
    (reports / "promotion_model_ledger.json").write_text(json.dumps(ledger), encoding="utf-8")

    bundle_dir = assemble_promotion_bundle(
        run_dir,
        panel_path=panel_path,
        portfolio_spec=spec,
        model_id="xgboost",
    )

    assert (bundle_dir / "stat_validity_report.json").is_file()
    assert (bundle_dir / "execution_assumptions.json").is_file()
    _, exit_code = validate_bundle(bundle_dir)
    assert exit_code == ExitCode.PASS

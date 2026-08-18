"""Unit tests for paper loop dry-run skeleton (PDR-003 Phase B)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from pysrc.tuning.execution.paper_loop import (
    _MAX_PAPER_BARS,
    paper_loop_dry_run,
    paper_trading_enabled,
)


def _write_bundle(bundle_dir: Path) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "strategy": "prediction_threshold_xgboost",
        "model_id": "xgboost",
        "source_run_id": "test-run",
        "portfolio": {"top_k": 2, "single_name_cap": 0.1, "cost_bps": 10.0},
    }
    (bundle_dir / "plan.json").write_text(json.dumps(plan), encoding="utf-8")


def _write_predictions(run_dir: Path) -> None:
    rows: list[dict[str, object]] = []
    for day in ("2024-01-02", "2024-01-03", "2024-01-04"):
        for instrument, pred in [("AAA", 0.3), ("BBB", 0.2), ("CCC", -0.1)]:
            rows.append(
                {
                    "date": day,
                    "instrument": instrument,
                    "model_id": "xgboost",
                    "prediction": pred,
                    "fold_id": "fold_0",
                    "split": "test",
                }
            )
    pred_dir = run_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(pred_dir / "model_prediction_panel.parquet", index=False)


def _write_xgboost_fixture(run_dir: Path) -> None:
    model_dir = run_dir / "models" / "xgboost"
    model_dir.mkdir(parents=True, exist_ok=True)
    feature_names = ["feat_a", "feat_b"]
    model = LinearRegression()
    model.fit(np.eye(len(feature_names), dtype=np.float64), np.ones(len(feature_names)))
    artifact_name = "model.joblib"
    joblib.dump(model, model_dir / artifact_name)
    manifest = {
        "artifact_name": artifact_name,
        "feature_names": feature_names,
        "model_id": "xgboost",
    }
    (model_dir / "model_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.determinism("d1")
def test_paper_loop_dry_run_emits_bar_log(tmp_path: Path, deterministic_seed: int) -> None:
    _ = deterministic_seed
    run_dir = tmp_path / "test-run"
    bundle_dir = tmp_path / "bundle"
    _write_bundle(bundle_dir)
    _write_predictions(run_dir)
    _write_xgboost_fixture(run_dir)

    payload = paper_loop_dry_run(bundle_dir=bundle_dir, run_dir=run_dir, n_bars=2)

    assert payload["schema_version"] == "paper_loop_log.v1"
    assert payload["paper_trading_enabled"] is False
    assert payload["ibkr_submit"] is False
    assert len(payload["bars"]) == 2
    assert payload["bars"][0]["reconciliation"]["has_mismatch"] is False
    assert payload["kill_switch_engaged"] is False
    report = json.loads((run_dir / "reports" / "paper_loop_log.json").read_text(encoding="utf-8"))
    assert report["total_orders_submitted"] > 0


@pytest.mark.determinism("d1")
def test_paper_trading_enabled_defaults_off(
    monkeypatch: pytest.MonkeyPatch, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    monkeypatch.delenv("PAPER_TRADING_ENABLED", raising=False)
    assert paper_trading_enabled() is False


@pytest.mark.determinism("d1")
def test_paper_loop_rejects_n_bars_above_cap(deterministic_seed: int) -> None:
    _ = deterministic_seed
    with pytest.raises(ValueError, match=str(_MAX_PAPER_BARS)):
        paper_loop_dry_run(
            bundle_dir=Path("bundle"),
            run_dir=Path("run"),
            n_bars=_MAX_PAPER_BARS + 1,
        )


@pytest.mark.determinism("d1")
def test_paper_loop_raises_when_live_submit_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    deterministic_seed: int,
) -> None:
    _ = deterministic_seed
    monkeypatch.setenv("PAPER_TRADING_ENABLED", "1")
    run_dir = tmp_path / "test-run"
    bundle_dir = tmp_path / "bundle"
    _write_bundle(bundle_dir)
    _write_predictions(run_dir)
    _write_xgboost_fixture(run_dir)

    with pytest.raises(RuntimeError, match="Kotlin execution service"):
        paper_loop_dry_run(bundle_dir=bundle_dir, run_dir=run_dir, n_bars=1)

    monkeypatch.delenv("PAPER_TRADING_ENABLED", raising=False)
    os.environ.pop("PAPER_TRADING_ENABLED", None)

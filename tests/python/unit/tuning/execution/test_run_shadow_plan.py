"""Unit tests for shadow replay plan v2."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pysrc.tuning.execution.run_shadow_plan import _MAX_SHADOW_DAYS, run_shadow_plan


def _write_bundle(bundle_dir: Path, *, panel_path: Path) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "strategy": "prediction_threshold_xgboost",
        "model_id": "xgboost",
        "source_run_id": "test-run",
        "panel_path": str(panel_path),
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


@pytest.mark.determinism("d1")
def test_run_shadow_plan_emits_per_bar_provenance(tmp_path: Path, deterministic_seed: int) -> None:
    _ = deterministic_seed
    run_dir = tmp_path / "test-run"
    bundle_dir = tmp_path / "bundle"
    panel_path = tmp_path / "panel.parquet"
    pd.DataFrame({"date": ["2024-01-02"], "instrument": ["AAA"]}).to_parquet(
        panel_path, index=False
    )
    _write_bundle(bundle_dir, panel_path=panel_path)
    _write_predictions(run_dir)

    payload = run_shadow_plan(bundle_dir=bundle_dir, run_dir=run_dir, n_days=2)
    assert payload["schema_version"] == "shadow_replay_log.v2"
    assert len(payload["bars"]) == 2
    assert payload["bars"][0]["bar_index"] == 0
    assert payload["bars"][0]["instrument_count"] == 3
    report = json.loads(
        (run_dir / "reports" / "shadow_replay_log.json").read_text(encoding="utf-8")
    )
    assert report["bars"][1]["date"] == "2024-01-04"


@pytest.mark.determinism("d1")
def test_run_shadow_plan_rejects_n_days_above_cap(deterministic_seed: int) -> None:
    _ = deterministic_seed
    with pytest.raises(ValueError, match=str(_MAX_SHADOW_DAYS)):
        run_shadow_plan(
            bundle_dir=Path("bundle"),
            run_dir=Path("run"),
            n_days=_MAX_SHADOW_DAYS + 1,
        )

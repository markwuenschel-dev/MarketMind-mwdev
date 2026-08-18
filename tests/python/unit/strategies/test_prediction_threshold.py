"""Tests for prediction-threshold strategy bridge."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pysrc.strategies.prediction_threshold import (
    load_promotion_bundle_runtime,
    predictions_to_intents_from_bundle,
)


@pytest.mark.determinism("d1")
def test_bundle_runtime_and_intents(deterministic_seed: int, tmp_path: Path) -> None:
    _ = deterministic_seed
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    plan = {
        "model_id": "xgboost",
        "strategy": "prediction_threshold_xgboost",
        "source_run_id": "run_test",
        "plan_hash": "plan123",
        "portfolio": {"top_k": 1, "single_name_cap": 1.0, "cost_bps": 10.0},
    }
    (bundle / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    runtime = load_promotion_bundle_runtime(bundle)
    assert runtime.strategy_id == "prediction_threshold_xgboost"
    preds = pd.DataFrame(
        {
            "date": ["2024-01-02"],
            "instrument": ["AAA"],
            "prediction": [0.1],
            "fold_id": ["fold_0"],
            "split": ["test"],
            "model_id": ["xgboost"],
        }
    )
    intents = predictions_to_intents_from_bundle(preds, runtime)
    assert not intents.empty
    assert intents["strategy_id"].iloc[0] == "prediction_threshold_xgboost"

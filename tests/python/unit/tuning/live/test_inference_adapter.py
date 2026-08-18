"""Unit tests for InferenceAdapter (PDR-003 Wave 3)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from pysrc.data.dataview import DataView
from pysrc.tuning.live.inference_adapter import InferenceAdapter


class _FixedPredictor:
    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return np.asarray([float(matrix.sum())], dtype=np.float64)


def _write_xgboost_fixture(run_dir: Path, *, feature_names: list[str]) -> None:
    model_dir = run_dir / "models" / "xgboost"
    model_dir.mkdir(parents=True, exist_ok=True)
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


def _feature_dataview() -> DataView:
    dv = DataView()
    rows = [
        {
            "symbol": "AAA",
            "valid_time": date(2024, 1, 1),
            "knowledge_time": date(2024, 1, 2),
            "feat_a": 1.0,
            "feat_b": 2.0,
        },
        {
            "symbol": "BBB",
            "valid_time": date(2024, 1, 1),
            "knowledge_time": date(2024, 1, 2),
            "feat_a": 3.0,
            "feat_b": 4.0,
        },
    ]
    dv.register_source(pd.DataFrame(rows))
    return dv


@pytest.mark.determinism("d1")
def test_load_xgboost_from_run_reads_manifest(tmp_path: Path, deterministic_seed: int) -> None:
    _ = deterministic_seed
    feature_names = ["feat_a", "feat_b"]
    _write_xgboost_fixture(tmp_path, feature_names=feature_names)
    adapter = InferenceAdapter()
    model, features = adapter.load_xgboost_from_run(tmp_path)
    assert features == feature_names
    assert hasattr(model, "predict")


@pytest.mark.determinism("d1")
def test_assemble_features_as_of_uses_dataview_pit(tmp_path: Path, deterministic_seed: int) -> None:
    _ = deterministic_seed
    adapter = InferenceAdapter()
    frame = adapter.assemble_features_as_of(
        _feature_dataview(),
        symbols=["AAA", "BBB"],
        feature_names=["feat_a", "feat_b"],
        knowledge_date=date(2024, 1, 2),
    )
    assert list(frame.columns) == ["date", "instrument", "feat_a", "feat_b"]
    assert len(frame) == 2
    assert set(frame["instrument"]) == {"AAA", "BBB"}


@pytest.mark.determinism("d1")
def test_predict_single_bar_batch_with_feature_list(deterministic_seed: int) -> None:
    _ = deterministic_seed
    adapter = InferenceAdapter()
    result = adapter.predict(
        {"feat_a": 1.0, "feat_b": 2.0},
        _FixedPredictor(),
        feature_names=["feat_a", "feat_b"],
    )
    assert result == {"prediction": 3.0, "confidence": 1.0}


@pytest.mark.determinism("d1")
def test_predict_rejects_multi_row_batch(deterministic_seed: int) -> None:
    _ = deterministic_seed
    adapter = InferenceAdapter()
    frame = pd.DataFrame(
        [
            {"feat_a": 1.0, "feat_b": 2.0},
            {"feat_a": 3.0, "feat_b": 4.0},
        ]
    )
    with pytest.raises(ValueError, match="single-bar batch"):
        adapter.predict(frame, _FixedPredictor(), feature_names=["feat_a", "feat_b"])

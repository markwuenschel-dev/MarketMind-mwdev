"""InferenceAdapter: translates feature batches into model predictions."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd

from pysrc.data.dataview import DataView

_XGBOOST_MODEL_ID = "xgboost"


class InferenceAdapter:
    """Thin adapter between replay/live feature rows and a trained panel model."""

    def load_checkpoint(self, artifact_path: str | Path) -> Any:
        path = Path(artifact_path)
        if not path.is_file():
            raise FileNotFoundError(f"Model artifact not found: {path}")
        if path.suffix == ".json":
            meta = json.loads(path.read_text(encoding="utf-8"))
            model_path = Path(str(meta.get("model_path", "")))
            if not model_path.is_file():
                raise FileNotFoundError(f"Referenced model_path missing: {model_path}")
            return joblib.load(model_path)
        return joblib.load(path)

    def load_from_run(
        self, run_dir: Path, model_id: str = _XGBOOST_MODEL_ID
    ) -> tuple[Any, list[str]]:
        """Load a trained model artifact and feature list from ``run_dir/models/<model_id>/``."""

        run_dir = Path(run_dir)
        model_dir = run_dir / "models" / model_id
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Missing model directory: {model_dir}")
        manifest_path = model_dir / "model_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            artifact = model_dir / str(manifest.get("artifact_name", "model.joblib"))
            features = [str(name) for name in (manifest.get("feature_names") or [])]
            return self.load_checkpoint(artifact), features
        artifact = model_dir / "model.joblib"
        return self.load_checkpoint(artifact), []

    def load_xgboost_from_run(self, run_dir: Path) -> tuple[Any, list[str]]:
        """Load the promoted xgboost checkpoint from ``run_dir/models/xgboost/``."""

        return self.load_from_run(run_dir, _XGBOOST_MODEL_ID)

    def assemble_features_as_of(
        self,
        dataview: DataView,
        *,
        symbols: Sequence[str],
        feature_names: Sequence[str],
        knowledge_date: date,
    ) -> pd.DataFrame:
        """Build a single-bar feature batch using ``DataView.as_of`` PIT semantics."""

        if not feature_names:
            raise ValueError("feature_names must be non-empty for assemble_features_as_of")
        snapshot = dataview.as_of(list(symbols), list(feature_names), knowledge_date)
        if snapshot.empty:
            return pd.DataFrame(columns=["date", "instrument", *feature_names])
        frame = snapshot.copy()
        frame["date"] = knowledge_date.isoformat()
        if "symbol" in frame.columns:
            frame = frame.rename(columns={"symbol": "instrument"})
        keep = ["date", "instrument", *feature_names]
        return frame.loc[:, [col for col in keep if col in frame.columns]].reset_index(drop=True)

    def predict(
        self,
        features: pd.DataFrame | dict[str, Any],
        model_handle: Any,
        *,
        feature_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Score one single-bar batch using an explicit feature column list."""

        frame = pd.DataFrame([features]) if isinstance(features, dict) else features.copy()
        columns = feature_names or [c for c in frame.columns if c not in {"date", "instrument"}]
        if not columns:
            raise ValueError("predict requires at least one feature column")
        missing = [name for name in columns if name not in frame.columns]
        if missing:
            raise ValueError(f"predict missing feature columns: {missing}")
        matrix = frame.loc[:, columns].to_numpy(dtype=np.float64)
        if matrix.shape[0] != 1:
            raise ValueError(f"predict expects a single-bar batch, got {matrix.shape[0]} rows")
        if hasattr(model_handle, "predict"):
            values = np.asarray(model_handle.predict(matrix), dtype=np.float64).reshape(-1)
        else:
            raise TypeError(
                f"model_handle {type(model_handle).__name__!r} lacks predict(); "
                "pass a joblib-loaded sklearn/xgboost model"
            )
        confidence = np.ones_like(values)
        if hasattr(model_handle, "predict_confidence"):
            confidence = np.asarray(
                model_handle.predict_confidence(matrix), dtype=np.float64
            ).reshape(-1)
        return {
            "prediction": float(values[-1]),
            "confidence": float(confidence[-1]),
        }


__all__ = ["InferenceAdapter"]

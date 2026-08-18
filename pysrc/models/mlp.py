"""Sklearn MLP regressor implementing PanelModel."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from pysrc.models.base import PanelModel


class MlpPanelModel:
    """Small MLP on flattened panel features."""

    def __init__(
        self,
        *,
        model_id: str = "mlp",
        hidden_layer_sizes: tuple[int, ...] = (64, 32),
        alpha: float = 0.001,
        random_seed: int = 42,
        max_iter: int = 200,
    ) -> None:
        self.model_id = model_id
        self._scaler = StandardScaler(copy=False)
        self._estimator = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=alpha,
            random_state=random_seed,
            max_iter=max_iter,
            early_stopping=True,
            validation_fraction=0.1,
        )
        self._feature_names: list[str] = []
        self._residual_std: float = 1.0

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, *, fold_id: str) -> None:
        from pysrc.models.tabular import _clean_feature_matrix, _residual_std_from_estimator

        del fold_id
        x_clean = _clean_feature_matrix(x_train)
        x_scaled = self._scaler.fit_transform(x_clean)
        self._estimator.fit(x_scaled, y_train.ravel())
        self._residual_std = _residual_std_from_estimator(self._estimator, x_scaled, y_train)

    def predict(self, x: np.ndarray) -> np.ndarray:
        from pysrc.models.tabular import _clean_feature_matrix

        x_clean = _clean_feature_matrix(x)
        x_scaled = self._scaler.transform(x_clean)
        return self._estimator.predict(x_scaled)

    def predict_with_confidence(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from pysrc.models.tabular import _confidence_from_predictions

        preds = self.predict(x).reshape(-1)
        return preds, _confidence_from_predictions(preds, self._residual_std)

    def predict_confidence(self, x: np.ndarray) -> np.ndarray:
        _, confidence = self.predict_with_confidence(x)
        return confidence

    def feature_usage(self) -> list[str]:
        return list(self._feature_names)

    def set_feature_names(self, names: list[str]) -> None:
        self._feature_names = list(names)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model_id": self.model_id,
                "estimator": self._estimator,
                "scaler": self._scaler,
                "feature_names": self._feature_names,
                "residual_std": self._residual_std,
            },
            path,
        )

    def load(self, path: Path) -> PanelModel:
        payload = joblib.load(path)
        self.model_id = str(payload["model_id"])
        self._estimator = payload["estimator"]
        self._scaler = payload["scaler"]
        self._feature_names = list(payload.get("feature_names", []))
        self._residual_std = float(payload.get("residual_std", 1.0))
        return self


def create_mlp_model(
    *,
    model_id: str = "mlp",
    params: dict[str, object] | None = None,
    random_seed: int = 42,
) -> MlpPanelModel:
    hp = params or {}
    hidden = hp.get("hidden_layer_sizes", (64, 32))
    if isinstance(hidden, list):
        hidden = tuple(int(x) for x in hidden)
    return MlpPanelModel(
        model_id=model_id,
        hidden_layer_sizes=hidden,
        alpha=float(hp.get("alpha", 0.001)),
        random_seed=random_seed,
        max_iter=int(hp.get("max_iter", 200)),
    )


__all__ = ["MlpPanelModel", "create_mlp_model"]

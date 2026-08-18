"""Sklearn tabular regressors for router and panel experiments."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import BayesianRidge, ElasticNet, QuantileRegressor, Ridge
from sklearn.metrics import r2_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from pysrc.contracts.candidate_spec import CandidateSpec

_FAMILY_ALIASES: dict[str, str] = {"pcr_ridge": "pcr"}

_LINEAR_FAMILIES = frozenset(
    {
        "ridge",
        "elastic_net",
        "bayesian_ridge",
        "pcr",
        "pls",
        "quantile_regression",
    }
)

_RESIDUAL_CHUNK_ROWS = 250_000


def _clean_feature_matrix(x: np.ndarray) -> np.ndarray:
    array = np.asarray(x)
    if not np.issubdtype(array.dtype, np.floating):
        array = np.asarray(x, dtype=np.float32)
    elif not array.flags.writeable:
        array = np.array(array, copy=True)
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0, copy=False)


def _confidence_from_predictions(preds: np.ndarray, residual_std: float) -> np.ndarray:
    flat_preds = np.asarray(preds, dtype=np.float64).reshape(-1)
    std = max(float(residual_std), 1e-6)
    return np.clip(1.0 / (1.0 + np.abs(flat_preds) / std), 0.0, 1.0)


def _residual_std_from_estimator(
    model: RegressorMixin,
    x_scaled: np.ndarray,
    y_train: np.ndarray,
) -> float:
    y_flat = np.asarray(y_train).reshape(-1)
    if len(y_flat) == 0:
        return 1.0
    residual_sum = 0.0
    residual_sum_sq = 0.0
    residual_count = 0
    for start in range(0, len(y_flat), _RESIDUAL_CHUNK_ROWS):
        end = min(start + _RESIDUAL_CHUNK_ROWS, len(y_flat))
        preds = np.asarray(model.predict(x_scaled[start:end]), dtype=np.float64).reshape(-1)
        residuals = np.asarray(y_flat[start:end], dtype=np.float64) - preds
        residual_sum += float(np.sum(residuals, dtype=np.float64))
        residual_sum_sq += float(np.dot(residuals, residuals))
        residual_count += int(residuals.size)
    if residual_count == 0:
        return 1.0
    residual_mean = residual_sum / residual_count
    variance = max((residual_sum_sq / residual_count) - residual_mean * residual_mean, 0.0)
    return float(np.sqrt(variance))


TABULAR_FAMILIES: frozenset[str] = frozenset(
    {
        "ridge",
        "elastic_net",
        "bayesian_ridge",
        "pcr",
        "pls",
        "quantile_regression",
        "random_forest",
        "extra_trees",
        "xgboost",
        "pcr_ridge",  # backward-compatible alias
    }
)


def _resolve_tabular_family(name: str) -> str:
    """Validate a tabular-only model family name."""

    canonical = _FAMILY_ALIASES.get(name, name)
    if canonical not in TABULAR_FAMILIES and name not in TABULAR_FAMILIES:
        raise ValueError(f"Unsupported tabular model_family: {name}")
    return canonical


def get_model_instance(
    candidate: CandidateSpec,
    random_seed: int,
    *,
    sklearn_n_jobs: int = 1,
) -> RegressorMixin:
    name = _resolve_tabular_family(candidate.model_family)
    hp = candidate.hyperparams or {}
    seed = int(random_seed)

    if name == "ridge":
        return Ridge(alpha=hp.get("alpha", 1.0))
    if name == "elastic_net":
        return ElasticNet(
            alpha=hp.get("alpha", 1.0),
            l1_ratio=hp.get("l1_ratio", 0.5),
            random_state=seed,
        )
    if name == "bayesian_ridge":
        return BayesianRidge(**{k: v for k, v in hp.items() if k in ["alpha_1", "alpha_2"]})
    if name == "pcr":
        from sklearn.decomposition import PCA
        from sklearn.pipeline import Pipeline

        n_comp = int(hp.get("n_components", 5))
        return Pipeline(
            [
                ("pca", PCA(n_components=min(n_comp, 20), random_state=seed)),
                ("ridge", Ridge(alpha=hp.get("alpha", 1.0))),
            ]
        )
    if name == "quantile_regression":
        backend = str(hp.get("backend", "linear_program"))
        if backend == "hist_gradient_boosting":
            return HistGradientBoostingRegressor(
                loss="quantile",
                quantile=float(hp.get("quantile", 0.5)),
                learning_rate=float(hp.get("learning_rate", 0.05)),
                max_iter=int(hp.get("max_iter", 100)),
                max_leaf_nodes=int(hp.get("max_leaf_nodes", 31)),
                min_samples_leaf=int(hp.get("min_samples_leaf", 20)),
                l2_regularization=float(hp.get("l2_regularization", 0.0)),
                random_state=seed,
            )
        if backend != "linear_program":
            raise ValueError(
                "quantile_regression backend must be 'linear_program' or "
                f"'hist_gradient_boosting', got {backend!r}"
            )
        return QuantileRegressor(
            quantile=float(hp.get("quantile", 0.5)),
            alpha=float(hp.get("alpha", 1.0)),
            solver=str(hp.get("solver", "highs")),
        )
    if name == "pls":
        from sklearn.cross_decomposition import PLSRegression

        return PLSRegression(n_components=hp.get("n_components", 5))
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=hp.get("n_estimators", 200),
            max_depth=hp.get("max_depth", 6),
            min_samples_leaf=hp.get("min_samples_leaf", 20),
            random_state=seed,
            n_jobs=sklearn_n_jobs,
        )
    if name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=hp.get("n_estimators", 200),
            max_depth=hp.get("max_depth", 6),
            min_samples_leaf=hp.get("min_samples_leaf", 20),
            random_state=seed,
            n_jobs=sklearn_n_jobs,
        )
    if name == "xgboost":
        try:
            from xgboost import XGBRegressor

            return XGBRegressor(
                n_estimators=hp.get("n_estimators", 150),
                max_depth=hp.get("max_depth", 4),
                learning_rate=hp.get("learning_rate", 0.05),
                subsample=hp.get("subsample", 0.8),
                tree_method=str(hp.get("tree_method", "hist")),
                random_state=seed,
                n_jobs=sklearn_n_jobs,
                verbosity=0,
            )
        except ImportError as err:
            raise RuntimeError("xgboost not installed.") from err
    raise ValueError(f"Unsupported model_family: {name}")


def train_and_predict(
    candidate: CandidateSpec,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    x_test: np.ndarray,
    random_seed: int,
    *,
    sklearn_n_jobs: int = 1,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    model = get_model_instance(candidate, random_seed, sklearn_n_jobs=sklearn_n_jobs)
    if candidate.model_family in _LINEAR_FAMILIES:
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_validation_scaled = (
            scaler.transform(x_validation)
            if x_validation.size
            else np.empty((0, x_train.shape[1]), dtype=np.float64)
        )
        x_test_scaled = scaler.transform(x_test)
    else:
        x_train_scaled, x_validation_scaled, x_test_scaled = x_train, x_validation, x_test

    if candidate.router_target in {"child_utility_regression", "pairwise_advantage_regression"}:
        if y_train.shape[1] == 1:
            model.fit(x_train_scaled, y_train.ravel())
            preds_test = model.predict(x_test_scaled)
            if preds_test.ndim == 1:
                preds_test = preds_test.reshape(-1, 1)
            preds_validation = (
                model.predict(x_validation_scaled) if x_validation_scaled.size else np.empty((0, 1))
            )
            if preds_validation.ndim == 1:
                preds_validation = preds_validation.reshape(-1, 1)
            train_preds = model.predict(x_train_scaled).reshape(-1, 1)
        else:
            mo_model = MultiOutputRegressor(model, n_jobs=1)
            mo_model.fit(x_train_scaled, y_train)
            preds_test = mo_model.predict(x_test_scaled)
            preds_validation = (
                mo_model.predict(x_validation_scaled)
                if x_validation_scaled.size
                else np.empty((0, y_train.shape[1]))
            )
            train_preds = mo_model.predict(x_train_scaled)
    else:
        raise NotImplementedError(
            f"router_target={candidate.router_target} is not implemented for tabular v1"
        )

    r2_value = float(r2_score(y_train, train_preds, multioutput="uniform_average"))
    if not np.isfinite(r2_value):
        r2_value = 0.0
    return (
        preds_test,
        preds_validation,
        {
            "train_r2": r2_value,
            "n_features": float(x_train.shape[1]),
        },
    )


class TabularPanelModel:
    """Wrap sklearn tabular estimators behind PanelModel."""

    def __init__(
        self,
        *,
        model_id: str,
        family: str,
        params: dict[str, object] | None = None,
        random_seed: int = 42,
        sklearn_n_jobs: int = 1,
    ) -> None:
        from pysrc.contracts.candidate_spec import CandidateSpec

        self.model_id = model_id
        self._family = _resolve_tabular_family(family)
        self._params = dict(params or {})
        self._random_seed = random_seed
        self._sklearn_n_jobs = max(1, int(sklearn_n_jobs))
        self._feature_names: list[str] = []
        self._estimator: RegressorMixin | None = None
        self._scaler: StandardScaler | None = None
        self._residual_std = 1.0
        self._candidate = CandidateSpec(
            candidate_id=f"panel__{family}",
            model_family=family,
            router_target="child_utility_regression",
            decision_rule="free_routing",
            input_surface="full_indicator_feature_panel",
            feature_allowlist="full_indicator_universe_v1",
            split_policy="w4a_fold_split",
            status="active",
            hyperparams=self._params,
            feature_policy="full_indicator_universe_v1",
        )

    def _fit_params(self, n_features: int) -> dict[str, object]:
        params = dict(self._params)
        if self._family in {"pcr", "pls"}:
            n_comp = int(params.get("n_components", 5))
            params["n_components"] = max(1, min(n_comp, n_features))
        return params

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, *, fold_id: str) -> None:
        del fold_id
        x_clean = _clean_feature_matrix(x_train)
        fit_params = self._fit_params(x_clean.shape[1])
        from pysrc.contracts.candidate_spec import CandidateSpec

        candidate = CandidateSpec(
            candidate_id=self._candidate.candidate_id,
            model_family=self._family,
            router_target=self._candidate.router_target,
            decision_rule=self._candidate.decision_rule,
            input_surface=self._candidate.input_surface,
            feature_allowlist=self._candidate.feature_allowlist,
            split_policy=self._candidate.split_policy,
            status=self._candidate.status,
            hyperparams=fit_params,
            feature_policy=self._candidate.feature_policy,
        )
        model = get_model_instance(
            candidate,
            self._random_seed,
            sklearn_n_jobs=self._sklearn_n_jobs,
        )
        if self._family in _LINEAR_FAMILIES:
            scaler = StandardScaler(copy=False)
            x_scaled = scaler.fit_transform(x_clean)
            self._scaler = scaler
        else:
            x_scaled = x_clean
            self._scaler = None
        model.fit(x_scaled, y_train.ravel())
        self._estimator = model
        self._residual_std = _residual_std_from_estimator(model, x_scaled, y_train)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self._estimator is None:
            raise RuntimeError("Model not fitted")
        x_clean = _clean_feature_matrix(x)
        x_in = self._scaler.transform(x_clean) if self._scaler is not None else x_clean
        return self._estimator.predict(x_in)

    def predict_with_confidence(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
        import joblib

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model_id": self.model_id,
                "family": self._family,
                "params": self._params,
                "estimator": self._estimator,
                "scaler": self._scaler,
                "feature_names": self._feature_names,
                "residual_std": self._residual_std,
            },
            path,
        )

    def load(self, path: Path) -> TabularPanelModel:
        import joblib

        payload = joblib.load(path)
        self.model_id = str(payload["model_id"])
        self._family = str(payload["family"])
        self._params = dict(payload.get("params", {}))
        self._estimator = payload["estimator"]
        self._scaler = payload.get("scaler")
        self._feature_names = list(payload.get("feature_names", []))
        self._residual_std = float(payload.get("residual_std", 1.0))
        return self


__all__ = [
    "TABULAR_FAMILIES",
    "TabularPanelModel",
    "get_model_instance",
    "train_and_predict",
]

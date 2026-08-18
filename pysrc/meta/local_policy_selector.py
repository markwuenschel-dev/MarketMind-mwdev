"""First local MetaRouter: boring on purpose.

Implements two complementary policies over the same predictions:

A. Pairwise/threshold override selector — one ridge regression per candidate
   predicting ``delta_utility_vs_default``; route to the best candidate only
   when its predicted advantage clears an uncertainty + cost buffer, otherwise
   abstain to the default blend. A no-switch margin adds inertia.

B. Softmax trust allocator — ``a_t = softmax(temperature * predicted_delta)``
   with exponential smoothing ``a_t = (1-rho) a_{t-1} + rho a_t``.

No deep models. The point is to prove the data contract and utility target.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pysrc.contracts.meta_router import (
    CASH_CANDIDATE_ID,
    PREDICTION_COLUMNS,
    ROUTE_DECISION_COLUMNS,
    TRAINING_TARGET_COLUMN,
    MetaRouterConfig,
    select_feature_columns,
)


@dataclass
class TrainedSelector:
    """Per-candidate ridge models with train-fit feature standardization."""

    feature_columns: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    default_candidate_id: str
    models: dict[str, object] = field(default_factory=dict)
    feature_means: dict[str, np.ndarray] = field(default_factory=dict)
    feature_stds: dict[str, np.ndarray] = field(default_factory=dict)
    residual_std: dict[str, float] = field(default_factory=dict)


def _standardize(features: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    return np.asarray(np.nan_to_num((features - means) / stds, nan=0.0), dtype=np.float64)


def train_local_policy_selector(
    frame: pd.DataFrame,
    config: MetaRouterConfig,
) -> TrainedSelector:
    """Fit one ridge per non-default candidate on train rows only."""

    from sklearn.linear_model import Ridge

    feature_columns = tuple(select_feature_columns(frame, whitelist=config.state_features))
    candidate_ids = tuple(
        sorted(cid for cid in frame["candidate_id"].unique() if cid != config.default_candidate_id)
    )
    selector = TrainedSelector(
        feature_columns=feature_columns,
        candidate_ids=candidate_ids,
        default_candidate_id=config.default_candidate_id,
    )

    for candidate_id in candidate_ids:
        rows = frame.loc[(frame["candidate_id"] == candidate_id) & (frame["split"] == "train")]
        target = pd.to_numeric(rows[TRAINING_TARGET_COLUMN], errors="coerce")
        usable = target.notna()
        rows = rows.loc[usable]
        target = target.loc[usable]
        if len(rows) < max(10, 2 * len(feature_columns)):
            continue
        features = rows[list(feature_columns)].to_numpy(dtype=np.float64)
        with warnings.catch_warnings():
            # All-NaN feature columns (e.g. liquidity proxy for cash) are
            # legitimate; they standardize to 0 below.
            warnings.simplefilter("ignore", RuntimeWarning)
            means = np.nanmean(features, axis=0)
            stds = np.nanstd(features, axis=0)
        means = np.nan_to_num(means, nan=0.0)
        stds = np.nan_to_num(stds, nan=0.0) + 1e-12
        standardized = _standardize(features, means, stds)
        model = Ridge(alpha=config.selector_ridge_alpha)
        model.fit(standardized, target.to_numpy(dtype=np.float64))
        residuals = target.to_numpy(dtype=np.float64) - model.predict(standardized)
        selector.models[candidate_id] = model
        selector.feature_means[candidate_id] = means
        selector.feature_stds[candidate_id] = stds
        selector.residual_std[candidate_id] = max(float(residuals.std(ddof=1)), 1e-6)
    if not selector.models:
        raise ValueError("Selector training produced no candidate models")
    return selector


def predict_candidate_deltas(
    selector: TrainedSelector,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Predicted delta-utility for every (date, candidate) row in the frame."""

    pieces: list[pd.DataFrame] = []
    for candidate_id, model in selector.models.items():
        rows = frame.loc[frame["candidate_id"] == candidate_id]
        if rows.empty:
            continue
        features = rows[list(selector.feature_columns)].to_numpy(dtype=np.float64)
        standardized = _standardize(
            features,
            selector.feature_means[candidate_id],
            selector.feature_stds[candidate_id],
        )
        predictions = model.predict(standardized)  # type: ignore[attr-defined]
        pieces.append(
            pd.DataFrame(
                {
                    "date": rows["date"].to_numpy(),
                    "candidate_id": candidate_id,
                    "split": rows["split"].to_numpy(),
                    "predicted_delta_utility": predictions,
                    "residual_std": selector.residual_std[candidate_id],
                }
            )
        )
    predictions_frame = pd.concat(pieces, ignore_index=True)
    predictions_frame = predictions_frame.sort_values(
        ["date", "candidate_id"], kind="mergesort"
    ).reset_index(drop=True)
    return predictions_frame[list(PREDICTION_COLUMNS)]


def route_decisions(
    predictions: pd.DataFrame,
    config: MetaRouterConfig,
) -> pd.DataFrame:
    """Date-level routing with abstain/default, threshold buffer, and inertia.

    The default action is explicit: when no candidate clears
    ``k * residual_std + cost_buffer``, the router abstains to the default
    blend. Switching away from the currently held candidate additionally
    requires a ``switch_margin`` predicted advantage (hold-current inertia).
    """

    rows: list[dict[str, object]] = []
    previous = config.default_candidate_id
    for date, day in predictions.groupby("date", sort=True):
        best_row = day.loc[day["predicted_delta_utility"].idxmax()]
        best_id = str(best_row["candidate_id"])
        best_delta = float(best_row["predicted_delta_utility"])
        buffer = (
            config.selector_uncertainty_k * float(best_row["residual_std"])
            + config.selector_cost_buffer
        )

        if best_delta <= buffer:
            chosen, action = config.default_candidate_id, "abstain_default"
        else:
            chosen, action = best_id, "route"
            if chosen != previous:
                held = day.loc[day["candidate_id"] == previous, "predicted_delta_utility"]
                held_delta = float(held.iloc[0]) if len(held) else 0.0
                if best_delta - held_delta <= config.switch_margin:
                    chosen, action = previous, "hold_inertia"

        rows.append(
            {
                "date": date,
                "split": str(day["split"].iloc[0]),
                "chosen_candidate_id": chosen,
                "action": action,
                "predicted_delta": best_delta,
                "decision_buffer": buffer,
                "previous_candidate_id": previous,
            }
        )
        previous = chosen

    decisions = pd.DataFrame(rows)
    return decisions[list(ROUTE_DECISION_COLUMNS)]


def softmax_trust_allocation(
    predictions: pd.DataFrame,
    config: MetaRouterConfig,
) -> pd.DataFrame:
    """Smoothed softmax trust weights over candidates (default included at 0).

    Returns a long frame (date, candidate_id, allocation); allocations sum to
    1 per date. Cash participates like any candidate, so the allocator can
    de-risk by shifting trust mass to cash.
    """

    wide = predictions.pivot_table(
        index="date", columns="candidate_id", values="predicted_delta_utility"
    ).sort_index()
    wide[config.default_candidate_id] = 0.0
    if CASH_CANDIDATE_ID not in wide.columns:
        wide[CASH_CANDIDATE_ID] = 0.0
    values = wide.to_numpy(dtype=np.float64)
    values = np.nan_to_num(values, nan=-np.inf)

    scaled = config.softmax_temperature * values
    scaled -= scaled.max(axis=1, keepdims=True)
    exponent = np.exp(scaled)
    softmax = exponent / exponent.sum(axis=1, keepdims=True)

    rho = config.inertia_rho if config.inertia_rho > 0 else 1.0
    smoothed = np.empty_like(softmax)
    smoothed[0] = softmax[0]
    for t in range(1, len(softmax)):
        smoothed[t] = (1.0 - rho) * smoothed[t - 1] + rho * softmax[t]
    smoothed /= smoothed.sum(axis=1, keepdims=True)

    allocation = pd.DataFrame(smoothed, index=wide.index, columns=wide.columns)
    long = allocation.reset_index().melt(
        id_vars="date", var_name="candidate_id", value_name="allocation"
    )
    return long.sort_values(["date", "candidate_id"], kind="mergesort").reset_index(drop=True)


def selector_coefficients(selector: TrainedSelector) -> dict[str, dict[str, float]]:
    """Per-candidate standardized ridge coefficients for diagnostics."""

    payload: dict[str, dict[str, float]] = {}
    for candidate_id, model in selector.models.items():
        coefficients = getattr(model, "coef_", None)
        if coefficients is None:
            continue
        payload[candidate_id] = {
            name: round(float(value), 8)
            for name, value in zip(selector.feature_columns, coefficients, strict=True)
        }
    return payload


__all__ = [
    "TrainedSelector",
    "predict_candidate_deltas",
    "route_decisions",
    "selector_coefficients",
    "softmax_trust_allocation",
    "train_local_policy_selector",
]

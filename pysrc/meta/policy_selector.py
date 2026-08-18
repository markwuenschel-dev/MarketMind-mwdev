"""Static, regime, and tree gating baselines for meta-router."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from pysrc.contracts.meta_router import (
    CASH_CANDIDATE_ID,
    DEFAULT_CANDIDATE_ID,
    META_ROUTER_DECISION_PANEL_COLUMNS,
    TRAINING_TARGET_COLUMN,
    MetaRouterConfig,
    select_feature_columns,
)
from pysrc.meta.abstention import abstain_probability_from_deltas
from pysrc.meta.exposure import exposure_scale_from_state


def _gate_feature_columns(
    frame: pd.DataFrame,
    config: MetaRouterConfig | None = None,
) -> list[str]:
    whitelist = config.state_features if config is not None else None
    features = select_feature_columns(frame, whitelist=whitelist)
    if config is not None and config.state_features and not features:
        raise ValueError(
            "gate: configured state_features not present in training frame "
            f"{list(config.state_features)}"
        )
    return features


def _weights_json(weights: dict[str, float]) -> str:
    return json.dumps({k: round(v, 6) for k, v in sorted(weights.items())})


def equal_weight_decisions(
    frame: pd.DataFrame,
    *,
    gate_id: str = "equal_weight_blend",
    default_candidate_id: str = DEFAULT_CANDIDATE_ID,
) -> pd.DataFrame:
    candidates = sorted(frame["candidate_id"].unique().tolist())
    active = [c for c in candidates if c not in {CASH_CANDIDATE_ID, default_candidate_id}]
    if not active:
        active = candidates
    weight = 1.0 / max(1, len(active))
    rows: list[dict[str, Any]] = []
    for date in sorted(frame["date"].unique()):
        split = str(frame.loc[frame["date"] == date, "split"].iloc[0])
        weights = dict.fromkeys(active, weight)
        rows.append(
            {
                "date": date,
                "fold_id": "all",
                "split": split,
                "gate_id": gate_id,
                "selected_candidate_id": max(weights, key=lambda k: weights[k]),
                "exposure_scale": 1.0,
                "abstain_probability": 0.0,
                "action": "route",
                "model_weights_json": _weights_json(weights),
            }
        )
    return pd.DataFrame(rows)


def validation_weighted_blend(
    frame: pd.DataFrame,
    *,
    gate_id: str = "validation_weighted_blend",
    default_candidate_id: str = DEFAULT_CANDIDATE_ID,
) -> pd.DataFrame:
    val = frame.loc[frame["split"].isin(["train", "embargo"])]
    util = val.groupby("candidate_id")[TRAINING_TARGET_COLUMN].mean().sort_values(ascending=False)
    candidates = [c for c in util.index.astype(str) if c != CASH_CANDIDATE_ID]
    if not candidates:
        return equal_weight_decisions(frame, gate_id=gate_id)
    exp_util = np.exp(util[candidates].to_numpy(dtype=np.float64))
    exp_util = exp_util - exp_util.max()
    probs = np.exp(exp_util)
    probs = probs / probs.sum()
    weights = {c: float(p) for c, p in zip(candidates, probs, strict=True)}
    rows: list[dict[str, Any]] = []
    for date in sorted(frame["date"].unique()):
        split = str(frame.loc[frame["date"] == date, "split"].iloc[0])
        rows.append(
            {
                "date": date,
                "fold_id": "all",
                "split": split,
                "gate_id": gate_id,
                "selected_candidate_id": max(weights, key=lambda k: weights[k]),
                "exposure_scale": 1.0,
                "abstain_probability": abstain_probability_from_deltas(
                    util[candidates].to_numpy(dtype=np.float64)
                ),
                "action": "route",
                "model_weights_json": _weights_json(weights),
            }
        )
    return pd.DataFrame(rows)


def regime_lookup_gate(
    frame: pd.DataFrame,
    regime_panel: pd.DataFrame,
    *,
    gate_id: str = "regime_lookup",
) -> pd.DataFrame:
    train = frame.loc[frame["split"] == "train"].copy()
    if "regime" not in regime_panel.columns:
        return equal_weight_decisions(frame, gate_id=gate_id)
    train_base = train.drop(columns=["regime"], errors="ignore")
    merged = train_base.merge(regime_panel[["date", "regime"]], on="date", how="left")
    if "regime" not in merged.columns:
        return equal_weight_decisions(frame, gate_id=gate_id)
    merged = merged.dropna(subset=["regime"])
    if merged.empty:
        return equal_weight_decisions(frame, gate_id=gate_id)
    lookup = merged.groupby(["regime", "candidate_id"])[TRAINING_TARGET_COLUMN].mean().reset_index()
    best = lookup.sort_values(TRAINING_TARGET_COLUMN, ascending=False).groupby("regime").first()
    rows: list[dict[str, Any]] = []
    for date in sorted(frame["date"].unique()):
        split = str(frame.loc[frame["date"] == date, "split"].iloc[0])
        regime = regime_panel.loc[regime_panel["date"] == date, "regime"]
        regime_val = str(regime.iloc[0]) if len(regime) else "unknown"
        if regime_val in best.index:
            chosen = str(best.loc[regime_val, "candidate_id"])
            weights = {chosen: 1.0}
        else:
            chosen = DEFAULT_CANDIDATE_ID
            weights = {chosen: 1.0}
        rows.append(
            {
                "date": date,
                "fold_id": "all",
                "split": split,
                "gate_id": gate_id,
                "selected_candidate_id": chosen,
                "exposure_scale": 1.0,
                "abstain_probability": 0.0,
                "action": "route",
                "model_weights_json": _weights_json(weights),
            }
        )
    return pd.DataFrame(rows)


def tree_gate_decisions(
    frame: pd.DataFrame,
    *,
    gate_id: str = "tree_gate",
    random_seed: int = 42,
    config: MetaRouterConfig | None = None,
) -> pd.DataFrame:
    from sklearn.ensemble import ExtraTreesClassifier

    features = _gate_feature_columns(frame, config)
    if not features:
        return equal_weight_decisions(frame, gate_id=gate_id)
    train = frame.loc[frame["split"] == "train"].copy()
    test = frame.loc[frame["split"] == "test"].copy()
    if train.empty or test.empty:
        return equal_weight_decisions(frame, gate_id=gate_id)

    # Date-level classification: best candidate per date on train
    best_train = (
        train.sort_values(TRAINING_TARGET_COLUMN, ascending=False)
        .groupby("date")
        .first()
        .reset_index()
    )
    date_features = (
        train.groupby("date")[features]
        .first()
        .reset_index()
        .merge(best_train[["date", "candidate_id"]], on="date")
    )
    x_train = date_features[features].to_numpy(dtype=np.float64)
    y_train = date_features["candidate_id"].astype(str).to_numpy()
    clf = ExtraTreesClassifier(n_estimators=100, random_state=random_seed)
    clf.fit(x_train, y_train)

    sorted(test["date"].unique())
    date_feats = test.groupby("date")[features].first().reset_index()
    preds = clf.predict(date_feats[features].to_numpy(dtype=np.float64))
    rows: list[dict[str, Any]] = []
    for date, chosen in zip(date_feats["date"].astype(str), preds, strict=True):
        split = str(test.loc[test["date"] == date, "split"].iloc[0])
        weights = {str(chosen): 1.0}
        rows.append(
            {
                "date": date,
                "fold_id": "all",
                "split": split,
                "gate_id": gate_id,
                "selected_candidate_id": str(chosen),
                "exposure_scale": exposure_scale_from_state(
                    volatility_state=None, model_disagreement=0.0
                ),
                "abstain_probability": 0.0,
                "action": "route",
                "model_weights_json": _weights_json(weights),
            }
        )
    return pd.DataFrame(rows)


def recent_winner_selector_decisions(
    frame: pd.DataFrame,
    *,
    gate_id: str = "recent_winner_selector",
    window_col: str = "cand_recent_utility_20",
    default_candidate_id: str = DEFAULT_CANDIDATE_ID,
) -> pd.DataFrame:
    """Route to the candidate with best lagged recent utility (train-safe feature)."""

    if window_col not in frame.columns:
        return equal_weight_decisions(frame, gate_id=gate_id)

    calendar = sorted(frame["date"].astype(str).unique())
    split_map = frame[["date", "split"]].drop_duplicates().set_index("date")["split"].to_dict()
    candidates = sorted(c for c in frame["candidate_id"].unique() if str(c) != CASH_CANDIDATE_ID)
    recent = frame.pivot_table(index="date", columns="candidate_id", values=window_col).reindex(
        calendar
    )
    choice = (
        recent[candidates].idxmax(axis=1).fillna(default_candidate_id)
        if candidates
        else pd.Series(default_candidate_id, index=calendar)
    )
    rows: list[dict[str, Any]] = []
    for date in calendar:
        chosen = str(choice.get(date, default_candidate_id))
        rows.append(
            {
                "date": date,
                "fold_id": "all",
                "split": split_map.get(date, "test"),
                "gate_id": gate_id,
                "selected_candidate_id": chosen,
                "exposure_scale": 1.0,
                "abstain_probability": 0.0,
                "action": "route",
                "model_weights_json": _weights_json({chosen: 1.0}),
            }
        )
    return pd.DataFrame(rows)


def boosted_tree_gate_decisions(
    frame: pd.DataFrame,
    *,
    gate_id: str = "boosted_tree_gate",
    random_seed: int = 42,
    config: MetaRouterConfig | None = None,
) -> pd.DataFrame:
    from sklearn.ensemble import GradientBoostingClassifier

    features = _gate_feature_columns(frame, config)
    if not features:
        return equal_weight_decisions(frame, gate_id=gate_id)
    train = frame.loc[frame["split"] == "train"].copy()
    test = frame.loc[frame["split"] == "test"].copy()
    if train.empty or test.empty:
        return equal_weight_decisions(frame, gate_id=gate_id)

    best_train = (
        train.sort_values(TRAINING_TARGET_COLUMN, ascending=False)
        .groupby("date")
        .first()
        .reset_index()
    )
    date_features = train.groupby("date")[features].first().reset_index()
    merged = date_features.merge(best_train[["date", "candidate_id"]], on="date")
    clf = GradientBoostingClassifier(random_state=random_seed)
    clf.fit(
        merged[features].to_numpy(dtype=np.float64),
        merged["candidate_id"].astype(str).to_numpy(),
    )
    test_feats = test.groupby("date")[features].first().reset_index()
    preds = clf.predict(test_feats[features].to_numpy(dtype=np.float64))
    rows: list[dict[str, Any]] = []
    for date, chosen in zip(test_feats["date"].astype(str), preds, strict=True):
        split = str(test.loc[test["date"] == date, "split"].iloc[0])
        weights = {str(chosen): 1.0}
        rows.append(
            {
                "date": date,
                "fold_id": "all",
                "split": split,
                "gate_id": gate_id,
                "selected_candidate_id": str(chosen),
                "exposure_scale": 1.0,
                "abstain_probability": 0.0,
                "action": "route",
                "model_weights_json": _weights_json(weights),
            }
        )
    return pd.DataFrame(rows)


def oracle_diagnostic_decisions(frame: pd.DataFrame) -> pd.DataFrame:
    """Ex-post best candidate — evaluation only, leakage flagged."""

    rows: list[dict[str, Any]] = []
    test = frame.loc[frame["split"] == "test"]
    for date in sorted(test["date"].unique()):
        day = test.loc[test["date"] == date]
        best = day.sort_values(TRAINING_TARGET_COLUMN, ascending=False).iloc[0]
        chosen = str(best["candidate_id"])
        rows.append(
            {
                "date": date,
                "fold_id": "all",
                "split": "test",
                "gate_id": "oracle_diagnostic",
                "selected_candidate_id": chosen,
                "exposure_scale": 1.0,
                "abstain_probability": 0.0,
                "action": "route_leakage_flagged",
                "model_weights_json": _weights_json({chosen: 1.0}),
            }
        )
    return pd.DataFrame(rows)


def normalize_decision_panel(panel: pd.DataFrame) -> pd.DataFrame:
    for col in META_ROUTER_DECISION_PANEL_COLUMNS:
        if col not in panel.columns:
            panel[col] = None
    return panel[list(META_ROUTER_DECISION_PANEL_COLUMNS)]


__all__ = [
    "boosted_tree_gate_decisions",
    "equal_weight_decisions",
    "normalize_decision_panel",
    "oracle_diagnostic_decisions",
    "recent_winner_selector_decisions",
    "regime_lookup_gate",
    "tree_gate_decisions",
    "validation_weighted_blend",
]

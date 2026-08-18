"""Small neural gating network: state features → model weights + abstain logit."""

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
)
from pysrc.meta.exposure import exposure_scale_from_state
from pysrc.meta.policy_selector import _gate_feature_columns


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(np.clip(x, -20, 20))
    out: np.ndarray = e / e.sum()
    return out


class NeuralGatingNetwork:
    """Two-layer numpy MLP gate (no torch dependency for milestone smoke)."""

    def __init__(self, n_features: int, n_candidates: int, *, random_seed: int = 42) -> None:
        rng = np.random.default_rng(random_seed)
        self.w1 = rng.normal(0, 0.1, size=(n_features, 32))
        self.b1 = np.zeros(32)
        self.w2 = rng.normal(0, 0.1, size=(32, n_candidates))
        self.b2 = np.zeros(n_candidates)
        self.w_abstain = rng.normal(0, 0.1, size=(32, 1))
        self.candidate_ids: list[str] = []

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        h = np.tanh(x @ self.w1 + self.b1)
        logits = h @ self.w2 + self.b2
        abstain_logit = float((h @ self.w_abstain).squeeze())
        return logits, abstain_logit

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        candidate_ids: list[str],
        *,
        epochs: int = 50,
        lr: float = 0.01,
    ) -> None:
        self.candidate_ids = list(candidate_ids)
        n_candidates = len(candidate_ids)
        self.w2 = np.zeros((32, n_candidates))
        # Simple one-hot target from best utility candidate index
        for _epoch in range(epochs):
            for row, target_idx in zip(x_train, y_train, strict=False):
                logits, _abstain_logit = self._forward(row)
                probs = _softmax(logits)
                grad = probs.copy()
                grad[int(target_idx)] -= 1.0
                h = np.tanh(row @ self.w1 + self.b1)
                self.w2 -= lr * np.outer(h, grad)
                self.b2 -= lr * grad

    def predict_weights(self, x: np.ndarray) -> tuple[dict[str, float], float, float]:
        logits, abstain_logit = self._forward(x)
        probs = _softmax(logits)
        ids = self.candidate_ids or [f"candidate_{idx}" for idx in range(len(probs))]
        if len(ids) != len(probs):
            ids = (
                ids[: len(probs)]
                if len(ids) > len(probs)
                else ids + [f"candidate_{idx}" for idx in range(len(ids), len(probs))]
            )
        weights = {cid: float(p) for cid, p in zip(ids, probs, strict=True)}
        abstain_p = float(1.0 / (1.0 + np.exp(-abstain_logit)))
        exposure = exposure_scale_from_state(
            volatility_state=None,
            model_disagreement=float(np.std(list(weights.values()))),
        )
        return weights, exposure, abstain_p


def neural_gate_decisions(
    frame: pd.DataFrame,
    *,
    gate_id: str = "neural_gate",
    random_seed: int = 42,
    config: MetaRouterConfig | None = None,
) -> pd.DataFrame:
    features = _gate_feature_columns(frame, config)
    if not features:
        from pysrc.meta.policy_selector import equal_weight_decisions

        return equal_weight_decisions(frame, gate_id=gate_id)

    train = frame.loc[frame["split"] == "train"]
    test = frame.loc[frame["split"] == "test"]
    candidates = sorted(c for c in train["candidate_id"].unique() if str(c) != CASH_CANDIDATE_ID)
    if not candidates or train.empty or test.empty:
        from pysrc.meta.policy_selector import equal_weight_decisions

        return equal_weight_decisions(frame, gate_id=gate_id)

    def _candidate_index(candidate_id: str) -> int:
        if candidate_id in candidates:
            return candidates.index(candidate_id)
        if DEFAULT_CANDIDATE_ID in candidates:
            return candidates.index(DEFAULT_CANDIDATE_ID)
        return 0

    # Date-level best candidate on train utilities
    date_train = (
        train.sort_values(TRAINING_TARGET_COLUMN, ascending=False)
        .groupby("date")
        .first()
        .reset_index()
    )
    date_feats = train.groupby("date")[features].first().reset_index()
    merged = date_feats.merge(
        date_train[["date", "candidate_id"]],
        on="date",
    )
    x_train = merged[features].to_numpy(dtype=np.float64)
    y_train = np.asarray([_candidate_index(c) for c in merged["candidate_id"].astype(str)])

    gate = NeuralGatingNetwork(x_train.shape[1], len(candidates), random_seed=random_seed)
    gate.fit(x_train, y_train, candidates)

    rows: list[dict[str, Any]] = []
    test_dates = sorted(test["date"].unique())
    date_test_feats = test.groupby("date")[features].first().reset_index()
    for row, date in zip(
        date_test_feats[features].to_numpy(dtype=np.float64), test_dates, strict=False
    ):
        weights, exposure, abstain_p = gate.predict_weights(row)
        chosen = max(weights, key=lambda k: weights[k])
        split = str(test.loc[test["date"] == date, "split"].iloc[0])
        rows.append(
            {
                "date": date,
                "fold_id": "all",
                "split": split,
                "gate_id": gate_id,
                "selected_candidate_id": chosen,
                "exposure_scale": exposure,
                "abstain_probability": abstain_p,
                "action": "abstain_default" if abstain_p > 0.5 else "route",
                "model_weights_json": json.dumps(weights),
            }
        )
    out = pd.DataFrame(rows)
    for col in META_ROUTER_DECISION_PANEL_COLUMNS:
        if col not in out.columns:
            out[col] = None
    return out[list(META_ROUTER_DECISION_PANEL_COLUMNS)]


__all__ = ["NeuralGatingNetwork", "neural_gate_decisions"]

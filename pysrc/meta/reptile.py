"""Fold-safe Reptile meta-learning for neural gating policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pysrc.contracts.meta_router import META_ROUTER_DECISION_PANEL_COLUMNS, MetaRouterConfig
from pysrc.meta.gating_network import NeuralGatingNetwork, neural_gate_decisions
from pysrc.meta.policy_selector import _gate_feature_columns


@dataclass
class ReptileConfig:
    inner_steps: int = 5
    inner_lr: float = 0.05
    outer_lr: float = 0.1
    meta_epochs: int = 3
    random_seed: int = 42


def _clone_gate(gate: NeuralGatingNetwork) -> NeuralGatingNetwork:
    cloned = NeuralGatingNetwork(gate.w1.shape[0], gate.w2.shape[1], random_seed=0)
    cloned.w1 = gate.w1.copy()
    cloned.b1 = gate.b1.copy()
    cloned.w2 = gate.w2.copy()
    cloned.b2 = gate.b2.copy()
    cloned.w_abstain = gate.w_abstain.copy()
    cloned.candidate_ids = list(gate.candidate_ids)
    return cloned


def reptile_meta_train_gate(
    tasks: list[tuple[np.ndarray, np.ndarray, list[str]]],
    n_features: int,
    n_candidates: int,
    config: ReptileConfig,
) -> NeuralGatingNetwork:
    """Outer Reptile loop over regime tasks."""

    meta = NeuralGatingNetwork(n_features, n_candidates, random_seed=config.random_seed)
    for _ in range(config.meta_epochs):
        for x_task, y_task, cids in tasks:
            inner = _clone_gate(meta)
            inner.fit(
                x_task,
                y_task,
                cids,
                epochs=config.inner_steps,
                lr=config.inner_lr,
            )
            meta.w1 += config.outer_lr * (inner.w1 - meta.w1)
            meta.b1 += config.outer_lr * (inner.b1 - meta.b1)
            meta.w2 += config.outer_lr * (inner.w2 - meta.w2)
            meta.b2 += config.outer_lr * (inner.b2 - meta.b2)
            meta.w_abstain += config.outer_lr * (inner.w_abstain - meta.w_abstain)
    if tasks:
        meta.candidate_ids = list(tasks[-1][2])
    return meta


def build_regime_tasks(
    frame: pd.DataFrame,
    regime_panel: pd.DataFrame,
    features: list[str],
    candidates: list[str],
) -> list[tuple[np.ndarray, np.ndarray, list[str]]]:
    from pysrc.contracts.meta_router import TRAINING_TARGET_COLUMN

    train = (
        frame.loc[frame["split"] == "train"]
        .drop(columns=["regime"], errors="ignore")
        .merge(regime_panel[["date", "regime"]], on="date", how="left")
    )
    if "regime" not in train.columns:
        return []
    tasks: list[tuple[np.ndarray, np.ndarray, list[str]]] = []
    for regime, group in train.groupby("regime"):
        if str(regime) == "nan":
            continue
        best = (
            group.sort_values(TRAINING_TARGET_COLUMN, ascending=False)
            .groupby("date")
            .first()
            .reset_index()
        )
        feats = group.groupby("date")[features].first().reset_index()
        merged = feats.merge(best[["date", "candidate_id"]], on="date")
        if merged.empty:
            continue
        x = merged[features].to_numpy(dtype=np.float64)
        y = np.asarray(
            [
                candidates.index(c) if c in candidates else 0
                for c in merged["candidate_id"].astype(str)
            ]
        )
        tasks.append((x, y, candidates))
    return tasks


def reptile_neural_gate_decisions(
    frame: pd.DataFrame,
    regime_panel: pd.DataFrame,
    *,
    gate_id: str = "reptile_neural_gate",
    config: ReptileConfig | None = None,
    router_config: MetaRouterConfig | None = None,
) -> pd.DataFrame:
    cfg = config or ReptileConfig()
    features = _gate_feature_columns(frame, router_config)
    if not features:
        return neural_gate_decisions(
            frame, gate_id=gate_id, random_seed=cfg.random_seed, config=router_config
        )

    train = frame.loc[frame["split"] == "train"]
    test = frame.loc[frame["split"] == "test"]
    candidates = sorted(c for c in train["candidate_id"].unique())
    tasks = build_regime_tasks(frame, regime_panel, features, candidates)
    if not tasks:
        return neural_gate_decisions(
            frame, gate_id=gate_id, random_seed=cfg.random_seed, config=router_config
        )

    gate = reptile_meta_train_gate(tasks, len(features), len(candidates), cfg)
    rows: list[dict[str, Any]] = []
    date_feats = test.groupby("date")[features].first().reset_index()
    for row, date in zip(
        date_feats[features].to_numpy(dtype=np.float64), date_feats["date"], strict=False
    ):
        weights, exposure, abstain_p = gate.predict_weights(row)
        chosen = max(weights, key=lambda k: weights[k])
        split = str(test.loc[test["date"] == date, "split"].iloc[0])
        import json

        rows.append(
            {
                "date": date,
                "fold_id": "all",
                "split": split,
                "gate_id": gate_id,
                "selected_candidate_id": chosen,
                "exposure_scale": exposure,
                "abstain_probability": abstain_p,
                "action": "route",
                "model_weights_json": json.dumps(weights),
            }
        )
    out = pd.DataFrame(rows)
    for col in META_ROUTER_DECISION_PANEL_COLUMNS:
        if col not in out.columns:
            out[col] = None
    return out[list(META_ROUTER_DECISION_PANEL_COLUMNS)]


__all__ = [
    "ReptileConfig",
    "build_regime_tasks",
    "reptile_meta_train_gate",
    "reptile_neural_gate_decisions",
]

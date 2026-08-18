"""Constrained decision objects for the local meta-router."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from pysrc.contracts.meta_router import CASH_CANDIDATE_ID


def normalize_candidate_weights(
    weights: dict[str, float],
    *,
    max_gross: float = 1.0,
    allow_short: bool = False,
) -> dict[str, float]:
    """Non-negative candidate weights summing to at most max_gross; cash gets residual."""

    cleaned: dict[str, float] = {}
    for cid, value in weights.items():
        if cid == CASH_CANDIDATE_ID:
            continue
        w = float(value)
        if not allow_short:
            w = max(0.0, w)
        cleaned[cid] = w
    total = sum(cleaned.values())
    if total > max_gross and total > 0:
        scale = max_gross / total
        cleaned = {k: v * scale for k, v in cleaned.items()}
    return cleaned


def apply_exposure_scale(
    weights: dict[str, float],
    exposure_scale: float,
    *,
    max_exposure: float = 1.0,
) -> tuple[dict[str, float], float]:
    """Scale weights and return cash residual weight."""

    scale = float(np.clip(exposure_scale, 0.0, max_exposure))
    scaled = {k: v * scale for k, v in weights.items()}
    gross = sum(scaled.values())
    cash_weight = max(0.0, 1.0 - gross)
    return scaled, cash_weight


def constrained_decision_from_weights(
    weights: dict[str, float],
    *,
    exposure_scale: float = 1.0,
    abstain_probability: float = 0.0,
    abstain_threshold: float = 0.5,
) -> dict[str, Any]:
    """Build a constrained decision dict with optional hard selection."""

    normalized = normalize_candidate_weights(weights)
    scaled, cash_weight = apply_exposure_scale(normalized, exposure_scale)
    if abstain_probability >= abstain_threshold:
        scaled = {}
        cash_weight = 1.0
        selected = CASH_CANDIDATE_ID
        action = "abstain_default"
    elif scaled:
        selected = max(scaled, key=lambda k: scaled[k])
        action = "route"
    else:
        selected = CASH_CANDIDATE_ID
        action = "abstain_default"
    return {
        "model_weights": scaled,
        "model_weights_json": json.dumps(scaled),
        "cash_weight": cash_weight,
        "exposure_scale": exposure_scale,
        "abstain_probability": abstain_probability,
        "selected_candidate_id": selected,
        "action": action,
        "decision_confidence": float(1.0 - abstain_probability),
    }


__all__ = [
    "apply_exposure_scale",
    "constrained_decision_from_weights",
    "normalize_candidate_weights",
]

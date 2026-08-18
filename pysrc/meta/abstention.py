"""Abstention and cash routing helpers."""

from __future__ import annotations

import numpy as np

from pysrc.contracts.meta_router import CASH_CANDIDATE_ID, DEFAULT_CANDIDATE_ID


def abstain_probability_from_deltas(
    deltas: np.ndarray,
    *,
    temperature: float = 1.0,
    cash_hurdle: float = 0.0,
) -> float:
    """Map candidate utility deltas to P(cash): high when all deltas are weak."""

    if deltas.size == 0:
        return 1.0
    max_delta = float(np.max(deltas))
    if max_delta <= cash_hurdle:
        return 1.0
    logits = np.exp(np.clip(deltas / max(temperature, 1e-9), -20, 20))
    float(np.max(logits))
    total = float(np.sum(logits) + 1.0)
    return float(1.0 / total)


def should_abstain(abstain_probability: float, *, threshold: float = 0.5) -> bool:
    return abstain_probability >= threshold


def abstain_candidate_id() -> str:
    return CASH_CANDIDATE_ID


def default_candidate_id() -> str:
    return DEFAULT_CANDIDATE_ID


__all__ = [
    "abstain_candidate_id",
    "abstain_probability_from_deltas",
    "default_candidate_id",
    "should_abstain",
]

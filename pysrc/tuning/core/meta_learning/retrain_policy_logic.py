"""Pure logic for evaluating whether a drift event should trigger retraining."""

from __future__ import annotations

from datetime import datetime


def should_retrain(
    drift_score: float,
    threshold: float,
    last_retrain_at: datetime,
    cooldown_seconds: int,
    now: datetime,
) -> bool:
    """Return True iff drift exceeds threshold and cooldown has elapsed."""
    if drift_score < threshold:
        return False
    elapsed = (now - last_retrain_at).total_seconds()
    return elapsed >= cooldown_seconds


__all__ = ["should_retrain"]

"""LiveTriggerSpec: validated, frozen spec for drift-triggered retraining."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LiveTriggerSpec:
    """Validated, immutable live trigger spec derived from LiveTriggerConfig."""

    name: str
    version: str
    spec_hash: str
    drift_threshold: float
    retrain_cooldown_seconds: int
    latency_budget_ms: int
    min_samples_before_trigger: int
    enable_online_features: bool


__all__ = ["LiveTriggerSpec"]

"""LiveTriggerConfig: declarative spec for drift-triggered retraining."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["LiveTriggerConfig"]


class LiveTriggerConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    version: str = "1.0.0"
    drift_threshold: float = Field(default=0.05, gt=0.0)
    retrain_cooldown_seconds: int = Field(default=86400, ge=0)
    latency_budget_ms: int = Field(default=200, ge=1)
    min_samples_before_trigger: int = Field(default=500, ge=1)
    enable_online_features: bool = False

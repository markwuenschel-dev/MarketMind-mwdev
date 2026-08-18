"""Neutral, validated handoffs between the active product-flow stages.

These models deliberately describe product semantics only.  Registry roles,
hashes, storage locations, and run lifecycle belong to ``artifact_registry``.
"""

from __future__ import annotations

from math import isfinite

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pysrc.contracts.trade_intent import TradeIntent


class PredictionValue(BaseModel):
    """One point-in-time prediction produced by a model fold."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    instrument: str = Field(min_length=1)
    decision_time: str = Field(min_length=1)
    value: float
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_finite_values(self) -> PredictionValue:
        if not isfinite(self.value):
            raise ValueError("prediction value must be finite")
        if self.confidence is not None and not isfinite(self.confidence):
            raise ValueError("prediction confidence must be finite")
        return self


class StandardizedPredictionArtifact(BaseModel):
    """Panel/tuning output consumed by a strategy implementation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(min_length=1)
    as_of: str = Field(min_length=1)
    data_lineage: dict[str, str] = Field(min_length=1)
    model_id: str = Field(min_length=1)
    fold_id: str = Field(min_length=1)
    split: str = Field(min_length=1)
    predictions: tuple[PredictionValue, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_lineage(self) -> StandardizedPredictionArtifact:
        if not all(key and value for key, value in self.data_lineage.items()):
            raise ValueError("data lineage entries must have non-empty keys and values")
        if any(item.decision_time > self.as_of for item in self.predictions):
            raise ValueError("predictions cannot be decided after artifact as_of")
        return self


class StandardizedTradeIntentArtifact(BaseModel):
    """Strategy output consumed by candidate-portfolio construction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = Field(min_length=1)
    strategy_id: str = Field(min_length=1)
    strategy_version: str = Field(min_length=1)
    decision_time: str = Field(min_length=1)
    prediction_lineage: dict[str, str] = Field(min_length=1)
    intents: tuple[TradeIntent, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_intent_lineage(self) -> StandardizedTradeIntentArtifact:
        if not all(key and value for key, value in self.prediction_lineage.items()):
            raise ValueError("prediction lineage entries must have non-empty keys and values")
        if any(intent.strategy_id != self.strategy_id for intent in self.intents):
            raise ValueError("every intent must belong to the artifact strategy")
        if any(intent.date > self.decision_time for intent in self.intents):
            raise ValueError("intents cannot be decided after artifact decision_time")
        return self


__all__ = [
    "PredictionValue",
    "StandardizedPredictionArtifact",
    "StandardizedTradeIntentArtifact",
]

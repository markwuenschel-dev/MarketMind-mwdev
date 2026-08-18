"""Pre-sizing strategy intent exchanged with candidate portfolio construction."""

from __future__ import annotations

from enum import StrEnum
from math import isfinite

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TradeDirection(StrEnum):
    """Directional view expressed by a strategy before portfolio sizing."""

    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


class TradeIntent(BaseModel):
    """A strategy's typed, pre-sizing opinion for one instrument and date.

    This contract deliberately contains no target weight, exposure, order, fill,
    cost, or risk-mechanics field.  Candidate portfolio construction owns sizing;
    MetaRouter owns candidate selection; backtesting owns execution economics.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    date: str = Field(min_length=1)
    instrument: str = Field(min_length=1)
    interval: str = Field(min_length=1)
    strategy_id: str = Field(min_length=1)
    intent_id: str = Field(min_length=1)
    score: float | None = None
    direction: TradeDirection
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    eligible: bool
    abstain: bool
    source_model_id: str | None = None
    source_product_id: str = Field(min_length=1)
    lineage: dict[str, str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_decision_state(self) -> TradeIntent:
        if not self.eligible:
            if self.abstain:
                raise ValueError("ineligible intents cannot also be abstaining")
            if self.direction is not TradeDirection.FLAT:
                raise ValueError("ineligible intents must be FLAT")
            if self.score is not None or self.confidence is not None:
                raise ValueError("ineligible intents cannot carry score or confidence")
            return self

        if self.abstain:
            if self.direction is not TradeDirection.FLAT:
                raise ValueError("abstaining intents must be FLAT")
            if self.score is not None or self.confidence is not None:
                raise ValueError("abstaining intents cannot carry score or confidence")
            return self

        if self.score is None or not isfinite(self.score):
            raise ValueError("active intents require a finite score")
        if self.confidence is not None and not isfinite(self.confidence):
            raise ValueError("confidence must be finite")

        if self.direction is TradeDirection.LONG and self.score <= 0.0:
            raise ValueError("LONG intents require a positive score")
        if self.direction is TradeDirection.SHORT and self.score >= 0.0:
            raise ValueError("SHORT intents require a negative score")
        if self.direction is TradeDirection.FLAT and self.score != 0.0:
            raise ValueError("FLAT intents require a zero score")

        if not all(key and value for key, value in self.lineage.items()):
            raise ValueError("lineage entries must have non-empty keys and values")
        return self


__all__ = ["TradeDirection", "TradeIntent"]

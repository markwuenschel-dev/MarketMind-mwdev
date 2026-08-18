"""Deterministic threshold strategy for standardized prediction panels."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import cast

import pandas as pd

from pysrc.contracts.trade_intent import TradeDirection, TradeIntent

_REQUIRED_COLUMNS = frozenset({"date", "instrument", "prediction"})


def build_threshold_intents(
    predictions: pd.DataFrame,
    *,
    strategy_id: str,
    source_product_id: str,
    lineage: Mapping[str, str],
    threshold: float = 0.0,
    interval: str = "1d",
    source_model_id: str | None = None,
) -> pd.DataFrame:
    """Express each finite prediction as a neutral or directional pre-sizing intent."""

    if not strategy_id or not source_product_id or not interval:
        raise ValueError("strategy_id, source_product_id, and interval must be non-empty")
    if threshold < 0.0 or not isfinite(threshold):
        raise ValueError("threshold must be a finite non-negative number")
    if not lineage or not all(key and value for key, value in lineage.items()):
        raise ValueError("lineage must contain non-empty keys and values")

    missing = sorted(_REQUIRED_COLUMNS - set(predictions.columns))
    if missing:
        raise ValueError(f"prediction panel missing columns: {missing}")

    records: list[dict[str, object]] = []
    for row in predictions.loc[:, ["date", "instrument", "prediction"]].itertuples(index=False):
        pred_scalar = row.prediction
        raw_score = float(cast(float, pred_scalar)) if pd.notna(pred_scalar) else None
        eligible = raw_score is not None and isfinite(raw_score)
        if not eligible:
            direction = TradeDirection.FLAT
            score = None
        elif raw_score > threshold:
            direction = TradeDirection.LONG
            score = raw_score
        elif raw_score < -threshold:
            direction = TradeDirection.SHORT
            score = raw_score
        else:
            direction = TradeDirection.FLAT
            score = 0.0

        intent = TradeIntent(
            date=str(row.date),
            instrument=str(row.instrument),
            interval=interval,
            strategy_id=strategy_id,
            intent_id=f"{strategy_id}:{row.instrument}:{row.date}",
            score=score,
            direction=direction,
            eligible=eligible,
            abstain=False,
            source_model_id=source_model_id,
            source_product_id=source_product_id,
            lineage=dict(lineage),
        )
        records.append(intent.model_dump(mode="json"))

    return pd.DataFrame.from_records(records)


__all__ = ["build_threshold_intents"]

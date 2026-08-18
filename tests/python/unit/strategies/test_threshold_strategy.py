"""Tests for the canonical reusable strategy decision layer."""

from __future__ import annotations

import pandas as pd
import pytest

from pysrc.contracts.trade_intent import TradeDirection
from pysrc.strategies import build_threshold_intents


@pytest.mark.determinism("d1")
def test_threshold_strategy_converts_predictions_to_pre_sizing_intents(
    deterministic_seed: int,
) -> None:
    _ = deterministic_seed
    predictions = pd.DataFrame(
        [
            {"date": "2026-01-05", "instrument": "SPY", "prediction": 0.6},
            {"date": "2026-01-05", "instrument": "TLT", "prediction": -0.4},
            {"date": "2026-01-05", "instrument": "GLD", "prediction": 0.05},
            {"date": "2026-01-05", "instrument": "IWM", "prediction": None},
        ]
    )

    intents = build_threshold_intents(
        predictions,
        strategy_id="prediction_threshold",
        source_product_id="prediction-product",
        lineage={"run_id": "run-1"},
        threshold=0.1,
    )

    records = intents[["instrument", "direction", "score", "eligible", "abstain"]].to_dict(
        "records"
    )
    assert records[:3] == [
        {
            "instrument": "SPY",
            "direction": TradeDirection.LONG.value,
            "score": 0.6,
            "eligible": True,
            "abstain": False,
        },
        {
            "instrument": "TLT",
            "direction": TradeDirection.SHORT.value,
            "score": -0.4,
            "eligible": True,
            "abstain": False,
        },
        {
            "instrument": "GLD",
            "direction": TradeDirection.FLAT.value,
            "score": 0.0,
            "eligible": True,
            "abstain": False,
        },
    ]
    assert records[3]["instrument"] == "IWM"
    assert records[3]["direction"] == TradeDirection.FLAT.value
    assert pd.isna(records[3]["score"])
    assert records[3]["eligible"] is False
    assert records[3]["abstain"] is False


@pytest.mark.determinism("d1")
def test_threshold_strategy_requires_prediction_identity_columns(
    deterministic_seed: int,
) -> None:
    _ = deterministic_seed

    with pytest.raises(ValueError, match="missing columns"):
        build_threshold_intents(
            pd.DataFrame([{"date": "2026-01-05", "prediction": 0.6}]),
            strategy_id="prediction_threshold",
            source_product_id="prediction-product",
            lineage={"run_id": "run-1"},
        )

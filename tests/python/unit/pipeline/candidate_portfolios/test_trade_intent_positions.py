"""Candidate sizing tests for canonical strategy intent products."""

from __future__ import annotations

import pandas as pd
import pytest

from pysrc.contracts.trade_intent import TradeDirection
from pysrc.pipeline.candidate_portfolios.trade_intent_positions import (
    trade_intents_to_candidate_positions,
)


@pytest.mark.determinism("d1")
def test_trade_intents_become_signed_candidate_positions(deterministic_seed: int) -> None:
    _ = deterministic_seed
    intents = pd.DataFrame(
        [
            {
                "date": "2026-01-05",
                "instrument": "SPY",
                "strategy_id": "cross_sectional",
                "score": 0.9,
                "direction": TradeDirection.LONG.value,
                "eligible": True,
                "abstain": False,
            },
            {
                "date": "2026-01-05",
                "instrument": "XLF",
                "strategy_id": "cross_sectional",
                "score": -0.8,
                "direction": TradeDirection.SHORT.value,
                "eligible": True,
                "abstain": False,
            },
            {
                "date": "2026-01-05",
                "instrument": "IWM",
                "strategy_id": "cross_sectional",
                "score": 0.7,
                "direction": TradeDirection.LONG.value,
                "eligible": True,
                "abstain": False,
            },
            {
                "date": "2026-01-05",
                "instrument": "GLD",
                "strategy_id": "cross_sectional",
                "score": 0.0,
                "direction": TradeDirection.FLAT.value,
                "eligible": True,
                "abstain": False,
            },
            {
                "date": "2026-01-05",
                "instrument": "TLT",
                "strategy_id": "cross_sectional",
                "score": None,
                "direction": TradeDirection.FLAT.value,
                "eligible": True,
                "abstain": True,
            },
        ]
    )

    positions = trade_intents_to_candidate_positions(
        intents,
        top_k=2,
        single_name_cap=0.6,
        fold_id="fold-1",
        split="test",
    )

    strategy_positions = positions.loc[positions["candidate_id"] == "cross_sectional"]
    assert strategy_positions[["ticker", "target_weight"]].to_dict("records") == [
        {"ticker": "SPY", "target_weight": 0.5},
        {"ticker": "XLF", "target_weight": -0.5},
    ]
    assert set(strategy_positions["fold_id"]) == {"fold-1"}
    assert set(strategy_positions["split"]) == {"test"}

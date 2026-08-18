"""Candidate construction preserves stage-owned fold and split envelopes."""

from __future__ import annotations

import pandas as pd
import pytest

from pysrc.contracts.meta_router import DEFAULT_CANDIDATE_ID
from pysrc.pipeline.candidate_portfolios.strategy_products import (
    trade_intent_envelope_to_candidate_positions,
)
from pysrc.pipeline.p2_config_loader import PortfolioSpec


@pytest.mark.determinism("d1")
def test_trade_intent_envelope_keeps_fold_and_split(
    deterministic_seed: int,
) -> None:
    _ = deterministic_seed
    intents = pd.DataFrame(
        [
            {
                "date": "2026-01-05",
                "instrument": "SPY",
                "strategy_id": "threshold:ridge",
                "score": 0.3,
                "direction": "LONG",
                "eligible": True,
                "abstain": False,
                "fold_id": "fold-0",
                "split": "test",
            },
            {
                "date": "2026-01-05",
                "instrument": "TLT",
                "strategy_id": "threshold:forest",
                "score": -0.4,
                "direction": "SHORT",
                "eligible": True,
                "abstain": False,
                "fold_id": "fold-1",
                "split": "test",
            },
        ]
    )

    positions = trade_intent_envelope_to_candidate_positions(
        intents, PortfolioSpec(top_k=1, single_name_cap=0.25)
    )

    active = positions.loc[positions["candidate_id"] != DEFAULT_CANDIDATE_ID]
    assert active[["candidate_id", "fold_id", "split", "target_weight"]].to_dict("records") == [
        {
            "candidate_id": "threshold:ridge",
            "fold_id": "fold-0",
            "split": "test",
            "target_weight": 0.25,
        },
        {
            "candidate_id": "threshold:forest",
            "fold_id": "fold-1",
            "split": "test",
            "target_weight": -0.25,
        },
    ]

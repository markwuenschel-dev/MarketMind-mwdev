"""Candidate portfolio construction from strategy-owned trade intents."""

from __future__ import annotations

import pandas as pd
import pytest

from pysrc.pipeline.candidate_portfolios import (
    build_candidate_portfolio_products_from_trade_intents,
)
from pysrc.pipeline.p2_config_loader import PortfolioSpec


@pytest.mark.determinism("d1")
def test_candidate_pipeline_consumes_trade_intents(
    deterministic_seed: int,
) -> None:
    _ = deterministic_seed
    intents = pd.DataFrame(
        [
            {
                "date": "2026-01-05",
                "instrument": "SPY",
                "strategy_id": "threshold",
                "score": 0.4,
                "direction": "LONG",
                "eligible": True,
                "abstain": False,
            }
        ]
    )
    panel = pd.DataFrame(
        [
            {
                "date": "2026-01-05",
                "instrument": "SPY",
                "forward_return_1d": 0.01,
            }
        ]
    )

    positions, outputs = build_candidate_portfolio_products_from_trade_intents(
        intents,
        panel,
        PortfolioSpec(top_k=1, single_name_cap=0.25),
        fold_id="fold-0",
        split="test",
    )

    assert positions[["candidate_id", "ticker", "target_weight"]].to_dict("records") == [
        {"candidate_id": "threshold", "ticker": "SPY", "target_weight": 0.25}
    ]
    assert outputs["model_id"].tolist() == ["threshold"]

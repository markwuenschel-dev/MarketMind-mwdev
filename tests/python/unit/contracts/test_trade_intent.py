"""Contract tests for strategy pre-sizing intents."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pysrc.contracts.trade_intent import TradeDirection, TradeIntent


@pytest.mark.determinism("d1")
def test_trade_intent_accepts_pre_sizing_long_intent(deterministic_seed: int) -> None:
    _ = deterministic_seed

    intent = TradeIntent(
        date="2026-01-05",
        instrument="SPY",
        interval="1d",
        strategy_id="momentum_tsmom",
        intent_id="momentum_tsmom:SPY:2026-01-05",
        score=0.42,
        direction=TradeDirection.LONG,
        confidence=0.8,
        eligible=True,
        abstain=False,
        source_model_id="ridge_v1",
        source_product_id="cas.v1:b3-256:prediction-panel",
        lineage={
            "feature_product_id": "cas.v1:b3-256:features",
            "config_id": "cas.v1:b3-256:strategy-config",
            "run_id": "run-123",
        },
    )

    assert intent.score == 0.42
    assert intent.direction is TradeDirection.LONG


@pytest.mark.determinism("d1")
@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"eligible": False, "direction": TradeDirection.LONG}, "ineligible"),
        ({"abstain": True, "direction": TradeDirection.LONG}, "abstaining"),
        ({"direction": TradeDirection.SHORT, "score": 0.1}, "negative"),
        ({"direction": TradeDirection.FLAT, "abstain": False, "score": 0.1}, "zero"),
    ],
)
def test_trade_intent_rejects_ambiguous_decision_states(
    deterministic_seed: int,
    overrides: dict[str, object],
    message: str,
) -> None:
    _ = deterministic_seed
    payload: dict[str, object] = {
        "date": "2026-01-05",
        "instrument": "SPY",
        "interval": "1d",
        "strategy_id": "momentum_tsmom",
        "intent_id": "momentum_tsmom:SPY:2026-01-05",
        "score": 0.42,
        "direction": TradeDirection.LONG,
        "confidence": 0.8,
        "eligible": True,
        "abstain": False,
        "source_model_id": None,
        "source_product_id": "cas.v1:b3-256:prediction-panel",
        "lineage": {"run_id": "run-123"},
    }
    payload.update(overrides)

    with pytest.raises(ValidationError, match=message):
        TradeIntent.model_validate(payload)

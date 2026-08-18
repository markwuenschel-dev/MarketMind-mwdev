from __future__ import annotations

import pytest

from pysrc.backtesting.spine.ledger.ledger import SimpleLedger


@pytest.mark.determinism("d1")
def test_ledger_updates_cash_and_position_directionally() -> None:
    snapshot = SimpleLedger(opening_cash=1000.0).apply(
        [
            {
                "symbol": "AAPL",
                "quantity": 2.0,
                "price": 100.0,
                "side": "BUY",
                "timestamp": "2026-01-01T00:00:00+00:00",
            },
            {
                "symbol": "AAPL",
                "quantity": 1.0,
                "price": 110.0,
                "side": "SELL",
                "timestamp": "2026-01-02T00:00:00+00:00",
            },
        ],
        None,
    )

    assert snapshot.cash == 910.0
    assert snapshot.positions[0].quantity == 1.0

"""Unit tests for reconciliation diff schema."""

from __future__ import annotations

import pytest

from pysrc.tuning.execution.reconciliation import compare_ledger_to_broker


@pytest.mark.determinism("d1")
def test_compare_ledger_to_broker_clean(deterministic_seed: int) -> None:
    _ = deterministic_seed
    diff = compare_ledger_to_broker(
        ledger_positions={"AAA": 10.0, "BBB": 5.0},
        broker_positions={"AAA": 10.0, "BBB": 5.0},
        ledger_cash=1000.0,
        broker_cash=1000.0,
        as_of_bar="2024-01-02T16:00:00Z",
    )
    assert diff.has_mismatch is False
    assert diff.position_diffs == ()
    assert diff.cash_diff is not None
    assert diff.cash_diff.delta == 0.0


@pytest.mark.determinism("d1")
def test_compare_ledger_to_broker_flags_position_and_cash_mismatch(deterministic_seed: int) -> None:
    _ = deterministic_seed
    diff = compare_ledger_to_broker(
        ledger_positions={"AAA": 10.0},
        broker_positions={"AAA": 9.0, "CCC": 1.0},
        ledger_cash=1000.0,
        broker_cash=999.0,
        as_of_bar="2024-01-02T16:05:00Z",
        tolerance=0.0,
    )
    assert diff.has_mismatch is True
    symbols = {item.symbol for item in diff.position_diffs}
    assert symbols == {"AAA", "CCC"}
    payload = diff.to_dict()
    assert payload["schema_version"] == "reconciliation_diff.v1"
    assert payload["has_mismatch"] is True

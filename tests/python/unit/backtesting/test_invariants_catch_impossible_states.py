from __future__ import annotations

import pytest

from pysrc.backtesting.validation.mechanical.properties.invariants import (
    InvariantViolation,
    assert_position_cash_directionality,
)


@pytest.mark.determinism("d1")
def test_invariants_catch_cash_mismatch() -> None:
    with pytest.raises(InvariantViolation):
        assert_position_cash_directionality(
            before_cash=100.0,
            after_cash=90.0,
            fills=[{"quantity": 1.0, "price": 20.0, "side": "BUY"}],
        )

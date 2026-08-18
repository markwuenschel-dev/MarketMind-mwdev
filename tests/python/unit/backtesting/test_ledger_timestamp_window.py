from __future__ import annotations

from datetime import UTC, datetime

import pytest

from pysrc.backtesting.validation.mechanical.properties.invariants import (
    InvariantViolation,
    assert_fill_timestamps_within_window,
)


@pytest.mark.determinism("d1")
def test_fill_timestamp_window_violation_is_caught() -> None:
    with pytest.raises(InvariantViolation):
        assert_fill_timestamps_within_window(
            [{"timestamp": "2027-01-01T00:00:00+00:00"}],
            datetime(2026, 1, 1, tzinfo=UTC),
            datetime(2026, 12, 31, tzinfo=UTC),
        )

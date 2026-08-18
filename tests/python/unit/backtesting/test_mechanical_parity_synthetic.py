from __future__ import annotations

import pytest

from pysrc.backtesting.contracts.plan import DeterminismTier
from pysrc.backtesting.validation.mechanical.parity.suite import compare_metrics


@pytest.mark.determinism("d1")
def test_mechanical_parity_respects_tolerance_tier() -> None:
    assert compare_metrics({"total_return": 1.0}, {"total_return": 1.0 + 1e-10}, DeterminismTier.D1)
    assert not compare_metrics({"total_return": 1.0}, {"total_return": 1.1}, DeterminismTier.D1)

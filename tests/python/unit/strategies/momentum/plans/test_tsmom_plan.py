from __future__ import annotations

import pytest

from pysrc.strategies.momentum.plans.tsmom import build_plan

pytestmark = [pytest.mark.unit, pytest.mark.determinism("d1")]


def test_tsmom_plan_uses_rolling_zscore_and_vol_scale() -> None:
    plan = build_plan({})
    assert [step.op for step in plan.steps] == [
        "feature.returns",
        "scaling.zscore_roll",
        "stats.rolling_std",
        "momentum.vol_scale",
    ]

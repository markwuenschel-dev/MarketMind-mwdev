from __future__ import annotations

import pytest

from pysrc.strategies.momentum.plans.dual import build_plan

pytestmark = [pytest.mark.unit, pytest.mark.determinism("d1")]


def test_dual_plan_combines_absolute_and_relative_signals() -> None:
    plan = build_plan({})
    ops = [step.op for step in plan.steps]
    assert "scaling.zscore_roll" in ops
    assert "momentum.xsec_rank" in ops
    assert plan.steps[-1].op == "momentum.vol_scale"

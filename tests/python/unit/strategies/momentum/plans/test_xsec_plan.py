from __future__ import annotations

import pytest

from pysrc.preprocessor.graph.factory import register_builtin_ops
from pysrc.strategies.momentum.plans.industry import build_plan as build_industry_plan
from pysrc.strategies.momentum.plans.residual import build_plan as build_residual_plan
from pysrc.strategies.momentum.plans.xsec import build_plan
from pysrc.strategies.pipeline_strategy import FeaturePlan

pytestmark = [pytest.mark.unit, pytest.mark.determinism("d1")]


def test_xsec_plan_uses_registered_ops() -> None:
    register_builtin_ops()
    plan = build_plan({})
    assert isinstance(plan, FeaturePlan)
    assert [step.op for step in plan.steps] == [
        "feature.returns",
        "momentum.xsec_rank",
        "stats.rolling_std",
        "momentum.vol_scale",
    ]


def test_vol_scale_is_final_step_for_required_variants() -> None:
    plans = [build_plan({}), build_industry_plan({}), build_residual_plan({})]
    for plan in plans:
        assert plan.steps[-1].op == "momentum.vol_scale"

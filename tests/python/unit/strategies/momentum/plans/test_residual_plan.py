from __future__ import annotations

import pytest

from pysrc.preprocessor.graph.factory import register_builtin_ops, registry_snapshot
from pysrc.strategies.momentum.exceptions import FeatureFlagError
from pysrc.strategies.momentum.plans.residual import build_kalman_plan, build_plan

pytestmark = [pytest.mark.unit, pytest.mark.determinism("d1")]


def test_residual_plan_uses_residual_ols_then_rank_then_vol_scale() -> None:
    register_builtin_ops()
    registry, _ = registry_snapshot()
    plan = build_plan({"factor_ret_cols": ["market_return", "sector_return"]})
    ops = [step.op for step in plan.steps]
    assert ops == [
        "feature.returns",
        "momentum.residual_ols",
        "momentum.xsec_rank",
        "stats.rolling_std",
        "momentum.vol_scale",
    ]
    for step in plan.steps:
        assert step.op in registry
    assert plan.steps[1].kwargs["factor_ret_cols"] == ["market_return", "sector_return"]
    assert plan.steps[-1].op == "momentum.vol_scale"


def test_kalman_plan_requires_feature_flag() -> None:
    with pytest.raises(FeatureFlagError, match="enable_kalman_residual=True"):
        build_kalman_plan({})


def test_kalman_plan_switches_to_residual_kf_when_enabled() -> None:
    plan = build_kalman_plan(
        {
            "enable_kalman_residual": True,
            "factor_ret_cols": ["market_return"],
            "process_noise": 1e-5,
            "obs_noise": 2e-3,
        }
    )
    ops = [step.op for step in plan.steps]
    assert ops == [
        "feature.returns",
        "momentum.residual_kf",
        "momentum.xsec_rank",
        "stats.rolling_std",
        "momentum.vol_scale",
    ]
    assert plan.steps[1].kwargs["process_noise"] == pytest.approx(1e-5)
    assert plan.steps[1].kwargs["obs_noise"] == pytest.approx(2e-3)
    assert plan.steps[-1].op == "momentum.vol_scale"

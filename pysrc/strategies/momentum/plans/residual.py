from __future__ import annotations

from typing import Any

from pysrc.strategies.momentum.exceptions import FeatureFlagError
from pysrc.strategies.pipeline_strategy import FeaturePlan, FeatureStep


def build_plan(params: dict[str, Any]) -> FeaturePlan:
    factor_ret_cols = list(params.get("factor_ret_cols", ["market_return"]))
    target_vol = float(params.get("target_vol", 0.15))
    max_leverage = float(params.get("max_leverage", 2.0))
    return FeaturePlan.from_steps(
        [
            FeatureStep(
                op="feature.returns",
                inputs=("close",),
                kwargs={"column": "close"},
            ),
            FeatureStep(
                op="momentum.residual_ols",
                inputs=("returns",),
                kwargs={
                    "asset_ret_col": "returns",
                    "factor_ret_cols": factor_ret_cols,
                    "window": int(params.get("residual_window", 63)),
                    "out_col": "residual_signal",
                },
            ),
            FeatureStep(
                op="momentum.xsec_rank",
                inputs=("residual_signal",),
                kwargs={
                    "col": "residual_signal",
                    "window": int(params.get("rank_window", 252)),
                    "skip": int(params.get("skip_window", 21)),
                    "out_col": "mom_rank",
                },
            ),
            FeatureStep(
                op="stats.rolling_std",
                inputs=("returns",),
                kwargs={"col": "returns", "window": 60, "out_col": "realized_vol_60"},
            ),
            FeatureStep(
                op="momentum.vol_scale",
                inputs=("mom_rank", "realized_vol_60"),
                kwargs={
                    "signal_col": "mom_rank",
                    "vol_col": "realized_vol_60",
                    "target_vol": target_vol,
                    "max_leverage": max_leverage,
                    "out_col": "mom_scaled",
                },
            ),
        ]
    )


def build_kalman_plan(params: dict[str, Any]) -> FeaturePlan:
    if params.get("enable_kalman_residual") is not True:
        raise FeatureFlagError(
            "MomentumStrategy residual_kalman requires enable_kalman_residual=True."
        )

    factor_ret_cols = list(params.get("factor_ret_cols", ["market_return"]))
    target_vol = float(params.get("target_vol", 0.15))
    max_leverage = float(params.get("max_leverage", 2.0))
    return FeaturePlan.from_steps(
        [
            FeatureStep(
                op="feature.returns",
                inputs=("close",),
                kwargs={"column": "close"},
            ),
            FeatureStep(
                op="momentum.residual_kf",
                inputs=("returns",),
                kwargs={
                    "asset_ret_col": "returns",
                    "factor_ret_cols": factor_ret_cols,
                    "process_noise": float(params.get("process_noise", 1e-4)),
                    "obs_noise": float(params.get("obs_noise", 1e-3)),
                    "out_col": "residual_signal",
                },
            ),
            FeatureStep(
                op="momentum.xsec_rank",
                inputs=("residual_signal",),
                kwargs={
                    "col": "residual_signal",
                    "window": int(params.get("rank_window", 252)),
                    "skip": int(params.get("skip_window", 21)),
                    "out_col": "mom_rank",
                },
            ),
            FeatureStep(
                op="stats.rolling_std",
                inputs=("returns",),
                kwargs={"col": "returns", "window": 60, "out_col": "realized_vol_60"},
            ),
            FeatureStep(
                op="momentum.vol_scale",
                inputs=("mom_rank", "realized_vol_60"),
                kwargs={
                    "signal_col": "mom_rank",
                    "vol_col": "realized_vol_60",
                    "target_vol": target_vol,
                    "max_leverage": max_leverage,
                    "out_col": "mom_scaled",
                },
            ),
        ]
    )

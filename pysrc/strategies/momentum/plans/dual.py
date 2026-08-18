from __future__ import annotations

from typing import Any

from pysrc.strategies.pipeline_strategy import FeaturePlan, FeatureStep


def build_plan(params: dict[str, Any]) -> FeaturePlan:
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
                op="scaling.zscore_roll",
                inputs=("returns",),
                kwargs={"col": "returns", "window": 252, "min_samples": 252, "out_col": "tsmom_z"},
            ),
            FeatureStep(
                op="momentum.xsec_rank",
                inputs=("returns",),
                kwargs={"col": "returns", "window": 252, "skip": 21, "out_col": "mom_rank"},
            ),
            FeatureStep(
                op="momentum.residual_ols",
                inputs=("tsmom_z", "mom_rank"),
                kwargs={
                    "asset_ret_col": "tsmom_z",
                    "factor_ret_cols": ["mom_rank"],
                    "window": int(params.get("dual_window", 63)),
                    "out_col": "dual_score",
                },
            ),
            FeatureStep(
                op="stats.rolling_std",
                inputs=("returns",),
                kwargs={"col": "returns", "window": 60, "out_col": "realized_vol_60"},
            ),
            FeatureStep(
                op="momentum.vol_scale",
                inputs=("dual_score", "realized_vol_60"),
                kwargs={
                    "signal_col": "dual_score",
                    "vol_col": "realized_vol_60",
                    "target_vol": target_vol,
                    "max_leverage": max_leverage,
                    "out_col": "mom_scaled",
                },
            ),
        ]
    )

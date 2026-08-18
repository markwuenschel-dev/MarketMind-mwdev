from __future__ import annotations

from typing import Any

from pysrc.strategies.pipeline_strategy import FeaturePlan, FeatureStep


def build_plan(params: dict[str, Any]) -> FeaturePlan:
    target_vol = float(params.get("target_vol", 0.15))
    max_leverage = float(params.get("max_leverage", 2.0))
    sector_col = str(params.get("sector_col", "sector"))
    return FeaturePlan.from_steps(
        [
            FeatureStep(
                op="feature.returns",
                inputs=("close",),
                kwargs={"column": "close"},
            ),
            FeatureStep(
                op="momentum.industry_score",
                inputs=("returns", sector_col),
                kwargs={
                    "ret_col": "returns",
                    "sector_col": sector_col,
                    "window": int(params.get("industry_window", 63)),
                    "out_col": "industry_score",
                },
            ),
            FeatureStep(
                op="stats.rolling_std",
                inputs=("returns",),
                kwargs={"col": "returns", "window": 60, "out_col": "realized_vol_60"},
            ),
            FeatureStep(
                op="momentum.vol_scale",
                inputs=("industry_score", "realized_vol_60"),
                kwargs={
                    "signal_col": "industry_score",
                    "vol_col": "realized_vol_60",
                    "target_vol": target_vol,
                    "max_leverage": max_leverage,
                    "out_col": "mom_scaled",
                },
            ),
        ]
    )

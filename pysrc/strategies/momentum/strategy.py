"""Governed momentum strategies on the canonical pipeline path (Programming Guidelines §3.4, §4.3).

Plans are produced only through the ``_PLAN_BUILDERS`` factory table. Signal generation
reads materialized features plus ``StrategyContext.pit_provenance`` when governed; it does
not open raw datasets.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

try:
    import polars as pl
except (ImportError, ModuleNotFoundError):
    pl = None  # type: ignore[assignment]

from pysrc.strategies.momentum.alpha_ir import AlphaIR
from pysrc.strategies.momentum.plans.dual import build_plan as build_dual_plan
from pysrc.strategies.momentum.plans.ensemble import build_plan as build_ensemble_plan
from pysrc.strategies.momentum.plans.industry import build_plan as build_industry_plan
from pysrc.strategies.momentum.plans.ml import build_plan as build_ml_plan
from pysrc.strategies.momentum.plans.residual import (
    build_kalman_plan,
)
from pysrc.strategies.momentum.plans.residual import (
    build_plan as build_residual_plan,
)
from pysrc.strategies.momentum.plans.tsmom import build_plan as build_tsmom_plan
from pysrc.strategies.momentum.plans.xsec import build_plan as build_xsec_plan
from pysrc.strategies.momentum.spec import MOMENTUM_VARIANTS, build_momentum_params
from pysrc.strategies.pipeline_strategy import (
    FeaturePlan,
    MaterializationError,
    PipelineError,
    PipelineStrategy,
    StrategyContext,
    StrategyRegistry,
    TradeIntent,
    ValidationError,
    _to_series_like,
    materialize_features,
)

PlanBuilder = Callable[[dict[str, Any]], FeaturePlan]


class MomentumStrategy(PipelineStrategy):
    _MIN_IC_UNIVERSE = 10
    _VALID_VARIANTS = MOMENTUM_VARIANTS

    _PLAN_BUILDERS: dict[str, PlanBuilder] = {
        "xsec": build_xsec_plan,
        "tsmom": build_tsmom_plan,
        "dual": build_dual_plan,
        "industry": build_industry_plan,
        "residual_ols": build_residual_plan,
        "residual_kalman": build_kalman_plan,
        "ensemble": build_ensemble_plan,
        "ml": build_ml_plan,
    }

    def __init__(self, *, variant: str = "xsec", **params: Any) -> None:
        merged = build_momentum_params(variant, params)
        super().__init__(**merged)
        self._active_ctx: StrategyContext | None = None
        self._last_alpha_ir: AlphaIR | None = None

    def features_plan(self) -> FeaturePlan:
        variant = str(self.params["variant"])
        builder = self._PLAN_BUILDERS.get(variant)
        if builder is None:
            raise ValueError(f"Unsupported momentum variant '{variant}'")
        plan = builder(dict(self.params))
        return plan

    def _features_to_pandas(self, features: pd.DataFrame | pl.DataFrame) -> pd.DataFrame:
        if isinstance(features, pd.DataFrame):
            return features
        if pl is not None and isinstance(features, pl.DataFrame):
            return features.to_pandas()
        raise TypeError("MomentumStrategy.generate_signal expects pandas or polars features")

    def _build_signal_series(self, features_pd: pd.DataFrame, *, final_col: str) -> pd.Series:
        return pd.Series(
            features_pd[final_col], index=features_pd.index, name=final_col, dtype=float
        )

    def _is_governed(self, ctx: StrategyContext | None) -> bool:
        return bool(self.params.get("governed", False)) or (
            ctx is not None and ctx.pit_provenance is not None
        )

    def _compute_cross_sectional_ic(
        self, features_pd: pd.DataFrame, signal: pd.Series
    ) -> float | None:
        if "returns" not in features_pd.columns:
            return None

        if isinstance(features_pd.index, pd.MultiIndex) and features_pd.index.nlevels >= 2:
            frame = features_pd.copy()
            frame["_signal"] = signal
            ic_values: list[float] = []
            for _, group in frame.groupby(level=0, sort=False):
                asset_count = int(group.index.get_level_values(-1).nunique())
                if asset_count < self._MIN_IC_UNIVERSE:
                    continue
                valid = group[["_signal", "returns"]].dropna()
                if len(valid) < self._MIN_IC_UNIVERSE:
                    continue
                ic = valid["_signal"].corr(valid["returns"])
                if pd.notna(ic):
                    ic_values.append(float(ic))
            return float(np.mean(ic_values)) if ic_values else None

        date_col = next(
            (
                name
                for name in ("date", "datetime", "timestamp", "as_of", "valid_time")
                if name in features_pd.columns
            ),
            None,
        )
        asset_col = next(
            (
                name
                for name in ("asset", "symbol", "ticker", "sid", "instrument")
                if name in features_pd.columns
            ),
            None,
        )
        if date_col is None or asset_col is None:
            return None

        frame = features_pd[[date_col, asset_col, "returns"]].copy()
        frame["_signal"] = signal.to_numpy(dtype=float, copy=False)
        grouped_ic_values: list[float] = []
        for _, group in frame.groupby(date_col, sort=False):
            if int(group[asset_col].nunique(dropna=True)) < self._MIN_IC_UNIVERSE:
                continue
            valid = group[["_signal", "returns"]].dropna()
            if len(valid) < self._MIN_IC_UNIVERSE:
                continue
            ic = valid["_signal"].corr(valid["returns"])
            if pd.notna(ic):
                grouped_ic_values.append(float(ic))
        return float(np.mean(grouped_ic_values)) if grouped_ic_values else None

    def _book_membership_payload(self, signal: pd.Series) -> dict[str, Any]:
        if isinstance(signal.index, pd.MultiIndex) and signal.index.nlevels >= 2:
            ordered = signal.groupby(level=signal.index.nlevels - 1, sort=False).last().sort_index()
        else:
            ordered = signal.sort_index()

        by_symbol: dict[str, str] = {}
        counts = {"long": 0, "short": 0, "flat": 0}
        for asset_id, weight in ordered.items():
            if pd.notna(weight) and float(weight) > 0.0:
                bucket = "long"
            elif pd.notna(weight) and float(weight) < 0.0:
                bucket = "short"
            else:
                bucket = "flat"
            by_symbol[str(asset_id)] = bucket
            counts[bucket] += 1
        return {"by_symbol": by_symbol, "counts": counts}

    def generate_signal(self, features: pd.DataFrame | pl.DataFrame) -> AlphaIR:
        features_pd = self._features_to_pandas(features)
        final_col = str(self.params.get("signal_out_col", "mom_scaled"))
        if final_col not in features_pd.columns:
            raise ValidationError(f"MomentumStrategy requires final signal column '{final_col}'")

        ctx = self._active_ctx
        governed = self._is_governed(ctx)
        if governed and (ctx is None or ctx.pit_provenance is None):
            raise ValidationError(
                "MomentumStrategy.generate_signal requires ctx.pit_provenance in governed execution."
            )

        signal = self._build_signal_series(features_pd, final_col=final_col)
        vol_col = str(self.params.get("vol_col", "realized_vol_60"))
        realized_vol = None
        if vol_col in features_pd.columns:
            realized_vol = pd.Series(
                features_pd[vol_col], index=features_pd.index, name=vol_col, dtype=float
            )

        alpha_ir = AlphaIR(
            signal=signal,
            information_coefficient=self._compute_cross_sectional_ic(features_pd, signal),
            realized_vol=realized_vol,
            task_embedding=np.zeros(64, dtype=np.float32),
            pit_provenance=ctx.pit_provenance if ctx is not None else None,
            variant=str(self.params["variant"]),
            diagnostics={
                "governed": governed,
                "n_assets": int(signal.notna().sum()),
                "final_col": final_col,
                "book_membership": self._book_membership_payload(signal),
                "beta_reversal_score": {
                    "status": "UNAVAILABLE",
                    "reason": "beta_reversal_score requires a governed input source and remains fail-closed in Phase I-E.",
                },
            },
        )
        self._last_alpha_ir = alpha_ir
        return alpha_ir

    def generate_trade_intent(self, ctx: StrategyContext) -> TradeIntent:
        t0 = time.perf_counter()
        self._active_ctx = ctx
        try:
            feats = materialize_features(ctx, self.features_plan())
            alpha_ir = self.generate_signal(feats)
            sig: pd.Series | pd.DataFrame = alpha_ir.signal

            if self.regime is not None:
                try:
                    gate = self.regime.gate(feats)
                    if isinstance(sig, pd.DataFrame):
                        sig = sig.mul(_to_series_like(sig, gate), axis=0)
                    else:
                        sig = sig * _to_series_like(sig, gate)
                except (
                    AttributeError,
                    TypeError,
                    ValidationError,
                    MaterializationError,
                    ValueError,
                ) as exc:
                    raise PipelineError(f"regime gating failed: {exc}") from exc

            if self.sizer is not None:
                try:
                    sig = self.sizer.size(sig)
                except (
                    AttributeError,
                    TypeError,
                    ValidationError,
                    MaterializationError,
                    ValueError,
                ) as exc:
                    raise PipelineError(f"position sizing failed: {exc}") from exc

            weights = sig
            if self.risk is not None:
                try:
                    weights = self.risk.clamp(weights=weights, prices=ctx.prices)
                except (
                    AttributeError,
                    TypeError,
                    ValidationError,
                    MaterializationError,
                    ValueError,
                ) as exc:
                    raise PipelineError(f"risk clamp failed: {exc}") from exc

            dt = time.perf_counter() - t0
            return TradeIntent(
                weights=weights,
                raw={"signal": alpha_ir.signal, "alpha_ir": alpha_ir, "features": feats},
                diagnostics={"latency_s": dt},
            )
        finally:
            self._active_ctx = None


StrategyRegistry.register("momentum", MomentumStrategy)
StrategyRegistry.register("momentum_tsmom", MomentumStrategy)
StrategyRegistry.register("momentum_dual", MomentumStrategy)
StrategyRegistry.register("momentum_industry", MomentumStrategy)
StrategyRegistry.register("momentum_residual", MomentumStrategy)
StrategyRegistry.register("momentum_kalman", MomentumStrategy)
StrategyRegistry.register("momentum_ensemble", MomentumStrategy)
StrategyRegistry.register("momentum_ml", MomentumStrategy)

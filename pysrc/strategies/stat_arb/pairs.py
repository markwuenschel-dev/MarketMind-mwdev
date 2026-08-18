from __future__ import annotations

import pandas as pd

from pysrc.artifact_registry.artifact_store import BundleBacktestArtifactStore
from pysrc.strategies.pipeline_strategy import (
    FeaturePlan,
    FeatureStep,
    PipelineStrategy,
    StrategyRegistry,
)

from .common.diagnostics import (
    build_execution_assumptions_payload,
    build_stat_validity_payload,
)
from .common.feature_contract import pairs_zscore_column, require_pairs_feature_columns
from .common.types import PairsColumns
from .config import PAIRS_DEFAULT, HedgeEstimator, PairsConfig


class StatArbPairsStrategy(PipelineStrategy):
    """Pairs stat-arb strategy using the canonical graph op path."""

    def __init__(
        self,
        leg_a: str,
        leg_b: str,
        config: PairsConfig | None = None,
        **params: object,
    ) -> None:
        super().__init__(**params)
        self.leg_a = leg_a
        self.leg_b = leg_b
        self.config: PairsConfig = config or PAIRS_DEFAULT
        self._cols = PairsColumns(leg_a=leg_a, leg_b=leg_b)

    def features_plan(self) -> FeaturePlan:
        """Build the canonical pairs feature graph in dependency order."""

        beta_step = FeatureStep(
            op="pairs.beta",
            inputs=(),
            kwargs={
                "a": self.leg_a,
                "b": self.leg_b,
                "beta_window": self.config.beta_window,
                "out_col": self._cols.beta,
            },
        )

        spread_step = FeatureStep(
            op="pairs.spread",
            inputs=(),
            kwargs={
                "a": self.leg_a,
                "b": self.leg_b,
                "beta_col": self._cols.beta,
                "out_col": self._cols.spread,
            },
        )

        zscore_step = FeatureStep(
            op="scaling.zscore_roll",
            inputs=(self._cols.spread,),
            kwargs={
                "col": self._cols.spread,
                "window": self.config.zscore_window,
                "out_col": pairs_zscore_column(self._cols, self.config.zscore_window),
            },
        )

        half_life_step = FeatureStep(
            op="stats.half_life",
            inputs=(),
            kwargs={
                "col": self._cols.spread,
                "half_life_window": self.config.half_life_window,
                "out_col": self._cols.half_life,
            },
        )

        return FeaturePlan.from_steps([beta_step, spread_step, zscore_step, half_life_step])

    def generate_signal(self, features: pd.DataFrame | object) -> pd.Series:
        """Generate {-1, 0, +1} signals from materialized features."""

        if self.config.hedge_estimator == HedgeEstimator.KALMAN:
            raise NotImplementedError(
                "HedgeEstimator.KALMAN requires stats.kf_beta op (Phase II). "
                "Use HedgeEstimator.OLS for Phase I-D execution."
            )

        if not isinstance(features, pd.DataFrame):
            raise TypeError(
                "StatArbPairsStrategy.generate_signal expects a pandas DataFrame of features"
            )

        z_col = pairs_zscore_column(self._cols, self.config.zscore_window)
        hl_col = self._cols.half_life

        require_pairs_feature_columns(
            features.columns,
            cols=self._cols,
            zscore_window=self.config.zscore_window,
        )

        z = features[z_col]
        half_life = features[hl_col]
        signals = pd.Series(0, index=features.index, dtype="int8")

        position = 0
        bars_held = 0

        for i, (z_val, hl_val) in enumerate(zip(z, half_life, strict=False)):
            hl_valid = pd.notna(hl_val) and (
                self.config.min_half_life <= float(hl_val) <= self.config.max_half_life
            )

            if position == 0:
                bars_held = 0
                if not hl_valid or pd.isna(z_val):
                    signals.iloc[i] = 0
                    continue

                if z_val <= -self.config.entry_z:
                    position = 1
                elif z_val >= self.config.entry_z:
                    position = -1
            else:
                bars_held += 1
                exit_band = pd.notna(z_val) and abs(float(z_val)) <= self.config.exit_z
                force_exit = bars_held >= self.config.max_hold_days
                if exit_band or force_exit:
                    position = 0
                    bars_held = 0

            signals.iloc[i] = position

        return signals

    def write_artifacts(
        self,
        store: BundleBacktestArtifactStore,
        *,
        signals: pd.Series | None = None,
        feats: pd.DataFrame | None = None,
        run_id: str | None = None,
        evaluation_window: str = "unknown",
        **kwargs: object,
    ) -> None:
        """Emit execution_assumptions.json and stat_validity_report.json via canonical store."""

        if signals is None or feats is None:
            signal_count = 0
            half_life_bars = None
            mean_spread_zscore = None
        else:
            signal_count = int((signals != 0).sum())
            z_col = pairs_zscore_column(self._cols, self.config.zscore_window)
            mean_spread_zscore = float(feats[z_col].mean()) if z_col in feats.columns else None
            hl_col = self._cols.half_life
            if hl_col in feats.columns:
                valid_hl = feats[hl_col].dropna()
                half_life_bars = float(valid_hl.iloc[-1]) if len(valid_hl) else None
            else:
                half_life_bars = None

        strategy_name = f"stat_arb_pairs:{self.leg_a}-{self.leg_b}"
        store.put_json(
            "execution_assumptions.json",
            build_execution_assumptions_payload(
                strategy=strategy_name,
                config=self.config,
            ),
        )
        store.put_json(
            "stat_validity_report.json",
            build_stat_validity_payload(
                strategy=strategy_name,
                pair=(self.leg_a, self.leg_b),
                config=self.config,
                evaluation_window=evaluation_window,
                half_life_bars=half_life_bars,
                mean_spread_zscore=mean_spread_zscore,
                signal_count=signal_count,
                pit_compliant=True,
            ),
        )


StrategyRegistry.register("stat_arb_pairs", StatArbPairsStrategy)

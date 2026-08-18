"""Execution-time adapter for panel prediction-threshold strategies."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from pysrc.pipeline.p2_config_loader import PortfolioSpec
from pysrc.strategies import build_threshold_intents
from pysrc.strategies.pipeline_strategy import PipelineStrategy, StrategyRegistry, StrategySignal

if TYPE_CHECKING:
    from polars import DataFrame as PolarsDataFrame
else:
    PolarsDataFrame = Any


@dataclass(frozen=True, slots=True)
class PromotionBundleRuntimeConfig:
    """Runtime config resolved from a Gate 7 promotion bundle."""

    run_id: str
    model_id: str
    strategy_id: str
    bundle_path: Path
    plan_hash: str
    top_k: int
    single_name_cap: float
    cost_bps: float


def load_promotion_bundle_runtime(bundle_dir: Path) -> PromotionBundleRuntimeConfig:
    bundle_dir = Path(bundle_dir)
    plan_path = bundle_dir / "plan.json"
    if not plan_path.is_file():
        raise FileNotFoundError(f"Missing plan.json in bundle: {bundle_dir}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    portfolio = plan.get("portfolio") or {}
    model_id = str(plan.get("model_id") or "xgboost")
    return PromotionBundleRuntimeConfig(
        run_id=str(plan.get("source_run_id") or plan.get("run_id") or ""),
        model_id=model_id,
        strategy_id=str(plan.get("strategy") or f"prediction_threshold_{model_id}"),
        bundle_path=bundle_dir,
        plan_hash=str(plan.get("plan_hash") or ""),
        top_k=int(portfolio.get("top_k", 20)),
        single_name_cap=float(portfolio.get("single_name_cap", 0.1)),
        cost_bps=float(portfolio.get("cost_bps", 10.0)),
    )


def predictions_to_intents_from_bundle(
    predictions: pd.DataFrame,
    runtime: PromotionBundleRuntimeConfig,
    *,
    lineage: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Build threshold intents matching Gate 5 production bridge semantics."""

    lineage_payload = dict(lineage or {})
    lineage_payload.setdefault("run_id", runtime.run_id)
    lineage_payload.setdefault("model_id", runtime.model_id)
    parts: list[pd.DataFrame] = []
    for (fold_id, split), group in predictions.groupby(["fold_id", "split"], sort=True):
        frame = group.loc[:, ["date", "instrument", "prediction"]].copy()
        frame["prediction"] = (
            pd.to_numeric(frame["prediction"], errors="coerce").astype(float).abs()
        )
        frame = frame.loc[frame["prediction"] > 0.0]
        if frame.empty:
            continue
        intents = build_threshold_intents(
            frame,
            strategy_id=runtime.strategy_id,
            source_product_id="model_prediction_panel",
            lineage=lineage_payload,
            threshold=0.0,
            source_model_id=runtime.model_id,
        )
        intents["fold_id"] = str(fold_id)
        intents["split"] = str(split)
        parts.append(intents)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def runtime_to_portfolio_spec(runtime: PromotionBundleRuntimeConfig) -> PortfolioSpec:
    return PortfolioSpec(
        top_k=runtime.top_k,
        single_name_cap=runtime.single_name_cap,
        cost_bps=runtime.cost_bps,
        capacity_constraints=False,
    )


class PredictionThresholdStrategy(PipelineStrategy):
    """Strategy registry entry for promoted panel threshold models."""

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.model_id = str(params.get("model_id", "xgboost"))
        self.strategy_id = f"prediction_threshold_{self.model_id}"
        bundle_dir = params.get("bundle_dir")
        self._bundle_dir = Path(bundle_dir) if bundle_dir is not None else None
        self._runtime: PromotionBundleRuntimeConfig | None = None

    def resolve_runtime(self, bundle_dir: Path | None = None) -> PromotionBundleRuntimeConfig:
        path = Path(bundle_dir) if bundle_dir is not None else self._bundle_dir
        if path is None:
            raise ValueError("bundle_dir required to resolve promotion runtime config")
        self._runtime = load_promotion_bundle_runtime(path)
        return self._runtime

    def features_plan(self) -> Any:
        raise NotImplementedError(
            "Panel promotion strategies use predictions_to_intents_from_bundle, not FeaturePlan IR"
        )

    def generate_signal(self, features: pd.DataFrame | PolarsDataFrame) -> StrategySignal:
        if not isinstance(features, pd.DataFrame):
            features = features.to_pandas()
        runtime = self._runtime or self.resolve_runtime()
        preds = features.copy()
        if "model_id" not in preds.columns:
            preds["model_id"] = runtime.model_id
        preds = preds.loc[preds["model_id"].astype(str) == runtime.model_id]
        intents = predictions_to_intents_from_bundle(preds, runtime)
        return intents


StrategyRegistry.register("prediction_threshold", PredictionThresholdStrategy)
StrategyRegistry.register("prediction_threshold_xgboost", PredictionThresholdStrategy)


__all__ = [
    "PredictionThresholdStrategy",
    "PromotionBundleRuntimeConfig",
    "load_promotion_bundle_runtime",
    "predictions_to_intents_from_bundle",
    "runtime_to_portfolio_spec",
]

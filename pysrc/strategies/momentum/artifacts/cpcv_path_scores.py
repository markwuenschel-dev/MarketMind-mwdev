from __future__ import annotations

import contextlib
import json
from collections.abc import Mapping
from typing import Any

import pandas as pd

from pysrc.artifact_registry.bundle_writer import BundleWriter
from pysrc.strategies.pipeline_strategy import BacktestConfig, TradeIntent, backtest_portfolio

_DATE_COLUMNS = ("date", "valid_time", "datetime", "timestamp", "as_of")
_ASSET_COLUMNS = ("asset", "symbol", "ticker", "sid", "instrument")


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def compute_payload_hash(payload: Mapping[str, Any]) -> str:
    normalized = json.loads(json.dumps(payload, sort_keys=True, default=_json_default))
    return f"sha256:{BundleWriter.compute_config_hash(normalized)}"


def _resolve_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def _normalize_wide_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    with contextlib.suppress(TypeError, ValueError):
        normalized.index = pd.to_datetime(normalized.index)
    normalized.columns = [str(column) for column in normalized.columns]
    return normalized.sort_index().sort_index(axis=1).astype(float)


def normalize_close_prices(prices: pd.DataFrame | pd.Series) -> pd.DataFrame:
    if isinstance(prices, pd.Series):
        return _normalize_wide_frame(prices.astype(float).to_frame(name="asset"))

    if not isinstance(prices, pd.DataFrame):
        raise TypeError("close prices must be a pandas Series or DataFrame")

    if "close" in prices.columns:
        date_col = _resolve_column(prices, _DATE_COLUMNS)
        asset_col = _resolve_column(prices, _ASSET_COLUMNS)
        if date_col is not None and asset_col is not None:
            wide = prices.pivot(index=date_col, columns=asset_col, values="close")
            return _normalize_wide_frame(wide)
        if list(prices.columns) == ["close"] or prices.shape[1] == 1:
            return _normalize_wide_frame(prices[["close"]].rename(columns={"close": "asset"}))

    return _normalize_wide_frame(prices)


def normalize_weights(
    weights: pd.DataFrame | pd.Series,
    features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if isinstance(weights, pd.DataFrame):
        if isinstance(weights.index, pd.MultiIndex) and weights.index.nlevels >= 2:
            if weights.shape[1] == 1:
                return normalize_weights(weights.iloc[:, 0], features=features)
            raise ValueError("multi-index weight frames must have a single value column")

        date_col = _resolve_column(weights, _DATE_COLUMNS)
        asset_col = _resolve_column(weights, _ASSET_COLUMNS)
        if date_col is not None and asset_col is not None:
            value_col = "weight" if "weight" in weights.columns else None
            if value_col is None:
                for candidate in weights.columns:
                    if candidate not in {date_col, asset_col}:
                        value_col = str(candidate)
                        break
            if value_col is None:
                raise ValueError("unable to identify a weight value column")
            wide = weights.pivot(index=date_col, columns=asset_col, values=value_col)
            return _normalize_wide_frame(wide)

        return _normalize_wide_frame(weights)

    if isinstance(weights, pd.Series):
        if isinstance(weights.index, pd.MultiIndex) and weights.index.nlevels >= 2:
            wide = weights.unstack(level=weights.index.nlevels - 1)
            return _normalize_wide_frame(wide)

        if features is not None:
            date_col = _resolve_column(features, _DATE_COLUMNS)
            asset_col = _resolve_column(features, _ASSET_COLUMNS)
            if date_col is not None and asset_col is not None and len(features) == len(weights):
                long_weights = pd.DataFrame(
                    {
                        "date": pd.to_datetime(features[date_col]),
                        "asset": features[asset_col].astype(str),
                        "weight": weights.to_numpy(dtype=float, copy=False),
                    }
                )
                wide = long_weights.pivot(index="date", columns="asset", values="weight")
                return _normalize_wide_frame(wide)

        return _normalize_wide_frame(weights.astype(float).to_frame(name="asset"))

    raise TypeError("weights must be a pandas Series or DataFrame")


def _extract_features(trade_intent: TradeIntent) -> pd.DataFrame:
    features = trade_intent.raw.get("features")
    if hasattr(features, "to_pandas"):
        features = features.to_pandas()
    if not isinstance(features, pd.DataFrame):
        raise TypeError("momentum CPCV path scoring requires pandas feature outputs")
    return features


def _compute_sharpe_ratio(pnl: pd.Series) -> float:
    valid = pnl.dropna()
    if valid.empty:
        return 0.0
    std = float(valid.std(ddof=0))
    if std == 0.0:
        return 0.0
    return float((valid.mean() / std) * (252.0**0.5))


def _shared_cost_payload(
    *,
    commission_bps: float,
    slippage_bps: float,
    cost_model_id: str,
) -> dict[str, Any]:
    return {
        "commission_bps": float(commission_bps),
        "slippage_bps": float(slippage_bps),
        "cost_model_id": str(cost_model_id),
    }


def build_cpcv_path_score_surface(
    *,
    variant: str,
    trade_intent: TradeIntent,
    prices: pd.DataFrame | pd.Series,
    splits_manifest: Mapping[str, Any],
    commission_bps: float,
    slippage_bps: float,
    cost_model_id: str,
) -> dict[str, Any]:
    split_method = str(splits_manifest.get("split_method", "none"))
    if split_method != "cpcv":
        raise ValueError("cpcv_path_scores requires a CPCV splits_manifest")

    splits = splits_manifest.get("splits")
    if not isinstance(splits, list) or not splits:
        raise ValueError("cpcv_path_scores requires a non-empty CPCV split surface")

    features = _extract_features(trade_intent)
    prices_wide = normalize_close_prices(prices)
    weights_wide = normalize_weights(trade_intent.weights, features=features)
    aligned_weights = weights_wide.reindex(
        index=prices_wide.index,
        columns=prices_wide.columns,
        fill_value=0.0,
    ).fillna(0.0)
    cost_per_unit_turnover = (float(commission_bps) + float(slippage_bps)) / 10000.0
    backtest = backtest_portfolio(
        prices_wide,
        aligned_weights,
        BacktestConfig(cost_per_unit_turnover=cost_per_unit_turnover),
    )

    evaluations: list[dict[str, Any]] = []
    in_sample_scores: list[float] = []
    out_of_sample_scores: list[float] = []
    for split in splits:
        in_sample = backtest.iloc[list(split["train_indices"])]
        out_of_sample = backtest.iloc[list(split["test_indices"])]
        in_sample_sharpe = _compute_sharpe_ratio(in_sample["pnl"])
        out_of_sample_sharpe = _compute_sharpe_ratio(out_of_sample["pnl"])
        in_sample_scores.append(in_sample_sharpe)
        out_of_sample_scores.append(out_of_sample_sharpe)
        evaluations.append(
            {
                "trial_id": str(variant),
                "path_id": str(split["path_id"]),
                "in_sample_net_sharpe": in_sample_sharpe,
                "out_of_sample_net_sharpe": out_of_sample_sharpe,
            }
        )

    summary = {
        "mean_in_sample_net_sharpe": float(sum(in_sample_scores) / len(in_sample_scores)),
        "mean_out_of_sample_net_sharpe": float(
            sum(out_of_sample_scores) / len(out_of_sample_scores)
        ),
        "mean_turnover": float(backtest["turnover"].mean()),
        "total_costs": float(backtest["costs"].sum()),
    }
    return {
        "schema_version": "1.0.0",
        "variant": str(variant),
        "split_surface_hash": compute_payload_hash(dict(splits_manifest)),
        "shared_cost_hash": compute_payload_hash(
            _shared_cost_payload(
                commission_bps=commission_bps,
                slippage_bps=slippage_bps,
                cost_model_id=cost_model_id,
            )
        ),
        "summary": summary,
        "evaluations": evaluations,
    }


__all__ = [
    "build_cpcv_path_score_surface",
    "compute_payload_hash",
    "normalize_close_prices",
    "normalize_weights",
]

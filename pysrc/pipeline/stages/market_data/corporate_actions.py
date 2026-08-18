"""Pure corporate-action adjustment for raw Massive daily bars."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd  # type: ignore[import-untyped]

RAW_PRICE_COLUMNS: Final[tuple[str, ...]] = ("open", "high", "low", "close")
RAW_VOLUME_COLUMN: Final[str] = "volume"
ADJUSTED_RETURN_PRICE_COLUMN: Final[str] = "adj_close"
FORWARD_LABEL_PRICE_COLUMN: Final[str] = "adj_close"
FILTER_PRICE_COLUMN: Final[str] = "raw_close"
LIQUIDITY_VOLUME_COLUMN: Final[str] = "raw_volume"


class CorporateActionAdjustmentError(ValueError):
    """Raised when inputs cannot produce a governed adjusted panel."""


@dataclass(frozen=True)
class CorporateActionAdjustmentConfig:
    """Configuration for diagnostics emitted by the pure adjuster."""

    extreme_return_threshold: float = 0.50


def build_adjusted_ohlcv_panel(
    bars: pd.DataFrame,
    *,
    splits: pd.DataFrame,
    dividends: pd.DataFrame,
    config: CorporateActionAdjustmentConfig | None = None,
) -> pd.DataFrame:
    """
    Build an adjusted OHLCV panel from raw Massive flat-file bars and Massive
    corporate-action factors.

    The input bars remain the raw truth. Adjusted prices are derived columns for
    return signals and forward labels; raw prices/volume remain available for
    filters, dollar volume, liquidity, and tradability checks.
    """
    cfg = config or CorporateActionAdjustmentConfig()
    out = _normalize_bars(bars)
    split_factor = _future_action_factor(
        out,
        actions=splits,
        event_date_column="execution_date",
        fallback_ratio_columns=("split_from", "split_to"),
    )
    dividend_factor = _future_action_factor(
        out,
        actions=dividends,
        event_date_column="ex_dividend_date",
        fallback_ratio_columns=None,
    )
    split_event_flag = _same_day_action_flag(
        out,
        actions=splits,
        event_date_column="execution_date",
        fallback_ratio_columns=("split_from", "split_to"),
    )
    dividend_event_flag = _same_day_action_flag(
        out,
        actions=dividends,
        event_date_column="ex_dividend_date",
        fallback_ratio_columns=None,
    )

    out["split_adjustment_factor"] = split_factor
    out["dividend_adjustment_factor"] = dividend_factor
    out["total_price_adjustment_factor"] = (
        out["split_adjustment_factor"] * out["dividend_adjustment_factor"]
    )
    out["adj_open"] = out["raw_open"] * out["total_price_adjustment_factor"]
    out["adj_high"] = out["raw_high"] * out["total_price_adjustment_factor"]
    out["adj_low"] = out["raw_low"] * out["total_price_adjustment_factor"]
    out["adj_close"] = out["raw_close"] * out["total_price_adjustment_factor"]
    out["adj_volume"] = out["raw_volume"] / out["split_adjustment_factor"].replace(0.0, pd.NA)

    grouped = out.groupby("symbol", sort=False, group_keys=False)
    out["raw_return_1d"] = grouped["raw_close"].pct_change()
    out["adjusted_return_1d"] = grouped["adj_close"].pct_change()
    out["raw_vs_adjusted_return_delta"] = out["raw_return_1d"] - out["adjusted_return_1d"]

    out["corporate_action_flag"] = pd.Series(
        [
            bool(split != 1.0 or dividend != 1.0) or bool(split_event) or bool(dividend_event)
            for split, dividend, split_event, dividend_event in zip(
                out["split_adjustment_factor"].tolist(),
                out["dividend_adjustment_factor"].tolist(),
                split_event_flag.tolist(),
                dividend_event_flag.tolist(),
                strict=False,
            )
        ],
        index=out.index,
        dtype=object,
    )
    out["extreme_raw_return_flag"] = _extreme_flag(
        out["raw_return_1d"],
        threshold=cfg.extreme_return_threshold,
    )
    out["extreme_adjusted_return_flag"] = _extreme_flag(
        out["adjusted_return_1d"],
        threshold=cfg.extreme_return_threshold,
    )

    out["bar_quality_mode"] = "none"
    out["bar_quality_detector_version"] = pd.Series(
        [None] * len(out), index=out.index, dtype=object
    )
    out["bar_quality_flags"] = pd.Series(
        [[] for _ in range(len(out))], index=out.index, dtype=object
    )
    return out.drop(columns=["_bar_date"]).reset_index(drop=True)


def _normalize_bars(bars: pd.DataFrame) -> pd.DataFrame:
    required = {"open", "high", "low", "close", "volume"}
    missing = sorted(required - set(bars.columns))
    if missing:
        raise CorporateActionAdjustmentError(f"raw bars missing columns: {missing}")

    out = bars.copy()
    if "symbol" not in out.columns:
        if "ticker" not in out.columns:
            raise CorporateActionAdjustmentError("raw bars require symbol or ticker")
        out["symbol"] = out["ticker"]
    if "date" not in out.columns:
        if "valid_time" not in out.columns:
            raise CorporateActionAdjustmentError("raw bars require date or valid_time")
        out["date"] = out["valid_time"]

    out["_bar_date"] = pd.to_datetime(out["date"]).dt.normalize().astype("datetime64[ns]")
    out["symbol"] = out["symbol"].astype(str)
    for column in RAW_PRICE_COLUMNS:
        out[f"raw_{column}"] = pd.to_numeric(out[column], errors="raise")
    out["raw_volume"] = pd.to_numeric(out[RAW_VOLUME_COLUMN], errors="raise")
    return out.sort_values(["symbol", "_bar_date"], kind="mergesort").reset_index(drop=True)


def _future_action_factor(
    bars: pd.DataFrame,
    *,
    actions: pd.DataFrame,
    event_date_column: str,
    fallback_ratio_columns: tuple[str, str] | None,
) -> pd.Series:
    if actions.empty:
        return pd.Series([1.0] * len(bars), index=bars.index, dtype="float64")

    normalized = _normalize_actions(
        actions,
        event_date_column=event_date_column,
        fallback_ratio_columns=fallback_ratio_columns,
    )
    if normalized.empty:
        return pd.Series([1.0] * len(bars), index=bars.index, dtype="float64")

    factors = pd.Series([1.0] * len(bars), index=bars.index, dtype="float64")
    for symbol, group in bars.groupby("symbol", sort=False):
        action_group = normalized.loc[normalized["symbol"] == symbol]
        if action_group.empty:
            continue
        event_dates = action_group["event_date"].to_numpy()
        event_factors = action_group["factor"].to_numpy(dtype="float64")
        cumulative: list[float] = []
        for bar_date in group["_bar_date"].tolist():
            include = event_dates > bar_date.to_datetime64()
            product = float(event_factors[include].prod()) if include.any() else 1.0
            cumulative.append(product)
        factors.loc[group.index] = cumulative
    return factors


def _same_day_action_flag(
    bars: pd.DataFrame,
    *,
    actions: pd.DataFrame,
    event_date_column: str,
    fallback_ratio_columns: tuple[str, str] | None,
) -> pd.Series:
    if actions.empty:
        return pd.Series([False] * len(bars), index=bars.index, dtype=object)

    normalized = _normalize_actions(
        actions,
        event_date_column=event_date_column,
        fallback_ratio_columns=fallback_ratio_columns,
    )
    if normalized.empty:
        return pd.Series([False] * len(bars), index=bars.index, dtype=object)

    event_keys = set(
        zip(normalized["symbol"].tolist(), normalized["event_date"].tolist(), strict=False)
    )
    return pd.Series(
        [
            bool((symbol, bar_date) in event_keys)
            for symbol, bar_date in zip(
                bars["symbol"].tolist(), bars["_bar_date"].tolist(), strict=False
            )
        ],
        index=bars.index,
        dtype=object,
    )


def _normalize_actions(
    actions: pd.DataFrame,
    *,
    event_date_column: str,
    fallback_ratio_columns: tuple[str, str] | None,
) -> pd.DataFrame:
    if "ticker" not in actions.columns and "symbol" not in actions.columns:
        return pd.DataFrame(columns=["symbol", "event_date", "factor"])
    if event_date_column not in actions.columns:
        return pd.DataFrame(columns=["symbol", "event_date", "factor"])

    out = pd.DataFrame(
        {
            "symbol": (
                actions["symbol"].astype(str)
                if "symbol" in actions.columns
                else actions["ticker"].astype(str)
            ),
            "event_date": (
                pd.to_datetime(actions[event_date_column], errors="coerce")
                .dt.normalize()
                .astype("datetime64[ns]")
            ),
            "factor": _historical_factor(actions, fallback_ratio_columns=fallback_ratio_columns),
        }
    )
    out = out.dropna(subset=["symbol", "event_date", "factor"])
    return out.loc[out["factor"] > 0.0].sort_values(["symbol", "event_date"], kind="mergesort")


def _historical_factor(
    actions: pd.DataFrame,
    *,
    fallback_ratio_columns: tuple[str, str] | None,
) -> pd.Series:
    if "historical_adjustment_factor" in actions.columns:
        factor = pd.to_numeric(actions["historical_adjustment_factor"], errors="coerce")
    else:
        factor = pd.Series([pd.NA] * len(actions), index=actions.index, dtype="Float64")

    if fallback_ratio_columns is None:
        return factor.fillna(1.0).astype("float64")

    numerator_column, denominator_column = fallback_ratio_columns
    if numerator_column in actions.columns and denominator_column in actions.columns:
        numerator = pd.to_numeric(actions[numerator_column], errors="coerce")
        denominator = pd.to_numeric(actions[denominator_column], errors="coerce")
        fallback = numerator / denominator.replace(0.0, pd.NA)
        factor = factor.fillna(fallback)
    return factor.fillna(1.0).astype("float64")


def _extreme_flag(values: pd.Series, *, threshold: float) -> pd.Series:
    return pd.Series(
        [bool(pd.notna(value) and abs(float(value)) >= threshold) for value in values.tolist()],
        index=values.index,
        dtype=object,
    )

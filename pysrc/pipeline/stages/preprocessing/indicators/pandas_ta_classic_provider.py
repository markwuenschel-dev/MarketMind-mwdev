"""pandas-ta-classic provider for compact W3-B indicator feature panels."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from pysrc.core.runtime.optional_imports import require
from pysrc.pipeline.stages.preprocessing.indicators.config import IndicatorLibraryConfig
from pysrc.pipeline.stages.preprocessing.indicators.registry import indicator_ids
from pysrc.pipeline.stages.preprocessing.indicators.schema import (
    REQUIRED_PROVIDER_INPUT_COLUMNS,
    W3B_INDICATOR_IDS,
)

logging.getLogger("pandas_ta_classic").setLevel(logging.ERROR)
logging.getLogger("pandas_ta_classic.utils._core").setLevel(logging.ERROR)


@dataclass(frozen=True, slots=True)
class IndicatorProviderResult:
    features: pd.DataFrame
    indicator_columns: tuple[str, ...]
    provider_mappings: dict[str, dict[str, object]]
    warmup: dict[str, int]


def load_pipeline_indicator_features(
    panel_path: Path,
    *,
    key_columns: tuple[str, ...] = ("date", "instrument"),
) -> IndicatorProviderResult:
    """Load precomputed indicator columns from a pipeline preprocessing product."""

    frame = pd.read_parquet(panel_path, engine="pyarrow")
    keys = list(key_columns)
    missing_keys = [column for column in keys if column not in frame.columns]
    if missing_keys:
        raise ValueError(f"Pipeline indicator panel missing key columns: {missing_keys}")

    indicator_columns = tuple(
        column for column in frame.columns if column not in keys and column in W3B_INDICATOR_IDS
    )
    if not indicator_columns:
        raise ValueError(f"Pipeline indicator panel at {panel_path} has no W3-B indicator columns.")

    features = frame.loc[:, [*keys, *indicator_columns]].copy()
    return IndicatorProviderResult(
        features=features,
        indicator_columns=indicator_columns,
        provider_mappings={"source": {"kind": "pipeline_preprocessing", "path": str(panel_path)}},
        warmup=dict.fromkeys(indicator_columns, 0),
    )


def compute_pandas_ta_classic_features(
    panel: pd.DataFrame,
    config: IndicatorLibraryConfig | None = None,
    *,
    workers: int = 1,
    copy_input: bool = True,
    ta_scratch_path: Path | None = None,
) -> IndicatorProviderResult:
    """Compute the W3-B v1 wide feature panel at ``date x instrument`` grain.

    All library calls are isolated in this module so experiment code can only consume
    normalized, one-bar-lagged features.
    """

    resolved = config or IndicatorLibraryConfig()
    rows = _prepare_provider_rows(panel, copy_input=bool(copy_input))
    instruments = sorted(rows["instrument"].astype(str).unique().tolist())
    max_workers = max(1, int(workers))
    parts: list[pd.DataFrame] = []
    mappings: dict[str, dict[str, object]] = {}
    scratch_path: Path | None = None
    try:
        if max_workers <= 1 or len(instruments) <= 1:
            ta_module = require("pandas_ta_classic", purpose="W3-B technical indicators")
            for _, group in rows.groupby("instrument", sort=False):
                part, group_mappings = _compute_one_instrument(group, ta_module)
                parts.append(part)
                mappings.update(group_mappings)
        else:
            if ta_scratch_path is None:
                raise ValueError("ta_scratch_path is required when workers > 1")
            scratch_path = Path(ta_scratch_path)
            _atomic_write_parquet(rows, scratch_path)
            pool_workers = min(max_workers, len(instruments))
            id_batches = _instrument_id_batches(instruments, pool_workers)
            worker_args = [(str(scratch_path), batch) for batch in id_batches]
            with ProcessPoolExecutor(max_workers=pool_workers, max_tasks_per_child=1) as executor:
                batch_results = list(
                    executor.map(_compute_instrument_batch_from_parquet_worker, worker_args)
                )
            for batch_parts, batch_mappings in batch_results:
                parts.extend(batch_parts)
                mappings.update(batch_mappings)
    finally:
        if scratch_path is not None and scratch_path.is_file():
            scratch_path.unlink()

    if parts:
        computed = pd.concat(parts, axis=0, ignore_index=True)
    else:
        computed = pd.DataFrame(columns=("date", "instrument", *W3B_INDICATOR_IDS))

    indicator_columns = indicator_ids()
    for column in indicator_columns:
        if column not in computed.columns:
            computed[column] = np.nan
    computed = computed.loc[:, ["date", "instrument", *indicator_columns]].copy()
    computed.loc[:, list(indicator_columns)] = computed.groupby("instrument", sort=False)[
        list(indicator_columns)
    ].shift(int(resolved.lag_bars))
    computed.loc[:, list(indicator_columns)] = (
        computed.loc[:, list(indicator_columns)]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    if resolved.panel_float32:
        for column in indicator_columns:
            computed[column] = _coerce_indicator_float32(computed[column])
    warmup = {
        "pre_warmup_row_count": int(len(computed)),
        "post_warmup_row_count": int(computed.dropna(subset=list(indicator_columns)).shape[0]),
        "warmup_rows_removed": int(
            len(computed) - computed.dropna(subset=list(indicator_columns)).shape[0]
        ),
        "post_warmup_instrument_count": int(
            computed.dropna(subset=list(indicator_columns))["instrument"].nunique()
        ),
        "post_warmup_date_count": int(
            computed.dropna(subset=list(indicator_columns))["date"].nunique()
        ),
    }
    return IndicatorProviderResult(
        features=computed,
        indicator_columns=indicator_columns,
        provider_mappings=mappings,
        warmup=warmup,
    )


def _coerce_indicator_float32(series: pd.Series[Any]) -> pd.Series[Any]:
    values = pd.to_numeric(series, errors="coerce")
    arr = values.to_numpy(dtype="float64", copy=True)
    finfo = np.finfo(np.float32)
    invalid = ~np.isfinite(arr) | (arr < float(finfo.min)) | (arr > float(finfo.max))
    if bool(invalid.any()):
        arr[invalid] = np.nan
    return pd.Series(arr, index=series.index, name=series.name).astype("float32")


def _prepare_provider_rows(panel: pd.DataFrame, *, copy_input: bool) -> pd.DataFrame:
    missing = [column for column in REQUIRED_PROVIDER_INPUT_COLUMNS if column not in panel.columns]
    if missing:
        raise ValueError(f"W3-B indicator provider missing required columns: {missing}")

    if copy_input:
        rows = panel.copy()
    else:
        rows = panel.loc[:, list(REQUIRED_PROVIDER_INPUT_COLUMNS)].copy()
    rows["date"] = pd.to_datetime(rows["date"], utc=False, errors="coerce")
    if rows["date"].isna().any():
        raise ValueError("W3-B indicator provider requires parseable dates")
    rows["instrument"] = rows["instrument"].astype(str)
    rows = rows.sort_values(["instrument", "date"], kind="mergesort").reset_index(drop=True)
    for column in REQUIRED_PROVIDER_INPUT_COLUMNS:
        if column not in {"date", "instrument"}:
            rows[column] = pd.to_numeric(rows[column], errors="coerce")
    return rows


def _instrument_id_batches(instruments: list[str], max_workers: int) -> list[list[str]]:
    """Split instrument ids into one batch per worker for process-pool execution."""

    workers = max(1, min(max_workers, len(instruments)))
    batch_size = max(1, (len(instruments) + workers - 1) // workers)
    return [
        instruments[index : index + batch_size] for index in range(0, len(instruments), batch_size)
    ]


def _instrument_batches(groups: list[pd.DataFrame], max_workers: int) -> list[list[pd.DataFrame]]:
    """Split instruments into one batch per worker for process-pool execution."""

    workers = max(1, min(max_workers, len(groups)))
    batch_size = max(1, (len(groups) + workers - 1) // workers)
    return [groups[index : index + batch_size] for index in range(0, len(groups), batch_size)]


def _compute_instrument_batch_from_parquet_worker(
    args: tuple[str, list[str]],
) -> tuple[list[pd.DataFrame], dict[str, dict[str, object]]]:
    """Process-pool entrypoint: load instrument batches from scratch parquet."""

    scratch_path, instrument_ids = args
    if not instrument_ids:
        return [], {}
    frame = pd.read_parquet(scratch_path, filters=[("instrument", "in", instrument_ids)])
    if frame.empty:
        return [], {}
    frame["instrument"] = frame["instrument"].astype(str)
    ta_module = require("pandas_ta_classic", purpose="W3-B technical indicators")
    parts: list[pd.DataFrame] = []
    mappings: dict[str, dict[str, object]] = {}
    for _, group in frame.groupby("instrument", sort=False):
        part, group_mappings = _compute_one_instrument(group, ta_module)
        parts.append(part)
        mappings.update(group_mappings)
    return parts, mappings


def _compute_instrument_batch_worker(
    batch: list[pd.DataFrame],
) -> tuple[list[pd.DataFrame], dict[str, dict[str, object]]]:
    """Process-pool entrypoint: compute indicators for a batch of instruments."""

    ta_module = require("pandas_ta_classic", purpose="W3-B technical indicators")
    parts: list[pd.DataFrame] = []
    mappings: dict[str, dict[str, object]] = {}
    for group in batch:
        part, group_mappings = _compute_one_instrument(group, ta_module)
        parts.append(part)
        mappings.update(group_mappings)
    return parts, mappings


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.parent / f".{target.name}.tmp"
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, target)


def _compute_one_instrument(
    group: pd.DataFrame, ta_module: object
) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    out = group.loc[:, ["date", "instrument"]].copy()
    close = group["adj_close"].astype("float64")
    high = group["adj_high"].astype("float64")
    low = group["adj_low"].astype("float64")
    open_ = group["adj_open"].astype("float64")
    adj_volume = group["adj_volume"].astype("float64")
    raw_dollar_volume = group["raw_close"].astype("float64") * group["raw_volume"].astype("float64")
    mappings: dict[str, dict[str, object]] = {}

    out["rsi_14"], mappings["rsi_14"] = _series_call(ta_module, "rsi", close, length=14)
    out["roc_5"], mappings["roc_5"] = _series_call(ta_module, "roc", close, length=5)
    out["roc_10"], mappings["roc_10"] = _series_call(ta_module, "roc", close, length=10)
    out["roc_20"], mappings["roc_20"] = _series_call(ta_module, "roc", close, length=20)

    stoch, stoch_mapping = _frame_call(ta_module, "stoch", high, low, close, k=14, d=3, smooth_k=3)
    out["stoch_k_14_3"] = _select_frame_column(stoch, ("STOCHk", "STOCHK", "k"))
    out["stoch_d_14_3"] = _select_frame_column(stoch, ("STOCHd", "STOCHD", "d"))
    mappings["stoch_k_14_3"] = {**stoch_mapping, "normalized_column": "stoch_k_14_3"}
    mappings["stoch_d_14_3"] = {**stoch_mapping, "normalized_column": "stoch_d_14_3"}

    out["williams_r_14"], mappings["williams_r_14"] = _series_call(
        ta_module, "willr", high, low, close, length=14
    )
    out["cci_20"], mappings["cci_20"] = _series_call(ta_module, "cci", high, low, close, length=20)

    macd, macd_mapping = _frame_call(ta_module, "macd", close, fast=12, slow=26, signal=9)
    out["macd_12_26_9"] = _select_frame_column(macd, ("MACD_", "MACD"))
    out["macd_hist_12_26_9"] = _select_frame_column(macd, ("MACDh", "MACDH", "hist"))
    mappings["macd_12_26_9"] = {**macd_mapping, "normalized_column": "macd_12_26_9"}
    mappings["macd_hist_12_26_9"] = {**macd_mapping, "normalized_column": "macd_hist_12_26_9"}

    adx, adx_mapping = _frame_call(ta_module, "adx", high, low, close, length=14)
    out["adx_14"] = _select_frame_column(adx, ("ADX",))
    out["dmp_14"] = _select_frame_column(adx, ("DMP", "DIP"))
    out["dmn_14"] = _select_frame_column(adx, ("DMN", "DIN"))
    mappings["adx_14"] = {**adx_mapping, "normalized_column": "adx_14"}
    mappings["dmp_14"] = {**adx_mapping, "normalized_column": "dmp_14"}
    mappings["dmn_14"] = {**adx_mapping, "normalized_column": "dmn_14"}

    ema_20, mappings["ema_distance_20"] = _series_call(ta_module, "ema", close, length=20)
    ema_50, mappings["ema_distance_50"] = _series_call(ta_module, "ema", close, length=50)
    sma_20, sma20_mapping = _series_call(ta_module, "sma", close, length=20)
    sma_50, sma50_mapping = _series_call(ta_module, "sma", close, length=50)
    out["ema_distance_20"] = close / ema_20.replace(0.0, np.nan) - 1.0
    out["ema_distance_50"] = close / ema_50.replace(0.0, np.nan) - 1.0
    out["sma_cross_20_50"] = sma_20 / sma_50.replace(0.0, np.nan) - 1.0
    mappings["sma_cross_20_50"] = {
        "provider_function": "sma",
        "provider_columns": [sma20_mapping["provider_column"], sma50_mapping["provider_column"]],
    }

    out["atr_14"], mappings["atr_14"] = _series_call(ta_module, "atr", high, low, close, length=14)
    out["natr_14"], mappings["natr_14"] = _series_call(
        ta_module, "natr", high, low, close, length=14
    )
    bbands, bb_mapping = _frame_call(ta_module, "bbands", close, length=20, std=2)
    out["bb_percent_b_20_2"] = _select_frame_column(bbands, ("BBP", "percent"))
    out["bb_bandwidth_20_2"] = _select_frame_column(bbands, ("BBB", "bandwidth"))
    mappings["bb_percent_b_20_2"] = {**bb_mapping, "normalized_column": "bb_percent_b_20_2"}
    mappings["bb_bandwidth_20_2"] = {**bb_mapping, "normalized_column": "bb_bandwidth_20_2"}

    kc, _ = _frame_call(ta_module, "kc", high, low, close, length=20)
    out["keltner_width_20"] = _channel_width(kc, close)
    mappings["keltner_width_20"] = {
        "provider_function": "kc",
        "normalized_column": "keltner_width_20",
    }
    donchian, _ = _frame_call(ta_module, "donchian", high, low, lower_length=20, upper_length=20)
    out["donchian_position_20"] = _channel_position(donchian, close)
    mappings["donchian_position_20"] = {
        "provider_function": "donchian",
        "normalized_column": "donchian_position_20",
    }

    obv, mappings["obv_change_20"] = _series_call(ta_module, "obv", close, adj_volume)
    out["obv_change_20"] = obv.diff(20)
    out["mfi_14"], mappings["mfi_14"] = _series_call(
        ta_module, "mfi", high, low, close, adj_volume, length=14
    )
    out["volume_zscore_20"] = _rolling_zscore(adj_volume, 20)
    out["volume_ratio_20"] = (
        adj_volume / adj_volume.rolling(20, min_periods=5).mean().replace(0.0, np.nan) - 1.0
    )
    out["volume_dollar_zscore_20"] = _rolling_zscore(raw_dollar_volume, 20)
    out["dollar_volume_ratio_20"] = (
        raw_dollar_volume / raw_dollar_volume.rolling(20, min_periods=5).mean().replace(0.0, np.nan)
        - 1.0
    )
    mappings["volume_zscore_20"] = {
        "provider_function": "derived",
        "source_columns": ["adj_volume"],
    }
    mappings["volume_ratio_20"] = {"provider_function": "derived", "source_columns": ["adj_volume"]}
    mappings["volume_dollar_zscore_20"] = {
        "provider_function": "derived",
        "source_columns": ["raw_close", "raw_volume"],
    }
    mappings["dollar_volume_ratio_20"] = {
        "provider_function": "derived",
        "source_columns": ["raw_close", "raw_volume"],
    }

    rolling_max = close.rolling(60, min_periods=5).max()
    out["drawdown_60"] = close / rolling_max.replace(0.0, np.nan) - 1.0
    out["realized_volatility_20"] = close.pct_change().rolling(20, min_periods=5).std()
    out["range_ratio_10"] = (
        ((high - low) / close.replace(0.0, np.nan)).rolling(10, min_periods=3).mean()
    )
    mappings["drawdown_60"] = {"provider_function": "derived", "source_columns": ["adj_close"]}
    mappings["realized_volatility_20"] = {
        "provider_function": "derived",
        "source_columns": ["adj_close"],
    }
    mappings["range_ratio_10"] = {
        "provider_function": "derived",
        "source_columns": ["adj_high", "adj_low", "adj_close"],
    }
    _ = open_
    return out, mappings


def _series_call(
    ta_module: object, function_name: str, *args: object, **kwargs: object
) -> tuple[pd.Series[Any], dict[str, object]]:
    try:
        raw = _call_provider(ta_module, function_name, *args, **kwargs)
        series = _coerce_series(raw, *args)
        return series, {
            "provider_function": function_name,
            "provider_column": str(series.name or function_name),
        }
    except Exception as exc:  # noqa: BLE001
        # pandas_ta_classic can raise provider-internal errors on short histories.
        index = _reference_index(*args)
        return pd.Series(np.nan, index=index, dtype="float64"), {
            "provider_function": function_name,
            "provider_column": function_name,
            "provider_failed": True,
            "provider_error": type(exc).__name__,
        }


def _frame_call(
    ta_module: object, function_name: str, *args: object, **kwargs: object
) -> tuple[pd.DataFrame, dict[str, object]]:
    try:
        raw = _call_provider(ta_module, function_name, *args, **kwargs)
        frame = _coerce_frame(raw, *args)
        return frame, {
            "provider_function": function_name,
            "provider_columns": list(frame.columns.astype(str)),
        }
    except Exception as exc:  # noqa: BLE001
        # pandas_ta_classic can raise provider-internal errors on short histories.
        index = _reference_index(*args)
        return pd.DataFrame(index=index), {
            "provider_function": function_name,
            "provider_columns": [],
            "provider_failed": True,
            "provider_error": type(exc).__name__,
        }


def _try_frame_call(
    ta_module: object, function_name: str, *args: object, **kwargs: object
) -> pd.DataFrame:
    frame, _ = _frame_call(ta_module, function_name, *args, **kwargs)
    return frame


def _call_provider(
    ta_module: object, function_name: str, *args: object, **kwargs: object
) -> object:
    function = getattr(ta_module, function_name)
    return cast(Callable[..., object], function)(*args, **kwargs)


def _reference_index(*args: object) -> pd.Index[Any]:
    for arg in args:
        if isinstance(arg, pd.Series):
            return arg.index
        if isinstance(arg, pd.DataFrame):
            return arg.index
    return pd.RangeIndex(0)


def _coerce_series(raw: object, *index_args: object) -> pd.Series[Any]:
    if raw is None:
        return pd.Series(np.nan, index=_reference_index(*index_args), dtype="float64")
    if isinstance(raw, pd.Series):
        return pd.to_numeric(raw, errors="coerce")
    if isinstance(raw, pd.DataFrame):
        return pd.to_numeric(raw.iloc[:, 0], errors="coerce")
    return pd.Series(raw, dtype="float64")


def _coerce_frame(raw: object, *index_args: object) -> pd.DataFrame:
    if raw is None:
        return pd.DataFrame(index=_reference_index(*index_args))
    if isinstance(raw, pd.DataFrame):
        frame: pd.DataFrame = raw.apply(pd.to_numeric, errors="coerce")
        return frame
    if isinstance(raw, pd.Series):
        series_frame: pd.DataFrame = raw.to_frame().apply(pd.to_numeric, errors="coerce")
        return series_frame
    fallback: pd.DataFrame = pd.DataFrame(cast(Any, raw)).apply(pd.to_numeric, errors="coerce")
    return fallback


def _select_frame_column(frame: pd.DataFrame, prefixes: Iterable[str]) -> pd.Series[Any]:
    if frame.empty:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    normalized = [str(column) for column in frame.columns]
    lowered = [column.lower() for column in normalized]
    for prefix in prefixes:
        low = prefix.lower()
        for idx, column in enumerate(lowered):
            if column.startswith(low) or low in column:
                return pd.to_numeric(frame.iloc[:, idx], errors="coerce")
    return pd.to_numeric(frame.iloc[:, 0], errors="coerce")


def _channel_width(frame: pd.DataFrame, close: pd.Series[Any]) -> pd.Series[Any]:
    if frame.shape[1] >= 3:
        lower = pd.to_numeric(frame.iloc[:, 0], errors="coerce")
        middle = pd.to_numeric(frame.iloc[:, 1], errors="coerce").replace(0.0, np.nan)
        upper = pd.to_numeric(frame.iloc[:, 2], errors="coerce")
        return (upper - lower) / middle
    ema = close.ewm(span=20, adjust=False, min_periods=5).mean()
    atr_proxy = close.diff().abs().rolling(20, min_periods=5).mean()
    return (4.0 * atr_proxy) / ema.replace(0.0, np.nan)


def _channel_position(frame: pd.DataFrame, close: pd.Series[Any]) -> pd.Series[Any]:
    if frame.shape[1] >= 3:
        lower = pd.to_numeric(frame.iloc[:, 0], errors="coerce")
        upper = pd.to_numeric(frame.iloc[:, 2], errors="coerce")
    else:
        lower = close.rolling(20, min_periods=5).min()
        upper = close.rolling(20, min_periods=5).max()
    return (close - lower) / (upper - lower).replace(0.0, np.nan)


def _rolling_zscore(values: pd.Series[Any], window: int) -> pd.Series[Any]:
    mean = values.rolling(window, min_periods=5).mean()
    std = values.rolling(window, min_periods=5).std().replace(0.0, np.nan)
    return (values - mean) / std

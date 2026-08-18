"""SIP adjusted-panel loader and base-panel builder for the indicator pipeline.

Relocated from the retired W2-v3M allocator-benchmark lane; the active flow uses only the
loader, the base-panel builder, and the supervision hygiene defined here.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Final, Literal

import numpy as np
import pandas as pd

from pysrc.core.runtime.optional_imports import optional_import

SIP_ADJUSTED_PANEL_SOURCE = Path("data/massive/us_stocks_sip/adjusted_day_panel_v1")
SIP_PANEL_SIGNAL_IDS: tuple[str, ...] = (
    "momentum_5d",
    "momentum_20d",
    "momentum_60d",
    "mean_reversion_5d",
    "trend_strength",
    "realized_volatility_20d",
    "volatility_breakout",
    "range_compression",
    "volume_proxy",
    "drawdown_60d",
)
_REQUIRED_MINIMUM_COLUMNS: tuple[str, ...] = (
    "raw_close",
    "raw_volume",
    "adj_close",
    "corporate_action_flag",
    "extreme_raw_return_flag",
    "extreme_adjusted_return_flag",
)
_PANEL_FLOAT32_COLUMNS: tuple[str, ...] = (
    "raw_close",
    "raw_volume",
    "adj_close",
    "adjusted_return_1d",
    "total_price_adjustment_factor",
    "split_adjustment_factor",
    "dividend_adjustment_factor",
    "adj_high",
    "adj_low",
)
# SIP test symbols and supervision caps.
_EXCLUDED_TEST_INSTRUMENTS: Final[frozenset[str]] = frozenset({"ZTEST", "ZVZZT", "ZWZZT", "ZJZZT"})
_SUPERVISION_ABS_RETURN_CAP: Final[float] = 1.0
_MIN_ADJ_CLOSE: Final[float] = 1e-6
_ADJ_RAW_RATIO_MIN: Final[float] = 0.01
_ADJ_RAW_RATIO_MAX: Final[float] = 100.0


@dataclass(frozen=True, slots=True)
class SipPanelConfig:
    horizon: int = 5
    base_cost: float = 0.0005
    liquidity_floor: float = 5_000_000.0
    top_k: int = 2
    max_gross_exposure: float = 1.0
    max_single_instrument_weight: float = 0.5
    score_threshold: float = 0.0
    score_higher_is_better: bool = True
    corporate_action_event_window_days: int = 1
    save_signal_rows: bool = True
    use_xgboost: bool = True
    xgboost_train_row_cap: int = 50_000
    xgboost_predict_batch_rows: int = 250_000
    exclude_earliest_train_calendar_year: bool = False
    panel_coalesce_float32: bool = False
    release_input_panel_after_surfaces: bool = False
    repair_reload_panel_after_surface_prep: bool = False
    w3_audit_spill_intermediates: bool = False
    w3_audit_skip_shuffled_label_rerun: bool = False
    w3_audit_force_heavy_surface_b_checks: bool = False
    workers: int = 1
    xgboost_n_jobs: int = 1
    seed: int = 6203
    timestamp_utc: str = "2026-05-09T00:00:00Z"


def _load_one_adjusted_panel_partition(path: Path) -> pd.DataFrame:
    part = pd.read_parquet(path)
    if "date" not in part.columns and "Date" not in part.columns:
        part = part.copy()
        part["date"] = path.stem
    return _normalize_adjusted_panel_columns(part)


def _coalesce_panel_float32_inplace(rows: pd.DataFrame) -> None:
    """Narrow wide price/volume columns in-place to reduce peak RAM (small numeric drift vs float64).

    Finite values outside float32 range are clipped so ``astype(float32)`` does not emit overflow
    warnings (or produce infinities) when upstream parquet has outliers or non-finite values.
    """

    finfo = np.finfo(np.float32)
    lower = float(finfo.min)
    upper = float(finfo.max)
    for column in _PANEL_FLOAT32_COLUMNS:
        if column not in rows.columns:
            continue
        values = pd.to_numeric(rows[column], errors="coerce").to_numpy(dtype=np.float64, copy=True)
        np.clip(values, lower, upper, out=values)
        rows[column] = values.astype(np.float32, copy=False)


def _load_adjusted_panel_partition_stacked(
    path: Path, *, panel_coalesce_float32: bool
) -> pd.DataFrame:
    part = _load_one_adjusted_panel_partition(path)
    if panel_coalesce_float32:
        _coalesce_panel_float32_inplace(part)
    return part


def _load_adjusted_panel_via_pandas(
    files: list[Path],
    *,
    panel_coalesce_float32: bool,
    workers: int,
) -> pd.DataFrame:
    max_workers = max(1, int(workers))
    loader = partial(
        _load_adjusted_panel_partition_stacked, panel_coalesce_float32=bool(panel_coalesce_float32)
    )
    if max_workers == 1 or len(files) == 1:
        it = iter(files)
        first_path = next(it, None)
        if first_path is None:
            raise ValueError("W2-v3M adjusted panel source contains no parquet files.")
        out = loader(first_path)
        for path in it:
            out = pd.concat([out, loader(path)], axis=0, ignore_index=True)
        return out
    indexed_parts: list[tuple[int, pd.DataFrame]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(loader, path): index for index, path in enumerate(files)}
        for future in as_completed(futures):
            indexed_parts.append((futures[future], future.result()))
    parts_sorted = sorted(indexed_parts, key=lambda item: item[0])
    del indexed_parts
    out = parts_sorted[0][1]
    for _, part in parts_sorted[1:]:
        out = pd.concat([out, part], axis=0, ignore_index=True)
    return out


def _load_adjusted_panel_via_polars(
    root: Path,
    *,
    panel_coalesce_float32: bool,
) -> pd.DataFrame:
    pl = optional_import("polars")
    if pl is None:
        raise RuntimeError("Polars backend requested but polars is not installed.")
    scan_glob = str(root / "year=*" / "month=*" / "*.parquet")
    lazy = pl.scan_parquet(scan_glob)
    columns = list(lazy.collect_schema().names())
    if "date" not in columns and "Date" not in columns:
        raise ValueError("W2-v3M polars loader requires a date/Date column in parquet schema.")
    frame = lazy.collect().to_pandas()
    out = _normalize_adjusted_panel_columns(frame)
    if panel_coalesce_float32:
        _coalesce_panel_float32_inplace(out)
    return out


def load_sip_adjusted_panel(
    source_path: Path,
    *,
    workers: int = 1,
    panel_coalesce_float32: bool = False,
    loader_backend: Literal["polars", "pandas"] = "polars",
) -> pd.DataFrame:
    """Load year/month/day parquet partitions without mutating source files."""

    root = Path(source_path)
    files = sorted(root.glob("year=*/month=*/*.parquet"))
    if not files:
        raise ValueError(f"W2-v3M adjusted panel source contains no parquet files: {root}")
    backend = str(loader_backend).lower()
    if backend == "polars":
        try:
            return _load_adjusted_panel_via_polars(
                root,
                panel_coalesce_float32=bool(panel_coalesce_float32),
            )
        except Exception:
            # Fail closed to pandas fallback so EC2 runs can continue while preserving behavior.
            return _load_adjusted_panel_via_pandas(
                files,
                panel_coalesce_float32=bool(panel_coalesce_float32),
                workers=max(1, int(workers)),
            )
    if backend == "pandas":
        return _load_adjusted_panel_via_pandas(
            files,
            panel_coalesce_float32=bool(panel_coalesce_float32),
            workers=max(1, int(workers)),
        )
    raise ValueError(f"Unsupported loader_backend: {loader_backend!r}")


def build_sip_base_panel(
    panel: pd.DataFrame,
    config: SipPanelConfig | None = None,
    *,
    copy: bool = True,
) -> pd.DataFrame:
    """Prepare the shared W2-v3M base panel used by all surfaces."""

    resolved = config or SipPanelConfig()
    base = _prepare_base_panel(
        panel,
        horizon=resolved.horizon,
        require_forward=True,
        corporate_action_event_window_days=int(resolved.corporate_action_event_window_days),
        panel_coalesce_float32=bool(resolved.panel_coalesce_float32),
        copy=copy,
    )
    return _attach_regime_and_signal_features(base, copy=copy)


def _normalize_adjusted_panel_columns(panel: pd.DataFrame) -> pd.DataFrame:
    rows = panel.copy()
    rename: dict[str, str] = {}
    if "date" not in rows.columns:
        for candidate in ("Date", "timestamp", "bar_date"):
            if candidate in rows.columns:
                rename[candidate] = "date"
                break
    if "instrument" not in rows.columns:
        for candidate in ("ticker", "Ticker", "symbol", "Symbol"):
            if candidate in rows.columns:
                rename[candidate] = "instrument"
                break
    if rename:
        rows = rows.rename(columns=rename)
    if "date" not in rows.columns or "instrument" not in rows.columns:
        raise ValueError("W2-v3M adjusted panel requires date and instrument/ticker columns")
    return rows


def _missing_required_columns(rows: pd.DataFrame) -> list[str]:
    missing = [column for column in _REQUIRED_MINIMUM_COLUMNS if column not in rows.columns]
    if "date" not in rows.columns:
        missing.append("date")
    if "instrument" not in rows.columns:
        missing.append("instrument")
    return missing


def _is_excluded_test_instrument(instruments: pd.Series) -> pd.Series:
    return instruments.astype(str).isin(_EXCLUDED_TEST_INSTRUMENTS)


def _supervision_hygiene_mask(rows: pd.DataFrame) -> pd.Series:
    """Rows whose forward supervision must be treated as missing (SIP adj_close / return spikes)."""

    adj_close = pd.to_numeric(rows["adj_close"], errors="coerce")
    raw_close = pd.to_numeric(rows["raw_close"], errors="coerce").replace(0.0, np.nan)
    forward = pd.to_numeric(rows["forward_return_horizon"], errors="coerce")
    adj_ret = pd.to_numeric(rows["adjusted_return_1d"], errors="coerce")
    ratio = adj_close / raw_close
    unusable = (
        adj_close.le(_MIN_ADJ_CLOSE)
        | ratio.lt(_ADJ_RAW_RATIO_MIN)
        | ratio.gt(_ADJ_RAW_RATIO_MAX)
        | forward.abs().gt(_SUPERVISION_ABS_RETURN_CAP)
        | adj_ret.abs().gt(_SUPERVISION_ABS_RETURN_CAP)
    )
    return unusable.fillna(True)


def _prepare_base_panel(
    panel: pd.DataFrame,
    *,
    horizon: int,
    require_forward: bool,
    corporate_action_event_window_days: int = 1,
    panel_coalesce_float32: bool = False,
    copy: bool = True,
) -> pd.DataFrame:
    rows = _normalize_adjusted_panel_columns(panel)
    missing = _missing_required_columns(rows)
    if missing:
        raise ValueError(f"W2-v3M adjusted panel missing required columns: {missing}")
    if copy:
        rows = rows.copy()
    rows["date"] = pd.to_datetime(rows["date"], utc=False, errors="coerce")
    rows["instrument"] = rows["instrument"].astype(str)
    test_instrument_mask = _is_excluded_test_instrument(rows["instrument"])
    rows.attrs["w2_excluded_test_instrument_row_count"] = int(test_instrument_mask.sum())
    if test_instrument_mask.any():
        rows = rows.loc[~test_instrument_mask].reset_index(drop=True)
    for column in ("raw_close", "raw_volume", "adj_close"):
        rows[column] = pd.to_numeric(rows[column], errors="coerce")
    for column in (
        "corporate_action_flag",
        "extreme_raw_return_flag",
        "extreme_adjusted_return_flag",
    ):
        rows[column] = rows[column].fillna(False).astype(bool)
    rows = rows.sort_values(["instrument", "date"], kind="mergesort").reset_index(drop=True)
    rows = _attach_corporate_action_semantic_columns(
        rows,
        event_window_days=int(corporate_action_event_window_days),
        copy=copy,
    )
    duplicate_before = int(rows.duplicated(["date", "instrument"], keep=False).sum())
    rows = _dedupe_canonical_date_instrument(rows)
    duplicate_after = int(rows.duplicated(["date", "instrument"], keep=False).sum())
    rows.attrs["w2_duplicate_date_instrument_count_before_dedupe"] = duplicate_before
    rows.attrs["w2_duplicate_date_instrument_count_after_dedupe"] = duplicate_after
    rows.attrs["w2_canonical_dedupe_policy"] = _CANONICAL_DEDUPE_POLICY
    # Dedupe sorts by corporate-action flags first; restore chronological order before shift-based labels.
    rows = rows.sort_values(["instrument", "date"], kind="mergesort").reset_index(drop=True)
    rows["dollar_volume"] = rows["raw_close"] * rows["raw_volume"]
    rows["forward_return_horizon"] = (
        rows.groupby("instrument", sort=False)["adj_close"].shift(-int(horizon)) / rows["adj_close"]
        - 1.0
    )
    if "adjusted_return_1d" not in rows.columns:
        rows["adjusted_return_1d"] = rows.groupby("instrument", sort=False)[
            "adj_close"
        ].pct_change()
    else:
        rows["adjusted_return_1d"] = pd.to_numeric(rows["adjusted_return_1d"], errors="coerce")
    hygiene_mask = _supervision_hygiene_mask(rows)
    rows.attrs["w2_supervision_hygiene_nullified_row_count"] = int(hygiene_mask.sum())
    if hygiene_mask.any():
        rows.loc[hygiene_mask, "forward_return_horizon"] = np.nan
    rows["rolling_20d_median_dollar_volume"] = rows.groupby("instrument", sort=False)[
        "dollar_volume"
    ].transform(lambda value: value.shift(1).rolling(20, min_periods=1).median())
    rows["cost_estimate"] = 0.0005
    rows["price_bucket"] = pd.cut(
        rows["raw_close"],
        bins=[-math.inf, 5.0, 20.0, 100.0, math.inf],
        labels=["sub_5", "5_to_20", "20_to_100", "100_plus"],
    ).astype(str)
    rows["liquidity_bucket"] = pd.cut(
        rows["dollar_volume"],
        bins=[-math.inf, 1_000_000.0, 5_000_000.0, 25_000_000.0, math.inf],
        labels=["lt_1m", "1m_to_5m", "5m_to_25m", "25m_plus"],
    ).astype(str)
    if require_forward:
        rows = rows.loc[rows["forward_return_horizon"].notna()]
        if copy:
            rows = rows.copy()
    if panel_coalesce_float32:
        _coalesce_panel_float32_inplace(rows)
    return rows


def _trading_event_window_mask(exact_block: np.ndarray, *, window_days: int) -> np.ndarray:
    """±N trading-day envelope around exact corporate-action rows (single-instrument contiguous block)."""

    length = int(len(exact_block))
    if length == 0:
        return np.zeros(0, dtype=bool)
    if window_days <= 0:
        exact_mask: np.ndarray = np.asarray(exact_block, dtype=bool)
        return exact_mask
    out: np.ndarray = np.zeros(length, dtype=bool)
    span = int(window_days)
    for position in range(length):
        left = max(0, position - span)
        right = min(length, position + span + 1)
        out[position] = bool(exact_block[left:right].any())
    return out


def _attach_corporate_action_semantic_columns(
    rows: pd.DataFrame,
    *,
    event_window_days: int,
    copy: bool = True,
) -> pd.DataFrame:
    """Derive explicit corporate-action semantics for Surface B filtering (W3-A repair)."""

    out: pd.DataFrame = rows.copy() if copy else rows
    row_date = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    exact = pd.Series(False, index=out.index)
    if "split_execution_date" in out.columns:
        split_d = pd.to_datetime(out["split_execution_date"], errors="coerce").dt.normalize()
        exact = exact | (row_date.eq(split_d) & split_d.notna())
    if "dividend_ex_dividend_date" in out.columns:
        div_d = pd.to_datetime(out["dividend_ex_dividend_date"], errors="coerce").dt.normalize()
        exact = exact | (row_date.eq(div_d) & div_d.notna())

    active = pd.Series(False, index=out.index)
    if "total_price_adjustment_factor" in out.columns:
        total = pd.to_numeric(out["total_price_adjustment_factor"], errors="coerce")
        active = total.notna() & (total.sub(1.0).abs() > 1e-8)
    elif {"split_adjustment_factor", "dividend_adjustment_factor"}.issubset(out.columns):
        split_f = pd.to_numeric(out["split_adjustment_factor"], errors="coerce").fillna(1.0)
        div_f = pd.to_numeric(out["dividend_adjustment_factor"], errors="coerce").fillna(1.0)
        active = split_f.sub(1.0).abs().gt(1e-8) | div_f.sub(1.0).abs().gt(1e-8)
    elif "corporate_action_flag" in out.columns:
        active = out["corporate_action_flag"].astype(bool)

    window_days = max(0, int(event_window_days))
    evt = exact.astype(bool).to_numpy()
    instruments = out["instrument"].astype(str).to_numpy()
    window_flag: np.ndarray = np.zeros(len(out), dtype=bool)
    start = 0
    while start < len(out):
        end = start + 1
        while end < len(out) and instruments[end] == instruments[start]:
            end += 1
        block = evt[start:end]
        if window_days == 0:
            window_flag[start:end] = block
        else:
            window_flag[start:end] = _trading_event_window_mask(block, window_days=window_days)
        start = end

    out["corporate_action_active_adjustment_flag"] = active.fillna(False).astype(bool)
    out["corporate_action_exact_event_flag"] = exact.fillna(False).astype(bool)
    out["corporate_action_event_window_flag"] = window_flag
    return out


_CANONICAL_DEDUPE_POLICY: Final[str] = (
    "corporate_action_exact_event_flag asc, corporate_action_event_window_flag asc, "
    "corporate_action_flag asc, instrument asc, date asc, mergesort stable keep=first"
)


def _dedupe_canonical_date_instrument(rows: pd.DataFrame) -> pd.DataFrame:
    """Keep one deterministic row per date x instrument before shift-based label math."""

    if rows.empty:
        return rows

    out = rows.copy()
    for column in (
        "corporate_action_exact_event_flag",
        "corporate_action_event_window_flag",
        "corporate_action_flag",
    ):
        if column not in out.columns:
            out[column] = False
        else:
            out[column] = out[column].fillna(False).astype(bool)

    out = out.sort_values(
        [
            "corporate_action_exact_event_flag",
            "corporate_action_event_window_flag",
            "corporate_action_flag",
            "instrument",
            "date",
        ],
        kind="mergesort",
        ascending=[True, True, True, True, True],
    )
    return out.drop_duplicates(["date", "instrument"], keep="first").reset_index(drop=True)


def _attach_regime_and_signal_features(rows: pd.DataFrame, *, copy: bool = True) -> pd.DataFrame:
    out = rows.copy() if copy else rows
    grouped_close = out.groupby("instrument", sort=False)["adj_close"]
    grouped_ret = out.groupby("instrument", sort=False)["adjusted_return_1d"]
    out["momentum_5d"] = grouped_close.shift(1) / grouped_close.shift(6) - 1.0
    out["momentum_20d"] = grouped_close.shift(1) / grouped_close.shift(21) - 1.0
    out["momentum_60d"] = grouped_close.shift(1) / grouped_close.shift(61) - 1.0
    out["mean_reversion_5d"] = -out["momentum_5d"]
    out["realized_volatility_20d"] = grouped_ret.transform(
        lambda value: value.shift(1).rolling(20, min_periods=5).std()
    )
    out["trend_strength"] = out["momentum_20d"] / out["realized_volatility_20d"].replace(
        0.0, np.nan
    )
    out["volatility_breakout"] = (
        out["realized_volatility_20d"]
        / (grouped_ret.transform(lambda value: value.shift(1).rolling(60, min_periods=20).std()))
        - 1.0
    )
    if {"adj_high", "adj_low"}.issubset(out.columns):
        out["adj_high"] = pd.to_numeric(out["adj_high"], errors="coerce")
        out["adj_low"] = pd.to_numeric(out["adj_low"], errors="coerce")
        raw_range = (out["adj_high"] - out["adj_low"]) / out["adj_close"]
        out["range_compression"] = -raw_range.groupby(out["instrument"], sort=False).transform(
            lambda value: value.rolling(10, min_periods=3).mean().shift(1)
        )
    out["volume_proxy"] = out.groupby("instrument", sort=False)["raw_volume"].transform(
        lambda value: value.shift(1) / value.shift(1).rolling(20, min_periods=5).mean() - 1.0
    )
    out["drawdown_60d"] = (
        grouped_close.shift(1)
        / grouped_close.transform(lambda value: value.shift(1).rolling(60, min_periods=5).max())
        - 1.0
    )
    trailing_20d_return = grouped_close.shift(1) / grouped_close.shift(21) - 1.0
    vol_median = out.groupby("instrument", sort=False)["realized_volatility_20d"].transform(
        lambda value: value.shift(1).expanding(min_periods=20).median()
    )
    trend = np.where(trailing_20d_return.gt(0.0), "trend_bull", "trend_bear")
    vol = np.where(out["realized_volatility_20d"].gt(vol_median), "vol_high", "vol_low")
    out["regime_id"] = (
        pd.Series(trend, index=out.index).astype(str)
        + "__"
        + pd.Series(vol, index=out.index).astype(str)
    )
    for column in SIP_PANEL_SIGNAL_IDS:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
    return out


def _valid_price_volume_mask(rows: pd.DataFrame) -> pd.Series:
    return rows["raw_close"].gt(0.0) & rows["raw_volume"].gt(0.0) & rows["adj_close"].gt(0.0)

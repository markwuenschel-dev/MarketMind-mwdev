"""
Data fixtures: sample_features, price DataFrames, OHLCV path builders.

Extracted from conftest; used by preprocessor, graph pipeline, backtesting, and leakage tests.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pandas as pd
import pytest

if TYPE_CHECKING:
    from pathlib import Path

try:
    import polars as pl
except (ModuleNotFoundError, ImportError):
    pl = None


def build_sample_features() -> pd.DataFrame:
    """
    Produce a small, deterministic feature frame shaped like a time-series.
    Uses TEST_SAMPLE_FEATURE_ROWS for row count (default 8, min 5).
    """
    rows_env = os.getenv("TEST_SAMPLE_FEATURE_ROWS")
    default_n = 8
    try:
        n_rows = max(5, int(rows_env)) if rows_env else default_n
    except ValueError:
        n_rows = default_n

    ts_index = pd.date_range(
        "2025-01-01 09:30:00",
        periods=n_rows,
        freq="min",
    )
    price_vals = [100.0 + i for i in range(n_rows)]
    volume_vals = [1000 + i * 10 for i in range(n_rows)]
    feat_a_vals = [0.1 * (i + 1) for i in range(n_rows)]
    feat_b_vals = [1.0 + 0.5 * i for i in range(n_rows)]
    vol_vals = [0.05 + 0.01 * i for i in range(n_rows)]
    market_open_vals = [True] * n_rows

    df = pd.DataFrame(
        {
            "price": price_vals,
            "volume": volume_vals,
            "feature_a": feat_a_vals,
            "feature_b": feat_b_vals,
            "volatility": vol_vals,
            "market_open": market_open_vals,
        },
        index=ts_index,
    )
    df = df.astype(
        {
            "price": "float64",
            "volume": "int64",
            "feature_a": "float64",
            "feature_b": "float64",
            "volatility": "float64",
            "market_open": "bool",
        }
    )
    if df.empty:
        raise AssertionError("sample_features fixture built an empty DataFrame")
    return df


@pytest.fixture
def sample_features() -> pd.DataFrame:
    """Per-test scope to avoid mutation bleed between param cases."""
    return build_sample_features()


@pytest.fixture
def small_prices_df() -> pd.DataFrame:
    """Small price DataFrame for loader/preprocessor tests."""
    ts = pd.date_range("2021-01-01", periods=10, freq="D")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "symbol": ["TEST"] * len(ts),
            "price": [100, 101, 99, 102, 100, 98, 103, 101, 99, 100],
        }
    )


def _ensure_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add close/open/high/low if missing (for preprocessing expectations)."""
    if "price" not in df.columns or "close" in df.columns:
        return df
    df = df.copy()
    df["close"] = df["price"]
    if "open" not in df.columns:
        df["open"] = df["price"]
    if "high" not in df.columns:
        df["high"] = df["price"] * 1.02
    if "low" not in df.columns:
        df["low"] = df["price"] * 0.98
    return df


@pytest.fixture
def prices_small_path(tmp_path: Path, small_prices_df: pd.DataFrame) -> str:
    """Small test dataset with OHLC columns for preprocessing."""
    df_to_write = _ensure_ohlcv_columns(small_prices_df)
    p = tmp_path / "prices_small.csv"
    df_to_write.to_csv(p, index=False)
    return str(p)


@pytest.fixture
def prices_small_v2_path(tmp_path: Path, small_prices_df: pd.DataFrame) -> str:
    """Variation of prices_small_path (v2)."""
    df_to_write = _ensure_ohlcv_columns(small_prices_df)
    p = tmp_path / "prices_small_v2.csv"
    df_to_write.to_csv(p, index=False)
    return str(p)


@pytest.fixture
def df_prices(small_prices_df: pd.DataFrame) -> pd.DataFrame:
    """Polars or pandas DF depending on availability."""
    if pl and isinstance(small_prices_df, pd.DataFrame):
        return pl.from_pandas(small_prices_df)
    return small_prices_df


@pytest.fixture
def prices_small_pl(small_prices_df: pd.DataFrame):
    """Polars DataFrame; skips if polars not installed."""
    if pl is None:
        pytest.skip("polars not installed")
    if isinstance(small_prices_df, pl.DataFrame):
        return small_prices_df
    return pl.from_pandas(small_prices_df)


@pytest.fixture
def prices_small_v2_pl(prices_small_pl):
    """Cloned Polars DataFrame (v2)."""
    return prices_small_pl.clone()

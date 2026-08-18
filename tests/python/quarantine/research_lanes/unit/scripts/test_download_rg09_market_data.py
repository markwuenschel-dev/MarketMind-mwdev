"""Unit tests for scripts/download_rg09_market_data.pysrc."""

from __future__ import annotations

import pandas as pd
import pytest
from scripts.download_rg09_market_data import (
    REQUIRED_ROWS,
    normalize_close_frame,
    ticker_safe_filename,
)


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_ticker_safe_filename_maps_es_futures() -> None:
    assert ticker_safe_filename("ES=F") == "ES_F.parquet"


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_normalize_close_frame_builds_sorted_unique_dates() -> None:
    idx = pd.date_range("2020-01-01", periods=REQUIRED_ROWS + 1, freq="B", tz="UTC")
    raw = pd.DataFrame({"Close": range(1, len(idx) + 1)}, index=idx)
    out = normalize_close_frame(raw, ticker="TEST")
    assert list(out.columns) == ["date", "close"]
    assert len(out) == REQUIRED_ROWS + 1
    assert out["close"].dtype == "float64"
    assert not out["date"].duplicated().any()


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_normalize_close_frame_rejects_too_few_rows() -> None:
    idx = pd.date_range("2020-01-01", periods=50, freq="B", tz="UTC")
    raw = pd.DataFrame({"Close": range(1, len(idx) + 1)}, index=idx)
    with pytest.raises(ValueError, match="insufficient rows"):
        normalize_close_frame(raw, ticker="SHORT")


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_normalize_close_frame_multiindex_columns() -> None:
    idx = pd.date_range("2020-01-01", periods=REQUIRED_ROWS + 1, freq="B", tz="UTC")
    cols = pd.MultiIndex.from_tuples([("TEST", "Close")])
    raw = pd.DataFrame([[float(i)] for i in range(1, len(idx) + 1)], index=idx, columns=cols)
    out = normalize_close_frame(raw, ticker="TEST")
    assert len(out) == REQUIRED_ROWS + 1

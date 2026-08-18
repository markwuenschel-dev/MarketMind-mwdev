"""Tests for preprocessor core functions."""

import polars as pl
import pytest

from pysrc.preprocessor.core import add_returns, add_rsi, add_sma


@pytest.fixture
def sample_ohlcv() -> pl.DataFrame:
    """Create sample OHLCV data for testing."""
    return pl.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
            "open": [100.0, 101.0, 102.0, 101.5, 103.0],
            "high": [101.0, 102.5, 103.0, 102.0, 104.0],
            "low": [99.0, 100.5, 101.0, 100.0, 102.0],
            "close": [100.5, 102.0, 101.5, 101.0, 103.5],
            "volume": [1000, 1100, 900, 1200, 1050],
        }
    )


class TestAddReturns:
    def test_creates_returns_column(self, sample_ohlcv):
        result = add_returns(sample_ohlcv)
        assert "returns" in result.columns

    def test_preserves_row_count(self, sample_ohlcv):
        result = add_returns(sample_ohlcv)
        assert result.height == sample_ohlcv.height

    def test_first_return_is_null(self, sample_ohlcv):
        result = add_returns(sample_ohlcv)
        assert result["returns"][0] is None

    def test_calculates_correct_return(self, sample_ohlcv):
        result = add_returns(sample_ohlcv)
        # Day 2: (102.0 - 100.5) / 100.5 = 0.01492...
        expected = (102.0 - 100.5) / 100.5
        assert abs(result["returns"][1] - expected) < 0.0001


class TestAddSma:
    def test_creates_sma_column(self, sample_ohlcv):
        result = add_sma(sample_ohlcv, window=3)
        assert "sma_3" in result.columns

    def test_preserves_row_count(self, sample_ohlcv):
        result = add_sma(sample_ohlcv, window=3)
        assert result.height == sample_ohlcv.height

    def test_sma_uses_correct_window(self, sample_ohlcv):
        result = add_sma(sample_ohlcv, window=3)
        # Row 3 (index 2): average of 100.5, 102.0, 101.5 = 101.333...
        expected = (100.5 + 102.0 + 101.5) / 3
        assert abs(result["sma_3"][2] - expected) < 0.001


class TestAddRsi:
    def test_creates_rsi_column(self, sample_ohlcv):
        result = add_rsi(sample_ohlcv, window=3)
        assert "rsi_3" in result.columns

    def test_rsi_bounded_when_valid(self):
        # Need more data for meaningful RSI
        df = pl.DataFrame({"close": [100.0 + i * 0.5 for i in range(30)]})
        result = add_rsi(df, window=14)
        valid_rsi = result["rsi_14"].drop_nulls()

        # RSI should be between 0 and 100
        assert valid_rsi.min() >= 0 or valid_rsi.is_empty()
        assert valid_rsi.max() <= 100 or valid_rsi.is_empty()

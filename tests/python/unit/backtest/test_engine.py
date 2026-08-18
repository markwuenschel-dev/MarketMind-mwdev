"""Tests for backtest engine."""

import polars as pl
import pytest

from pysrc.backtesting.engines.vectorized.engine import (
    BacktestResult,
    run_backtest,
)
from pysrc.backtesting.engines.vectorized.engine import (
    _add_signals as add_signals,
)


@pytest.fixture
def preprocessed_data() -> pl.DataFrame:
    """Simulated preprocessed data with SMAs and returns."""
    return pl.DataFrame(
        {
            "date": [f"2024-01-{i:02d}" for i in range(1, 21)],
            "close": [100.0 + i * 0.5 for i in range(20)],
            "returns": [
                0.005,
                -0.003,
                0.008,
                -0.002,
                0.006,
                0.004,
                -0.001,
                0.007,
                -0.004,
                0.003,
                0.005,
                -0.002,
                0.006,
                -0.003,
                0.004,
                0.002,
                -0.001,
                0.005,
                -0.002,
                0.003,
            ],
            "sma_5": [100.0 + i * 0.6 for i in range(20)],  # Fast - steeper
            "sma_10": [100.0 + i * 0.4 for i in range(20)],  # Slow - flatter
        }
    )


class TestBacktestResult:
    def test_to_dict_contains_all_fields(self):
        result = BacktestResult(
            total_return=0.15,
            sharpe_ratio=1.2,
            max_drawdown=-0.08,
            win_rate=0.55,
            num_trades=24,
        )
        d = result.to_dict()

        assert d["total_return"] == 0.15
        assert d["sharpe_ratio"] == 1.2
        assert d["max_drawdown"] == -0.08
        assert d["win_rate"] == 0.55
        assert d["num_trades"] == 24


class TestAddSignals:
    def test_creates_signal_column(self, preprocessed_data):
        result = add_signals(preprocessed_data, "sma_5", "sma_10")
        assert "signal" in result.columns

    def test_signal_values_are_valid(self, preprocessed_data):
        result = add_signals(preprocessed_data, "sma_5", "sma_10")
        unique_signals = set(result["signal"].to_list())
        assert unique_signals.issubset({-1, 1})

    def test_long_when_fast_above_slow(self):
        df = pl.DataFrame(
            {
                "sma_5": [110.0],
                "sma_10": [100.0],
            }
        )
        result = add_signals(df, "sma_5", "sma_10")
        assert result["signal"][0] == 1

    def test_short_when_fast_below_slow(self):
        df = pl.DataFrame(
            {
                "sma_5": [90.0],
                "sma_10": [100.0],
            }
        )
        result = add_signals(df, "sma_5", "sma_10")
        assert result["signal"][0] == -1


class TestRunBacktest:
    def test_returns_backtest_result(self, preprocessed_data):
        result = run_backtest(preprocessed_data, fast_sma=5, slow_sma=10)
        assert isinstance(result, BacktestResult)

    def test_raises_on_missing_columns(self):
        df = pl.DataFrame({"close": [100.0, 101.0]})
        with pytest.raises(ValueError, match="Missing required columns"):
            run_backtest(df, fast_sma=5, slow_sma=10)

    def test_max_drawdown_is_negative_or_zero(self, preprocessed_data):
        result = run_backtest(preprocessed_data, fast_sma=5, slow_sma=10)
        assert result.max_drawdown <= 0

    def test_win_rate_between_zero_and_one(self, preprocessed_data):
        result = run_backtest(preprocessed_data, fast_sma=5, slow_sma=10)
        assert 0 <= result.win_rate <= 1

    def test_deterministic_results(self, preprocessed_data):
        """Same input should produce same output."""
        result1 = run_backtest(preprocessed_data, fast_sma=5, slow_sma=10)
        result2 = run_backtest(preprocessed_data, fast_sma=5, slow_sma=10)

        assert result1.total_return == result2.total_return
        assert result1.sharpe_ratio == result2.sharpe_ratio
        assert result1.max_drawdown == result2.max_drawdown

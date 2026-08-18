"""Simple backtest engine - SMA crossover strategy."""

from dataclasses import dataclass

import polars as pl


@dataclass
class BacktestResult:
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    num_trades: int

    def to_dict(self) -> dict:
        return {
            "total_return": self.total_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "num_trades": self.num_trades,
        }


def add_signals(df: pl.DataFrame, fast_col: str, slow_col: str) -> pl.DataFrame:
    """Generate signals: 1 = long when fast > slow, -1 = short when fast < slow."""
    return df.with_columns(
        pl.when(pl.col(fast_col) > pl.col(slow_col)).then(1).otherwise(-1).alias("signal")
    )


def add_strategy_returns(df: pl.DataFrame) -> pl.DataFrame:
    """Compute strategy returns: previous signal * today's return."""
    return df.with_columns(
        (pl.col("signal").shift(1) * pl.col("returns")).alias("strategy_returns")
    )


def compute_metrics(df: pl.DataFrame) -> BacktestResult:
    """Compute backtest metrics from strategy returns."""
    returns = df["strategy_returns"].drop_nulls()

    if returns.is_empty():
        return BacktestResult(0.0, 0.0, 0.0, 0.0, 0)

    # Total return
    total_return = float((1 + returns).product() - 1)

    # Sharpe ratio (annualized, assuming daily returns)
    std = returns.std()
    sharpe = float(returns.mean() / std * (252**0.5)) if std and std > 0 else 0.0

    # Max drawdown
    cumulative = (1 + returns).cum_prod()
    rolling_max = cumulative.cum_max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = float(drawdown.min())

    # Win rate
    wins = (returns > 0).sum()
    total = returns.len()
    win_rate = float(wins / total) if total > 0 else 0.0

    # Number of trades (signal changes)
    signals = df["signal"].drop_nulls()
    num_trades = int((signals.diff().abs() > 0).sum()) if signals.len() > 1 else 0

    return BacktestResult(
        total_return=total_return,
        sharpe_ratio=sharpe,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        num_trades=num_trades,
    )


def run_backtest(
    df: pl.DataFrame,
    fast_sma: int = 20,
    slow_sma: int = 50,
) -> BacktestResult:
    """Run SMA crossover backtest on preprocessed data.

    Expects df to have columns: returns, sma_{fast_sma}, sma_{slow_sma}
    """
    fast_col = f"sma_{fast_sma}"
    slow_col = f"sma_{slow_sma}"

    # Validate required columns
    required = ["returns", fast_col, slow_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = add_signals(df, fast_col, slow_col)
    df = add_strategy_returns(df)
    return compute_metrics(df)

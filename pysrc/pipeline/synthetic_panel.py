"""Synthetic pipeline panel fixtures shared across lanes."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRAILING_RETURN_COLUMN = "adjusted_return_1d"


def generate_synthetic_panel_frame(
    *,
    random_seed: int = 42,
    n_tickers: int = 40,
    n_days: int = 180,
    regime_length: int = 40,
) -> pd.DataFrame:
    """Synthetic ticker-date panel matching the pipeline product schema subset."""

    rng = np.random.default_rng(random_seed)
    dates = pd.bdate_range("2024-01-02", periods=n_days).strftime("%Y-%m-%d")
    tickers = [f"SYN{i:03d}" for i in range(n_tickers)]

    frames: list[pd.DataFrame] = []
    regime = np.array([(t // regime_length) % 2 for t in range(n_days)])
    for ticker in tickers:
        noise = rng.normal(0.0, 0.012, size=n_days)
        returns = np.zeros(n_days)
        for t in range(n_days):
            trail = returns[max(0, t - 5) : t].sum()
            if regime[t] == 0:
                returns[t] = 0.35 * np.tanh(8.0 * trail) * 0.01 + noise[t]
            else:
                returns[t] = -0.35 * np.tanh(8.0 * trail) * 0.01 + noise[t]
        price = 50.0 * np.cumprod(1.0 + returns)
        series = pd.Series(price)
        ret = pd.Series(returns)
        forward = series.shift(-1) / series - 1.0

        ema_12 = series.ewm(span=12, adjust=False).mean()
        ema_26 = series.ewm(span=26, adjust=False).mean()
        ema_20 = series.ewm(span=20, adjust=False).mean()
        ema_50 = series.ewm(span=50, adjust=False).mean()
        ma_20 = series.rolling(20, min_periods=5).mean()
        sd_20 = series.rolling(20, min_periods=5).std()
        rolling_max_60 = series.rolling(60, min_periods=10).max()

        frame = pd.DataFrame(
            {
                "date": dates,
                "instrument": ticker,
                "interval": "daily",
                "adj_close": series.to_numpy(dtype=np.float64),
                TRAILING_RETURN_COLUMN: returns,
                "raw_return_1d": returns,
                "forward_return_horizon": forward.to_numpy(dtype=np.float64),
                "roc_5": series.pct_change(5),
                "roc_20": series.pct_change(20),
                "rsi_14": 50.0 + 50.0 * np.tanh(ret.rolling(14, min_periods=5).mean() * 30.0),
                "cci_20": (series - ma_20) / (0.015 * sd_20.replace(0.0, np.nan)),
                "macd_12_26_9": (ema_12 - ema_26) / series,
                "ema_distance_20": series / ema_20 - 1.0,
                "ema_distance_50": series / ema_50 - 1.0,
                "bb_percent_b_20_2": (series - ma_20) / (4.0 * sd_20.replace(0.0, np.nan)) + 0.5,
                "natr_14": ret.abs().rolling(14, min_periods=5).mean(),
                "drawdown_60": series / rolling_max_60 - 1.0,
                "realized_volatility_20": ret.rolling(20, min_periods=5).std(),
                "volume_zscore_20": rng.normal(0.0, 1.0, size=n_days),
                "dollar_volume_ratio_20": np.abs(rng.normal(1.0, 0.2, size=n_days)),
            }
        )
        frame.loc[0, [TRAILING_RETURN_COLUMN, "raw_return_1d"]] = np.nan
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


__all__ = ["TRAILING_RETURN_COLUMN", "generate_synthetic_panel_frame"]

"""
py/backtesting/validation/mechanical/fixtures/generators/synthetic_ohlcv.py

Deterministic synthetic OHLCV generator for unit / integration tests.
Uses a seeded random walk so the same call always produces the same data.

Usage::

    from pysrc.backtesting.validation.mechanical.fixtures.generators.synthetic_ohlcv import (
        generate_ohlcv,
    )

    df = generate_ohlcv(n_rows=252, seed=42)
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import polars as pl


def generate_ohlcv(
    n_rows: int = 252,
    *,
    start_price: float = 100.0,
    daily_vol: float = 0.01,
    start_date: date | None = None,
    seed: int = 42,
) -> pl.DataFrame:
    """Generate a deterministic synthetic OHLCV DataFrame.

    Uses a seeded geometric Brownian motion to produce close prices,
    then derives open/high/low/volume from them so the result is
    internally consistent and reproducible across platforms.

    Args:
        n_rows:       Number of trading-day rows to generate.
        start_price:  Starting close price.
        daily_vol:    Per-period log-return volatility.
        start_date:   First date in the series (defaults to 2020-01-01).
        seed:         RNG seed for reproducibility.

    Returns:
        DataFrame with columns: ``date``, ``open``, ``high``, ``low``,
        ``close``, ``volume``.
    """
    if start_date is None:
        start_date = date(2020, 1, 1)

    # Seeded LCG — no numpy/random dependency so the output is identical
    # across every platform and Python version.
    def _lcg(state: int) -> tuple[int, float]:
        """Return (next_state, uniform_float_in_(-1,1))."""
        # Parameters from Knuth MMIX
        state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        # Map to (-1, 1)
        return state, (state / 0x8000000000000000) - 1.0

    dates: list[date] = []
    opens: list[float] = []
    highs: list[float] = []
    lows: list[float] = []
    closes: list[float] = []
    volumes: list[int] = []

    price = start_price
    state = seed
    current_date = start_date

    for _ in range(n_rows):
        state, z = _lcg(state)
        log_ret = daily_vol * z
        close = price * math.exp(log_ret)

        # Intraday range: |z| * vol * price for high/low spread
        state, z2 = _lcg(state)
        half_spread = abs(z2) * daily_vol * price * 0.5
        high = max(price, close) + half_spread
        low = min(price, close) - half_spread

        state, z3 = _lcg(state)
        open_price = price * math.exp(daily_vol * z3 * 0.3)

        state, z4 = _lcg(state)
        volume = max(1, int(1_000_000 * (0.5 + abs(z4))))

        dates.append(current_date)
        opens.append(round(open_price, 6))
        highs.append(round(high, 6))
        lows.append(round(max(low, 0.01), 6))
        closes.append(round(close, 6))
        volumes.append(volume)

        price = close
        # Advance by one calendar day; skip weekends simply by always
        # incrementing — tests don't need market-calendar accuracy.
        current_date += timedelta(days=1)

    return pl.DataFrame(
        {
            "date": [str(d) for d in dates],
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )

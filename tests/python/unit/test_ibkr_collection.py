# tests/python/unit/test_ibkr_collection.py
from types import SimpleNamespace

import pandas as pd

from pysrc.infra.brokers.ibkr import ib_data_collection as ibdc


def _bars(n=3):
    out = []
    for i in range(n):
        out.append(
            SimpleNamespace(
                date=pd.Timestamp("2025-01-01") + pd.Timedelta(days=i),
                open=100.0 + i,
                high=100.5 + i,
                low=98.5 + i,
                close=99.0 + i,
                volume=1000 + 10 * i,
                average=99.5 + i,  # provide optional fields your code may expect
                barCount=100 + i,
            )
        )
    return out


def test_bars_to_df_happy():
    df = ibdc._bars_to_df(_bars(3))
    # accept either minimal or extended column set, but order should start with OHLCV
    cols = list(df.columns)
    assert cols[:6] == ["open", "high", "low", "close", "volume", "average"] or cols[:5] == [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

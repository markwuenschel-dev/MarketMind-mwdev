# tests/python/unit/test_ibkr_edges.py
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd

from pysrc.infra.brokers.ibkr import ib_data_collection as ibdc


def create_mock_bars(n=2, start_date="2025-01-01"):
    base = pd.Timestamp(start_date)
    out = []
    for i in range(n):
        out.append(
            SimpleNamespace(
                date=base + pd.Timedelta(days=i),
                open=100.0 + i,
                high=101.0 + i,
                low=99.0 + i,
                close=100.0 + i,
                volume=1000 + i,
                average=100.0 + i,
                barCount=100 + i,
            )
        )
    return out


def test_cache_write(tmp_path, monkeypatch):
    ib = MagicMock()
    bars = create_mock_bars(2, start_date="2025-01-01")
    ib.reqHistoricalData.return_value = bars
    outp = tmp_path / "AAPL.parquet"
    monkeypatch.setattr(ibdc, "_get_cache_path", lambda s: outp)

    df = ibdc.fetch_historical_data("AAPL", "20250101 16:00:00", ib_client=ib, use_cache=True)
    assert outp.exists()
    assert abs(df["close"].iloc[0] - 100.0) < 1e-6


def test_uses_provided_client_no_context(monkeypatch):
    # ensure we never call ib_connection if ib_client is supplied
    used_ctx = {"hit": False}

    def _no_ctx():
        used_ctx["hit"] = True
        raise RuntimeError("shouldn't use ctx")

    monkeypatch.setattr(ibdc, "ib_connection", _no_ctx, raising=True)
    ib = MagicMock()
    ib.reqHistoricalData.return_value = create_mock_bars(1)
    _ = ibdc.fetch_historical_data("SPY", "20250101 16:00:00", ib_client=ib, use_cache=False)
    ib.reqHistoricalData.assert_called_once()
    assert used_ctx["hit"] is False

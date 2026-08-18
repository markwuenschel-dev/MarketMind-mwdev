# Relocated from pysrc.data.ib_data_collection
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from pysrc.core.runtime.optional_imports import optional_import

pd = optional_import("pandas")


def _get(obj, *names, default=None):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
        if isinstance(obj, dict) and n in obj:
            return obj[n]
    return default


def _bars_to_df(bars: Iterable) -> pd.DataFrame:
    idx = []
    rows = []
    for b in bars:
        date = getattr(b, "date", pd.Timestamp.now())
        idx.append(date)
        avg = float(getattr(b, "average", (float(b.open) + float(b.close)) / 2.0))
        count = int(getattr(b, "barCount", 0))
        rows.append(
            [
                float(b.open),
                float(b.high),
                float(b.low),
                float(b.close),
                int(b.volume),
                avg,
                count,
            ]
        )
    return pd.DataFrame(
        rows,
        index=pd.DatetimeIndex(idx),
        columns=["open", "high", "low", "close", "volume", "average", "barCount"],
    )


@dataclass
class _Bar:
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


def create_mock_bars(n: int, start_date: str = "2025-01-01"):
    start = pd.to_datetime(start_date)
    out = []
    for i in range(n):
        ts = start + timedelta(days=i)
        o = 100.0 + i
        c = o + (i % 3) - 1
        h = max(o, c) + 0.5
        l = min(o, c) - 0.5
        v = 1000 + 10 * i
        out.append(_Bar(ts.to_pydatetime(), o, h, l, c, v))
    return out


class NoDataError(RuntimeError):
    pass


def _get_cache_path(symbol: str) -> Path:
    return Path(f"cache/{symbol}.parquet")


async def _fetch_historical_async(*args, **kwargs):
    return _bars_to_df(create_mock_bars(1))


@contextmanager
def ib_connection(*args, **kwargs):
    """Mock IB connection context manager for tests."""

    class MockIBConnection:
        def reqHistoricalData(self, symbol, end_datetime, duration, bar_size, what_to_show):
            return create_mock_bars(10)

    try:
        yield MockIBConnection()
    finally:
        pass


def fetch_historical_data(
    symbol: str,
    end_datetime: str,
    *,
    duration: str = "1 D",
    bar_size: str = "1 min",
    what_to_show: str = "TRADES",
    use_cache: bool = True,
    ib_client: object | None = None,
):
    cache_path = _get_cache_path(symbol)
    if use_cache and cache_path.exists():
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            pass

    if ib_client is not None:
        bars = ib_client.reqHistoricalData(symbol, end_datetime, duration, bar_size, what_to_show)
        df = _bars_to_df(bars)
        if use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path)
        return df

    try:
        from pysrc.data.ib_api import ib_connection as real_ib_connection

        connection = real_ib_connection
    except ImportError:
        connection = ib_connection

    with connection() as ctx:
        bars = ctx.reqHistoricalData(symbol, end_datetime, duration, bar_size, what_to_show)
        df = _bars_to_df(bars)
        if use_cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path)
        return df


def _fetch_with_client(ib, symbol, end_datetime, *, use_cache: bool, **kwargs):
    return ib.reqHistoricalData(symbol, end_datetime, **kwargs)

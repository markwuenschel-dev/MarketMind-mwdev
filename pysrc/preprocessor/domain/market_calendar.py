from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import UTC, datetime, timedelta, timezone
from time import perf_counter

from pysrc.core.errors import PreprocessingError
from pysrc.ops.mm_logkit import get_logger
from pysrc.preprocessor.utils.cuda_runtime import capabilities

logger = get_logger(__name__)

try:
    import cudf
except ImportError:
    cudf = None
try:
    import polars as pl
except ImportError:
    pl = None


class MarketCalendar(ABC):
    @abstractmethod
    def is_open_at(self, ts: datetime) -> bool: ...

    @abstractmethod
    def next_open(self, ts: datetime) -> datetime: ...

    @abstractmethod
    def next_close(self, ts: datetime) -> datetime: ...


class FallbackCalendar(MarketCalendar):
    def _ny(self, ts: datetime) -> datetime:
        try:
            import zoneinfo

            tz = zoneinfo.ZoneInfo("America/New_York")
        except Exception:
            tz = timezone(timedelta(hours=-4 if 3 < ts.month < 11 else -5))
        return ts.astimezone(tz)

    def is_open_at(self, ts: datetime) -> bool:
        tny = self._ny(ts)
        if tny.weekday() >= 5:
            return False
        hm = tny.hour * 60 + tny.minute
        return 9 * 60 + 30 <= hm < 16 * 60

    def next_open(self, ts: datetime) -> datetime:
        t = self._ny(ts)
        if self.is_open_at(t):
            return t.astimezone(ts.tzinfo)
        d = t
        while True:
            d += timedelta(minutes=1)
            if d.weekday() >= 5:
                d = d.replace(hour=9, minute=30, second=0, microsecond=0) + timedelta(
                    days=7 - d.weekday()
                )
            if self.is_open_at(d):
                return d.astimezone(ts.tzinfo)

    def next_close(self, ts: datetime) -> datetime:
        t = self._ny(ts)
        if not self.is_open_at(t):
            return (
                self.next_open(t)
                .replace(hour=16, minute=0, second=0, microsecond=0)
                .astimezone(ts.tzinfo)
            )
        return t.replace(hour=16, minute=0, second=0, microsecond=0).astimezone(ts.tzinfo)


def profile_calendar(func: Callable) -> Callable:
    metrics: dict[str, float] = {}

    def wrapper(self, *args, **kwargs):
        key = type(self).__name__
        start = perf_counter()
        result = func(self, *args, **kwargs)
        duration = perf_counter() - start
        if key not in metrics or duration < metrics[key]:
            metrics[key] = duration
            logger.info("Evolved: Faster calendar %s: %ss", key, duration)
        return result

    return wrapper


MarketCalendar.is_open_at = profile_calendar(MarketCalendar.is_open_at)
MarketCalendar.next_open = profile_calendar(MarketCalendar.next_open)
MarketCalendar.next_close = profile_calendar(MarketCalendar.next_close)


class MarketCalendarFactory:
    @staticmethod
    def get_calendar(exchange: str = "NYSE", governed: bool = False) -> MarketCalendar:
        try:
            import exchange_calendars as ec

            class ExchangeCal(MarketCalendar):
                def __init__(self, exc: str):
                    self.cal = ec.get_calendar(exc)

                def is_open_at(self, ts: datetime) -> bool:
                    return self.cal.is_session(ts.date())

                def next_open(self, ts: datetime) -> datetime:
                    return self.cal.next_open(ts).to_pydatetime()

                def next_close(self, ts: datetime) -> datetime:
                    return self.cal.next_close(ts).to_pydatetime()

            return ExchangeCal(exchange)
        except ImportError:
            pass
        try:
            import pandas_market_calendars as mcal

            class PMCCal(MarketCalendar):
                def __init__(self, exc: str):
                    self.cal = mcal.get_calendar(exc)

                def is_open_at(self, ts: datetime) -> bool:
                    sched = self.cal.schedule(start_date=ts.date(), end_date=ts.date())
                    return not sched.empty

                def next_open(self, ts: datetime) -> datetime:
                    return self.cal.date_range(ts, ts + timedelta(days=10), frequency="B")[
                        0
                    ].to_pydatetime()

                def next_close(self, ts: datetime) -> datetime:
                    sched = self.cal.schedule(start_date=ts.date(), end_date=ts.date())
                    return sched.market_close[0].to_pydatetime()

            return PMCCal(exchange)
        except ImportError as exc:
            if governed:
                raise PreprocessingError(
                    "governed path rejected approximate calendar fallback"
                ) from exc
            logger.warning(
                "Using fallback calendar; install exchange_calendars or pandas_market_calendars for accuracy"
            )
            return FallbackCalendar()


_CAL = MarketCalendarFactory.get_calendar()


def is_session(ts: datetime | None = None, cal: MarketCalendar | None = None) -> bool:
    cal = cal or _CAL
    ts = ts or datetime.now(UTC)
    return cal.is_open_at(ts)


def next_session(ts: datetime | None = None, cal: MarketCalendar | None = None) -> datetime:
    cal = cal or _CAL
    ts = ts or datetime.now(UTC)
    return cal.next_open(ts)


def time_bucket(ts: datetime, seconds: int) -> datetime:
    epoch = int(ts.timestamp())
    bucket = epoch - (epoch % seconds)
    return datetime.fromtimestamp(bucket, tz=ts.tzinfo)


def resample_ohlcv(df, ts_col: str, seconds: int, engine: str | None = None):
    caps = capabilities()
    engine = engine or (caps.has_polars_gpu and "polars" or caps.has_cudf and "cudf")
    try:
        if engine == "polars" and pl and isinstance(df, pl.DataFrame):
            return (
                df.lazy()
                .with_columns(pl.col(ts_col).dt.truncate(f"{seconds}s").alias("_b"))
                .group_by("_b")
                .agg(
                    [
                        pl.col("open").first().alias("open"),
                        pl.col("high").max().alias("high"),
                        pl.col("low").min().alias("low"),
                        pl.col("close").last().alias("close"),
                        pl.col("volume").sum().alias("volume"),
                    ]
                )
                .sort("_b")
                .collect(engine="streaming")
            )
    except Exception as exc:
        logger.warning("Polars resample failed: %s", exc)
    try:
        if engine == "cudf" and cudf and isinstance(df, cudf.DataFrame):
            grouped = df.set_index(ts_col).groupby(cudf.Grouper(freq=f"{seconds}s"))
            return grouped.agg(
                {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
            ).reset_index()
    except Exception as exc:
        logger.warning("cuDF resample failed: %s", exc)
    raise RuntimeError("Provide cudf or polars DataFrame for GPU resampling")

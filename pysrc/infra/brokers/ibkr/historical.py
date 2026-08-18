import asyncio
from asyncio import Semaphore
from datetime import datetime
from pathlib import Path

import pandas as pd

from pysrc.core.errors import DataFetchError, IBKRConnectionError
from pysrc.core.validation import validate_date, validate_symbol
from pysrc.ops.mm_logkit import configure_logger, get_logger
from pysrc.pipeline.pipeline_config import get_config

try:
    from ibkr_insync import IBKR, BarData, Stock, util
except ImportError:
    # For smoke tests - create dummy classes
    class IBKR:
        def __init__(self, *args, **kwargs):
            pass

        def connect(self, *args, **kwargs):
            pass

        def reqHistoricalData(self, *args, **kwargs):
            return []

        def disconnect(self):
            pass

    class BarData:
        def __init__(self, *args, **kwargs):
            pass

    class Stock:
        def __init__(self, *args, **kwargs):
            pass

    class util:
        @staticmethod
        def df(*args, **kwargs):
            return pd.DataFrame()


configure_logger(
    "marketmind",
    level="DEBUG",
    handlers=[{"type": "stream", "target": "stderr", "level": "DEBUG", "kind": "kv"}],
)
logger = get_logger(__name__)


class NoDataError(DataFetchError):
    def __init__(self, symbol: str):
        super().__init__(f"No historical data returned for {symbol}")


def _get_cache_path(symbol: str) -> Path:
    cache_dir = Path("data/raw/historical_prices_ibkr")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{symbol}.parquet"


def _bars_to_df(bars: list[BarData]) -> pd.DataFrame:
    try:
        if not bars:
            raise DataFetchError("Empty DataFrame")
        required_fields = {"open", "high", "low", "close", "volume", "barCount", "average"}
        for bar in bars:
            missing = [field for field in required_fields if getattr(bar, field, None) is None]
            if missing:
                raise DataFetchError(f"Missing BarData fields: {', '.join(missing)}")
        df = util.df(bars)
        if df.empty:
            raise DataFetchError("Empty DataFrame after conversion")
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.set_index("date")
        if "wap" in df.columns:
            df = df.rename(columns={"wap": "average"})
        expected_columns = {"open", "high", "low", "close", "volume", "barCount", "average"}
        if not expected_columns.issubset(df.columns):
            raise DataFetchError(
                f"Missing expected columns: {', '.join(expected_columns - set(df.columns))}"
            )
        df = df[~df.index.duplicated(keep="last")].sort_index()
        if df[["open", "high", "low", "close", "volume", "average", "barCount"]].isna().any().any():
            logger.warning(
                "Missing values detected in DataFrame", action="filling with forward fill"
            )
            df = df.ffill()
        return df[["open", "high", "low", "close", "volume", "average", "barCount"]]
    except IBKRConnectionError:
        raise
    except Exception as e:
        logger.error("Data conversion error", error=str(e))
        raise DataFetchError(f"Failed to convert bars to DataFrame: {str(e)}") from e


def create_mock_bars(n: int, start_date: str = "2025-01-01") -> list[BarData]:
    bars = []
    base_date = pd.to_datetime(start_date, utc=True)
    for i in range(n):
        date = (base_date + pd.Timedelta(days=i)).strftime("%Y%m%d %H:%M:%S")
        bars.append(
            BarData(
                date=date,
                open=100.0 + i,
                high=101.0 + i,
                low=99.0 + i,
                close=100.5 + i,
                volume=1000 + i,
                barCount=1,
                average=100.25 + i,
            )
        )
    return bars


async def _fetch_historical_async(
    symbol: str,
    end_date: str,
    duration: str,
    bar_size: str,
    ibkr: IBKR,
    sem: Semaphore,
    what_to_show: str,
    use_rth: bool,
    format_date: int,
    use_cache: bool,
) -> pd.DataFrame:
    validate_symbol(symbol)
    validate_date(end_date)
    log = logger.bind(symbol=symbol, duration=duration, end_date=end_date)
    log.info("Starting async historical data fetch")
    cache_path = _get_cache_path(symbol) if use_cache else None
    cached_df = None
    if use_cache and cache_path.exists():
        try:
            cached_df = pd.read_parquet(cache_path)
            if not cached_df.empty and all(
                col in cached_df.columns
                for col in ["open", "high", "low", "close", "volume", "average", "barCount"]
            ):
                last_date = cached_df.index.max().tz_convert("UTC")
                log.info("Found cached data", last_date=last_date)
                end_ts = pd.to_datetime(end_date or datetime.now(), utc=True)
                if end_ts > last_date:
                    duration = "1 D"
                    end_date = end_ts.strftime("%Y%m%d %H:%M:%S")
                else:
                    log.info("Cache covers requested period, returning cached data")
                    return cached_df
            else:
                cached_df = None
        except Exception as e:
            log.warning("Failed to read cache file", path=str(cache_path), error=str(e))
            cached_df = None
    try:
        async with sem:
            bars = await ibkr.reqHistoricalDataAsync(
                Stock(symbol, "SMART", "USD"),
                endDateTime=end_date,
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=use_rth,
                formatDate=format_date,
            )
        if not bars:
            if cached_df is not None and (not cached_df.empty):
                log.info("No new data, returning cached data")
                return cached_df
            raise NoDataError(symbol)
        df = _bars_to_df(bars)
        if cached_df is not None and (not cached_df.empty):
            df = pd.concat([cached_df, df], axis=0).sort_index().drop_duplicates(keep="last")
        log.info("Successfully fetched historical data", rows=len(df))
        if use_cache:
            try:
                df.to_parquet(cache_path, engine="pyarrow")
                log.info("Saved data to cache", path=str(cache_path))
            except Exception as e:
                log.warning("Failed to save cache file", path=str(cache_path), error=str(e))
        return df
    except IBKRConnectionError as e:
        log.error("IBKR connection failed", error=str(e))
        raise
    except NoDataError:
        log.error("No data returned")
        raise
    except Exception as e:
        log.error("Unexpected error during async data fetch", error=str(e))
        raise DataFetchError(str(e)) from e


def fetch_historical_data(
    symbol: str,
    end_date: str = "",
    duration: str = "1 Y",
    bar_size: str = "1 day",
    ibkr_client: IBKR | None = None,
    use_cache: bool = True,
    what_to_show: str = "TRADES",
    use_rth: bool = True,
    format_date: int = 1,
) -> pd.DataFrame:
    config = get_config()
    if config.real_time_market_data and config.real_time_market_data.interactive_brokers:
        ibkr_config = config.real_time_market_data.interactive_brokers
        what_to_show = ibkr_config.what_to_show
        use_rth = ibkr_config.use_rth
        format_date = ibkr_config.format_date
    else:
        logger.warning(
            "Interactive Brokers configuration missing; using default values",
            what_to_show=what_to_show,
            use_rth=use_rth,
            format_date=format_date,
        )
    validate_symbol(symbol)
    validate_date(end_date)
    log = logger.bind(symbol=symbol, duration=duration, end_date=end_date)
    log.info("Starting historical data fetch")
    cache_path = _get_cache_path(symbol) if use_cache else None
    cached_df = None
    if use_cache and cache_path.exists():
        try:
            cached_df = pd.read_parquet(cache_path)
            if not cached_df.empty and all(
                col in cached_df.columns
                for col in ["open", "high", "low", "close", "volume", "average", "barCount"]
            ):
                last_date = cached_df.index.max().tz_convert("UTC")
                log.info("Found cached data", last_date=last_date)
                end_ts = pd.to_datetime(end_date or datetime.now(), utc=True)
                if end_ts > last_date:
                    duration = "1 D"
                    end_date = end_ts.strftime("%Y%m%d %H:%M:%S")
                else:
                    log.info("Cache covers requested period, skipping fetch")
                    return cached_df
            else:
                cached_df = None
        except Exception as e:
            log.warning("Failed to read cache file", path=str(cache_path), error=str(e))
            cached_df = None
    try:
        if ibkr_client:
            ibkr = ibkr_client
            own_client = False
        else:
            from pysrc.data.ibkr_api import ibkr_connection

            own_client = True
        if own_client:
            with ibkr_connection() as ibkr:
                bars = ibkr.reqHistoricalData(
                    Stock(symbol, "SMART", "USD"),
                    endDateTime=end_date,
                    durationStr=duration,
                    barSizeSetting=bar_size,
                    whatToShow=what_to_show,
                    useRTH=use_rth,
                    formatDate=format_date,
                )
        else:
            bars = ibkr.reqHistoricalData(
                Stock(symbol, "SMART", "USD"),
                endDateTime=end_date,
                durationStr=duration,
                barSizeSetting=bar_size,
                whatToShow=what_to_show,
                useRTH=use_rth,
                formatDate=format_date,
            )
        if not bars:
            if cached_df is not None and (not cached_df.empty):
                log.info("No new data, returning cached data")
                return cached_df
            raise NoDataError(symbol)
        df = _bars_to_df(bars)
        if cached_df is not None and (not cached_df.empty):
            df = pd.concat([cached_df, df], axis=0).sort_index().drop_duplicates(keep="last")
        log.info("Successfully fetched historical data", rows=len(df))
        if use_cache:
            try:
                df.to_parquet(cache_path, engine="pyarrow")
                log.info("Saved data to cache", path=str(cache_path))
            except Exception as e:
                log.warning("Failed to save cache file", path=str(cache_path), error=str(e))
        return df
    except IBKRConnectionError as e:
        log.error("IBKR connection failed", error=str(e))
        raise
    except NoDataError:
        log.error("No data returned")
        raise
    except Exception as e:
        log.error("Unexpected error during data fetch", error=str(e))
        raise DataFetchError(str(e)) from e


async def fetch_multiple_historical_data(
    symbols: list[str],
    end_date: str = "",
    duration: str = "1 Y",
    bar_size: str = "1 day",
    use_cache: bool = True,
    what_to_show: str = "TRADES",
    use_rth: bool = True,
    format_date: int = 1,
) -> dict[str, pd.DataFrame]:
    from pysrc.data.ibkr_api import ibkr_connection

    config = get_config()
    if config.real_time_market_data and config.real_time_market_data.interactive_brokers:
        ibkr_config = config.real_time_market_data.interactive_brokers
        what_to_show = ibkr_config.what_to_show
        use_rth = ibkr_config.use_rth
        format_date = ibkr_config.format_date
    else:
        logger.warning(
            "Interactive Brokers configuration missing; using default values",
            what_to_show=what_to_show,
            use_rth=use_rth,
            format_date=format_date,
        )
    data: dict[str, pd.DataFrame] = {}
    sem = Semaphore(5)

    async def fetch_all():
        with ibkr_connection() as ibkr:
            tasks = [
                _fetch_historical_async(
                    symbol,
                    end_date,
                    duration,
                    bar_size,
                    ibkr,
                    sem,
                    what_to_show,
                    use_rth,
                    format_date,
                    use_cache,
                )
                for symbol in symbols
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for symbol, result in zip(symbols, results, strict=False):
                if isinstance(result, pd.DataFrame):
                    data[symbol] = result
                elif isinstance(result, Exception):
                    logger.warning("Failed to fetch data", symbol=symbol, error=str(result))

    await fetch_all()
    return data

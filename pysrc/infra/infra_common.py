# py/infra/brokers/infra_common.py
"""Provides shared utilities (retry, normalization, logging)."""

from __future__ import annotations

import asyncio
import functools
import logging
import random
from collections.abc import Callable
from typing import Any

import polars as pl

from pysrc.core.errors import DataError, DataFetchError, DataValidationError, PreprocessingError
from pysrc.ops.mm_logkit import get_logger


def retry_async(
    retries: int = 3,
    backoff_factor: float = 0.5,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    jitter: float = 0.2,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator for async retry with exponential backoff (+ jitter).
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            attempt = 0
            while True:
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    if attempt >= retries:
                        # concrete, not abstract
                        raise DataFetchError(
                            message=f"Retry failed after {retries} attempts",
                            details={
                                "function": getattr(func, "__name__", "<unknown>"),
                                "retries": retries,
                                "last_error": str(e),
                            },
                        ) from e
                    sleep_time = backoff_factor * (2 ** (attempt - 1))
                    # ± jitter%
                    sleep_time *= 1.0 + random.uniform(-jitter, jitter)
                    await asyncio.sleep(max(0.0, sleep_time))

        return wrapper

    return decorator


def normalize_dataframe(
    df: Any,
    schema: dict[str, pl.DataType] | None = None,
    engine: str = "polars",
) -> pl.DataFrame:
    """
    Normalize data to a Polars DataFrame, with optional schema enforcement.
    """
    if engine != "polars":
        raise PreprocessingError(f"Unsupported engine: {engine}")

    try:
        if isinstance(df, pl.DataFrame):
            pl_df = df
        else:
            # convert common shapes to Polars
            try:
                import pandas as pd  # optional; only for conversion
            except ImportError as ie:
                # only needed if input is pandas
                if not (isinstance(df, list) and all(isinstance(it, dict) for it in df)) and not (
                    isinstance(df, dict) and all(isinstance(v, list) for v in df.values())
                ):
                    raise PreprocessingError(
                        "pandas required for conversion; install if needed."
                    ) from ie
                pd = None  # satisfy type checkers

            if pd is not None and "pandas.core.frame" in type(df).__module__:
                pl_df = pl.from_pandas(df)  # type: ignore[arg-type]
            elif (
                isinstance(df, list)
                and all(isinstance(it, dict) for it in df)
                or isinstance(df, dict)
                and all(isinstance(v, list) for v in df.values())
            ):
                pl_df = pl.DataFrame(df)
            else:
                raise PreprocessingError("Unsupported input type for normalization.")

        if schema:
            try:
                pl_df = pl_df.cast(schema)
            except Exception as ce:
                raise DataValidationError(
                    "Schema casting failed",
                    details={"error": str(ce), "schema_keys": list(schema.keys())},
                )

        return pl_df

    except PreprocessingError:
        # already well-typed; just bubble up
        raise


def ensure_lazy(df: Any, *, schema: dict[str, pl.DataType] | None = None) -> pl.LazyFrame:
    """
    Ensure a Polars LazyFrame (best for preprocessing pipelines).
    Accepts pl.LazyFrame, pl.DataFrame, pandas.DataFrame, list[dict], dict[str, list].
    """
    if isinstance(df, pl.LazyFrame):
        lf = df
    elif isinstance(df, pl.DataFrame):
        lf = df.lazy()
    else:
        # Convert other shapes to DataFrame first
        try:
            import pandas as pd  # optional

            if isinstance(df, pd.DataFrame):
                lf = pl.from_pandas(df).lazy()
            elif (
                isinstance(df, list)
                and all(isinstance(x, dict) for x in df)
                or isinstance(df, dict)
                and all(isinstance(v, list) for v in df.values())
            ):
                lf = pl.DataFrame(df).lazy()
            else:
                raise DataError("Unsupported input type for ensure_lazy().")
        except ImportError as e:
            raise DataError("pandas required for conversion in ensure_lazy().") from e

    if schema is not None:
        # cast at lazy level via with_columns to ensure types on collect
        lf = lf.with_columns([pl.col(k).cast(t) for k, t in schema.items() if k in lf.columns])
    return lf


def setup_logger(
    name: str,
    level: int = logging.INFO,
    fmt: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
) -> logging.Logger:
    """
    Set up or fetch a project logger; integrates with structlog if configured in utils.
    """
    logger = get_logger(name)
    try:
        logger.setLevel(level)  # if plain logging.Logger
    except Exception:
        pass
    if not getattr(logger, "handlers", None):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt))
        try:
            logger.addHandler(handler)
        except Exception:
            # structlog BoundLogger won't have addHandler; ignore
            pass
    return logger

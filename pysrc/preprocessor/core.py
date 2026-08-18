"""
preprocessor/core.py — public convenience API

Thin delegators to the graph engine via PreprocessorBuilder.
Public signatures preserved so existing callers do not break.

DO NOT add computational logic here.
Add ops to preprocessor/graph/ops.py and register them in the backend
registry instead.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from pysrc.core.errors import PreprocessingError
from pysrc.preprocessor.api import Backend, PreprocessorBuilder, run


def load_ohlcv(path: Path, *, backend: Backend = "auto") -> pl.DataFrame:
    """Load OHLCV data from CSV through the graph engine.
    Graph op: data.load_csv.
    """
    try:
        plan = (
            PreprocessorBuilder(backend=backend)
            .add_op("data.load_csv", path=str(path), try_parse_dates=True)
            .build_plan()
        )
        return run(None, plan, backend=backend)
    except Exception as exc:
        raise PreprocessingError(
            "governed path rejected compatibility fallback: direct_csv"
        ) from exc


def add_returns(
    df: pl.DataFrame,
    column: str = "close",
    *,
    backend: Backend = "auto",
) -> pl.DataFrame:
    """Add daily returns column. Output column: `returns`"""
    plan = (
        PreprocessorBuilder(backend=backend).add_op("feature.returns", column=column).build_plan()
    )
    return run(df, plan, backend=backend)


def add_sma(
    df: pl.DataFrame,
    column: str = "close",
    window: int = 20,
    *,
    backend: Backend = "auto",
) -> pl.DataFrame:
    """Add simple moving average. Output column: `sma_{window}`"""
    plan = (
        PreprocessorBuilder(backend=backend)
        .add_op("feature.sma", column=column, window=window)
        .build_plan()
    )
    return run(df, plan, backend=backend)


def add_rsi(
    df: pl.DataFrame,
    column: str = "close",
    window: int = 14,
    *,
    backend: Backend = "auto",
) -> pl.DataFrame:
    """Add RSI indicator. Output column: `rsi_{window}`"""
    plan = (
        PreprocessorBuilder(backend=backend)
        .add_op("feature.rsi", column=column, window=window)
        .build_plan()
    )
    return run(df, plan, backend=backend)


def build_features(
    df: pl.DataFrame,
    *,
    sma_windows: list[int] | None = None,
    rsi_window: int = 14,
    backend: Backend = "auto",
) -> pl.DataFrame:
    """Add returns + multiple SMAs + RSI in a single graph pass.

    Preferred call site for cli/preprocess.py — one plan, one executor
    invocation, no intermediate DataFrames.
    """
    sma_windows = sma_windows or [20, 50]
    builder = PreprocessorBuilder(backend=backend).add_op("feature.returns")
    for w in sma_windows:
        builder.add_op("feature.sma", window=w)
    builder.add_op("feature.rsi", window=rsi_window)
    return run(df, builder.build_plan(), backend=backend)

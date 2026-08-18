# py/pipeline/core/pipeline_core_context.py
"""Manages PipelineContext (e.g., frequency inference, indicator flags)."""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass as pydantic_dataclass

from pysrc.core.runtime.optional_imports import optional_import

pd = optional_import("pandas")
pl = optional_import("polars")

TimeFreq = Literal["tick", "min", "day", "1d"]


@pydantic_dataclass(config=ConfigDict(arbitrary_types_allowed=True), frozen=True)
class PipelineContext:
    frequency: TimeFreq = "min"
    asset_class: str = "equity"
    latency: Literal["ultra", "low", "batch"] = "low"
    streaming: bool = False
    time_col: str = "index"
    df: pl.DataFrame | pl.LazyFrame | pd.DataFrame | None = None
    assume_sorted: bool = True
    sample: int = 200_000
    backend: str | None = None
    executor: str | None = None
    optimize: bool = False
    cache: bool = False

    def as_lazy(self) -> pl.LazyFrame | None:  # type: ignore[name-defined]
        if self.df is None or pl is None:
            return None
        # Already lazy
        if isinstance(self.df, getattr(pl, "LazyFrame", ())):
            return self.df  # type: ignore[return-value]
        # Eager Polars -> make lazy
        if isinstance(self.df, getattr(pl, "DataFrame", ())):
            return self.df.lazy()  # type: ignore[return-value]
        # Pandas or other types: we don't coerce here
        return None

    def infer_frequency(self) -> TimeFreq:
        lf = self.as_lazy()
        if lf is None:
            return self.frequency

        # Validate time_col exists in the DataFrame (schema names avoid LazyFrame.columns resolution cost)
        col_names = lf.collect_schema().names()
        if self.time_col not in col_names:
            # Auto-detect timestamp column if specified column doesn't exist
            from pysrc.pipeline.orchestrator import _TS_NAME_CANDIDATES

            actual_cols = set(col_names)
            for candidate in _TS_NAME_CANDIDATES:
                if candidate in actual_cols:
                    # Use this column instead, but don't modify frozen context
                    ts = pl.col(candidate)
                    break
            else:
                # No timestamp column found - return default
                return self.frequency
        else:
            ts = pl.col(self.time_col)

        try:
            med_ns = (
                (
                    lf.select(
                        (
                            (ts if self.assume_sorted else ts.sort())
                            .dt.epoch("ns")
                            .diff()
                            .drop_nulls()
                        )
                        .limit(self.sample)
                        .median()
                        .alias("med_ns")
                    )
                )
                .collect(engine="streaming")
                .item()
            )
            if med_ns is None:
                return "day"
            secs = med_ns / 1e9
            return "tick" if secs <= 1 else "min" if secs <= 60 else "day"
        except (AttributeError, ValueError, TypeError):
            # Fallback if frequency detection fails
            return self.frequency

    def refine(self, **kwargs) -> PipelineContext:
        from dataclasses import replace

        return replace(self, **kwargs)

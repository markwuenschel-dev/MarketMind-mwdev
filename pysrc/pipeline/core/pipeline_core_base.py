# py/pipeline/core/pipeline_core_base.py
"""Defines PipelineStep ABC (Polars-first), DataError hierarchy, and infra."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, AsyncIterator
from enum import Enum
from typing import (
    Any,
    Generic,
    Literal,
    TypeVar,
)

from pysrc.core.runtime.optional_imports import optional_import
from pysrc.pipeline.core.pipeline_core_context import PipelineContext

pd = optional_import("pandas")
pl = optional_import("polars")

try:
    from .pipeline_core_registry import StepRegistry as _StepRegistry

    StepRegistry = _StepRegistry
except Exception:
    # minimal fallback if registry module isn't available
    class StepRegistry(dict):
        def register(self, name, cls):
            self[name] = cls
            return cls


InT = TypeVar("InT")
OutT = TypeVar("OutT")
Engine = Literal["polars", "pandas", "dask"]


class PipelineError(Exception):
    code = "PIPELINE_ERROR"

    def to_dict(self) -> dict:
        return {"code": self.code, "msg": str(self)}


class PipelineGraphError(PipelineError):
    code = "PIPELINE_GRAPH"


class PipelineConfigError(PipelineError):
    code = "PIPELINE_CONFIG"


class ErrorCode(Enum):
    """Enumeration of error codes for data processing issues."""

    MISSING_DATA = "MISSING_DATA"
    INVALID_SCHEMA = "INVALID_SCHEMA"
    OUTLIER_DETECTED = "OUTLIER_DETECTED"
    DRIFT_DETECTED = "DRIFT_DETECTED"
    PROCESSING_FAILURE = "PROCESSING_FAILURE"
    RESOURCE_EXHAUSTED = "RESOURCE_EXHAUSTED"
    # Add more as needed


class DataError(Exception):
    """Base exception for data-related errors in the pipeline."""

    def __init__(self, message: str, code: ErrorCode, details: dict[str, Any] | None = None):
        self.message = message
        self.code = code
        self.details = details or {}
        super().__init__(message)

    def __str__(self) -> str:
        return f"{self.code.value}: {self.message}"

    def to_dict(self) -> dict[str, Any]:
        """Convert the error to a dictionary for logging or serialization."""
        return {"code": self.code.value, "message": self.message, "details": self.details}


# Example derived errors
class MissingDataError(DataError): ...


class InvalidSchemaError(DataError): ...


class DataSource(ABC):  # Existing, extend here
    def __init__(self, config: Any):
        self.config = config

    @abstractmethod
    async def load_data(self, *args, **kwargs) -> pl.DataFrame:
        """Load batch data as Polars DataFrame."""
        pass

    async def stream_data(self) -> AsyncIterable[pl.DataFrame]:
        """Optional: Yield streaming chunks as Polars frames."""
        raise NotImplementedError("Streaming not supported")


# replace the PipelineStep class with this corrected version + a CleaningStep


class PipelineStep(Generic[InT, OutT], ABC):
    STEP_NAME = None
    STEP_VERSION = "0"
    requires: set[str] = set()
    produces: set[str] = set()
    preferred_engine: Engine | None = None

    def __init__(self, name: str | None = None, **_):
        self.name = name or self.__class__.__name__

    async def execute(self, data: Any, context: PipelineContext) -> Any:
        # Import here to avoid circular dependencies
        pl = optional_import("polars")
        pd = optional_import("pandas")

        # Handle Polars DataFrames
        if pl is not None and isinstance(data, (pl.DataFrame, pl.LazyFrame)):
            # Convert to LazyFrame if needed
            lf = data.lazy() if isinstance(data, pl.DataFrame) else data
            result = self.apply_batch(lf, context)
            # Return collected result if it's still lazy
            return result.collect() if isinstance(result, pl.LazyFrame) else result

        # Handle Pandas DataFrames
        if pd is not None and isinstance(data, pd.DataFrame):
            try:
                result = self.apply_batch_pandas(data, context)
                return result
            except NotImplementedError:
                # Fall back to polars conversion if pandas not implemented
                if pl is not None:
                    df_pl = pl.from_pandas(data)
                    lf = df_pl.lazy()
                    result = self.apply_batch(lf, context)
                    result_df = result.collect() if isinstance(result, pl.LazyFrame) else result
                    return result_df.to_pandas()

        # For other types (dicts, etc.), just return as-is
        # Subclasses can override this method for custom handling
        return data

    # Polars-first batch path. Steps may override.
    def apply_batch(self, lf: pl.LazyFrame, ctx: PipelineContext) -> pl.LazyFrame:
        if pl is None:
            raise RuntimeError("polars not available")
        df = lf.collect()
        out = self.fit_transform(df, ctx)  # type: ignore[attr-defined]
        if isinstance(out, pl.DataFrame):
            return out.lazy()
        raise NotImplementedError(
            f"{self.__class__.__name__}.fit_transform must return a Polars DataFrame"
        )

    # Optional: pandas fallback
    def apply_batch_pandas(self, df: pd.DataFrame, ctx: PipelineContext) -> pd.DataFrame:  # type: ignore[name-defined]
        raise NotImplementedError

    # Streaming default passthrough
    async def apply_stream(
        self, aiter: AsyncIterator[dict], ctx: PipelineContext
    ) -> AsyncIterator[dict]:
        async for item in aiter:
            yield item

    @classmethod
    def compose(cls, *steps: PipelineStep) -> CompositeStep:
        return CompositeStep(list(steps))

    def __rshift__(self, other: PipelineStep) -> CompositeStep:
        return CompositeStep([self, other])


class CompositeStep(PipelineStep):
    def __init__(self, steps: list[PipelineStep], **kwargs):
        super().__init__(**kwargs)
        self.steps = steps

    def apply_batch(self, lf: pl.LazyFrame, ctx: PipelineContext) -> pl.LazyFrame:
        for s in self.steps:
            lf = s.apply_batch(lf, ctx)
        return lf

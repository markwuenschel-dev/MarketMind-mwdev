# py/pipeline/stages/market_data/joins.py
from collections.abc import Callable
from functools import reduce

import polars as pl
from polars import LazyFrame
from pydantic import BaseModel

from pysrc.core.errors import DataValidationError
from pysrc.ops.caching import ttl_cache
from pysrc.ops.mm_logkit import get_logger
from pysrc.pipeline.core.pipeline_core_base import PipelineStep

from .transforms import build_steps  # Shared generic builder

logger = get_logger(__name__)

# Simple global registry for source resolution (extensible: could load from pipeline_config/file)
SOURCE_REGISTRY: dict[str, LazyFrame] = {
    # Example: "source1": pl.scan_csv("data/source1.csv"),
    # Populate dynamically or via init function
}


class JoinSpec(BaseModel):
    source_name: str  # Reference to source by name for serialization
    on: list[str]
    how: str = "inner"
    suffix: str | None = None  # User can specify to avoid clashes on repeated joins


class MultiSourceJoinConfig(BaseModel):
    joins: list[JoinSpec]


class MultiSourceJoinStep(PipelineStep):
    def __init__(self, config: MultiSourceJoinConfig):
        self.config = config

    def apply(self, lf: LazyFrame) -> LazyFrame:
        try:
            # Resolve sources from registry
            sources = []
            for spec in self.config.joins:
                source = SOURCE_REGISTRY.get(spec.source_name)
                if source is None:
                    raise DataValidationError(f"Source not found: {spec.source_name}")
                sources.append((source, spec))

            # Chained lazy joins
            lf = reduce(
                lambda left, source_spec: left.join(
                    source_spec[0],
                    on=source_spec[1].on,
                    how=source_spec[1].how,
                    suffix=source_spec[1].suffix or "",
                ),
                sources,
                lf,
            )
            logger.info(f"Joined {len(self.config.joins)} sources")
            return lf
        except Exception as e:
            logger.error(f"Join failed: {e}")
            raise DataValidationError(f"Join failed: {e}") from e


AGG_MAP: dict[str, Callable] = {
    "first": pl.first,
    "last": pl.last,
    "max": pl.max,
    "min": pl.min,
    "sum": pl.sum,
}


class ResampleConfig(BaseModel):
    freq: str
    group_by: list[str] = ["symbol"]
    agg: dict[str, str] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    timestamp_col: str = "timestamp"


class ResampleStep(PipelineStep):
    def __init__(self, config: ResampleConfig):
        self.freq = config.freq
        self.group_by = config.group_by
        self.agg = config.agg
        self.timestamp_col = config.timestamp_col

    def apply(self, lf: LazyFrame) -> LazyFrame:
        unknown = set(self.agg.values()) - set(AGG_MAP)
        if unknown:
            raise DataValidationError(f"Unknown agg functions: {unknown}")
        agg_exprs = [AGG_MAP[f](pl.col(c)).alias(c) for c, f in self.agg.items()]
        try:
            lf = lf.group_by_dynamic(
                self.timestamp_col,
                every=self.freq,
                group_by=self.group_by,
            ).agg(agg_exprs)
            lf = lf.drop_nulls()
            logger.info(f"Resampled to {self.freq}")
            return lf
        except Exception as e:
            logger.error(f"Resample failed: {e}")
            raise DataValidationError(f"Resample failed: {e}") from e


JOIN_STEPS: dict[str, type[PipelineStep]] = {
    "multi_join": MultiSourceJoinStep,
    "resample": ResampleStep,
}

JOIN_CONFIGS: dict[str, type[BaseModel]] = {
    "multi_join": MultiSourceJoinConfig,
    "resample": ResampleConfig,
}


@ttl_cache(ttl=3600)
def build_join_steps(configs: list[dict]) -> list[PipelineStep]:
    return build_steps(configs, JOIN_STEPS, JOIN_CONFIGS)

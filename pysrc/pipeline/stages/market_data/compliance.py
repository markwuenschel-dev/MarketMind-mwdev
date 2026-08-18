# py/pipeline/stages/market_data/compliance.py
import hashlib
import operator
from collections.abc import Callable
from functools import reduce

import polars as pl
from polars import LazyFrame
from pydantic import BaseModel, Field

from pysrc.core.errors import DataValidationError
from pysrc.ops.caching import ttl_cache
from pysrc.ops.mm_logkit import get_logger
from pysrc.pipeline.core.pipeline_core_base import PipelineStep

from .transforms import build_steps  # Shared generic builder

logger = get_logger(__name__)


class AnonymizationConfig(BaseModel):
    enabled: bool = Field(default=True)
    sensitive_columns: list[str] = Field(default_factory=list)
    hash_algorithm: str = Field(default="sha256")
    salt: str | None = Field(default=None)


def get_hash_func(algo: str, salt: str | None = None) -> Callable[[str], str]:
    hash_map = {
        "sha256": hashlib.sha256,
        "md5": hashlib.md5,
    }
    if algo not in hash_map:
        raise ValueError(f"Unsupported hash algorithm: {algo}")

    def hasher(value: str) -> str:
        salted = f"{salt}{value}" if salt else value
        return hash_map[algo](salted.encode()).hexdigest()

    return hasher


class DataAnonymizationStep(PipelineStep):
    def __init__(self, config: AnonymizationConfig):
        self.config = config
        self.hasher = get_hash_func(self.config.hash_algorithm, self.config.salt)

    def apply(self, lf: LazyFrame) -> LazyFrame:
        if not self.config.enabled or not self.config.sensitive_columns:
            return lf

        schema = lf.collect_schema()
        missing = set(self.config.sensitive_columns) - set(schema.names())
        if missing:
            raise DataValidationError(f"Missing sensitive columns: {missing}")

        exprs = [
            pl.col(col).cast(pl.Utf8).map_elements(self.hasher, return_dtype=pl.Utf8).alias(col)
            for col in self.config.sensitive_columns
        ]
        try:
            lf = lf.with_columns(exprs)
            logger.info(f"Anonymized columns: {self.config.sensitive_columns}")
            return lf
        except Exception as e:
            logger.error(f"Anonymization failed: {e}")
            raise DataValidationError(f"Anonymization failed: {e}") from e


class RegulatoryFilterConfig(BaseModel):
    enabled: bool = Field(default=True)
    restricted_symbols: list[str] = Field(default_factory=list)
    min_volume_threshold: float | None = Field(default=None)


class RegulatoryFilterStep(PipelineStep):
    def __init__(self, config: RegulatoryFilterConfig):
        self.config = config

    def apply(self, lf: LazyFrame) -> LazyFrame:
        if not self.config.enabled:
            return lf

        filters: list[pl.Expr] = []
        if self.config.restricted_symbols:
            filters.append(~pl.col("symbol").is_in(self.config.restricted_symbols))

        if self.config.min_volume_threshold is not None:
            filters.append(pl.col("volume") >= self.config.min_volume_threshold)

        if filters:
            combined_filter = reduce(operator.and_, filters)
            try:
                lf = lf.filter(combined_filter)
                logger.info("Applied regulatory filters")
                return lf
            except Exception as e:
                logger.error(f"Regulatory filter failed: {e}")
                raise DataValidationError(f"Regulatory filter failed: {e}") from e
        return lf


COMPLIANCE_STEPS: dict[str, type[PipelineStep]] = {
    "anonymization": DataAnonymizationStep,
    "regulatory_filter": RegulatoryFilterStep,
}

COMPLIANCE_CONFIGS: dict[str, type[BaseModel]] = {
    "anonymization": AnonymizationConfig,
    "regulatory_filter": RegulatoryFilterConfig,
}


@ttl_cache(ttl=3600)
def build_compliance_steps(configs: list[dict]) -> list[PipelineStep]:
    return build_steps(configs, COMPLIANCE_STEPS, COMPLIANCE_CONFIGS)

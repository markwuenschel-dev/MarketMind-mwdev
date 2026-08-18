# py/pipeline/stages/market_data/transforms.py

import polars as pl
from polars import LazyFrame
from pydantic import BaseModel, ConfigDict

from pysrc.core.errors import DataValidationError
from pysrc.ops.caching import ttl_cache
from pysrc.ops.mm_logkit import get_logger
from pysrc.pipeline.core.pipeline_core_base import PipelineStep

logger = get_logger(__name__)


class ColumnRenameConfig(BaseModel):
    mapping: dict[str, str]


class ColumnRenameStep(PipelineStep):
    def __init__(self, config: ColumnRenameConfig):
        self.mapping = config.mapping

    def apply(self, lf: LazyFrame) -> LazyFrame:
        schema = lf.collect_schema()
        missing = set(self.mapping) - set(schema.names())
        if missing:
            raise DataValidationError(f"Missing columns for rename: {missing}")
        try:
            lf = lf.rename(self.mapping)
            logger.info(f"Renamed columns: {list(self.mapping.keys())}")
            return lf
        except Exception as e:
            logger.error(f"Column rename failed: {e}")
            raise DataValidationError(f"Column rename failed: {e}") from e


class TypeCastConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    dtypes: dict[str, pl.DataType]


class TypeCastStep(PipelineStep):
    def __init__(self, config: TypeCastConfig):
        self.dtypes = config.dtypes

    def apply(self, lf: LazyFrame) -> LazyFrame:
        schema = lf.collect_schema()
        missing = set(self.dtypes) - set(schema.names())
        if missing:
            raise DataValidationError(f"Missing columns for cast: {missing}")
        try:
            lf = lf.cast(self.dtypes)
            logger.info(f"Cast types for columns: {list(self.dtypes.keys())}")
            return lf
        except Exception as e:
            logger.error(f"Type cast failed: {e}")
            raise DataValidationError(f"Type cast failed: {e}") from e


TRANSFORM_STEPS: dict[str, type[PipelineStep]] = {
    "rename": ColumnRenameStep,
    "cast": TypeCastStep,
}

TRANSFORM_CONFIGS: dict[str, type[BaseModel]] = {
    "rename": ColumnRenameConfig,
    "cast": TypeCastConfig,
}


@ttl_cache(ttl=3600)
def build_steps(
    configs: list[dict],
    step_registry: dict[str, type[PipelineStep]],
    config_registry: dict[str, type[BaseModel]],
) -> list[PipelineStep]:
    steps = []
    for cfg_dict in configs:
        step_type = cfg_dict.get("type")
        if step_type not in step_registry:
            raise ValueError(f"Unknown step type: {step_type}")
        cfg_class = config_registry[step_type]
        cfg = cfg_class(**cfg_dict)
        steps.append(step_registry[step_type](cfg))
    return steps


def build_transform_steps(configs: list[dict]) -> list[PipelineStep]:
    return build_steps(configs, TRANSFORM_STEPS, TRANSFORM_CONFIGS)

from __future__ import annotations

import polars as pl
from pydantic import BaseModel, ConfigDict

from pysrc.core.errors import DataValidationError
from pysrc.pipeline.stages.cleaning.core.base import CleaningStep
from pysrc.pipeline.stages.cleaning.core.contracts import (
    CleaningPipelineState,
    CleaningRuntimeContext,
)
from pysrc.pipeline.stages.cleaning.core.providers import GovernedColumns
from pysrc.pipeline.stages.cleaning.core.registry import register_cleaning_step


class MacroFeatureParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_key: str = "macro"
    output_columns: tuple[str, ...] = ()


@register_cleaning_step(
    step_type="feature.macro",
    version="1",
    params_model=MacroFeatureParams,
    provider_requirements=("macro",),
)
class EconomicIndicatorNormalizerStep(CleaningStep):
    def _apply(
        self,
        df: pl.DataFrame,
        state: CleaningPipelineState,
        context: CleaningRuntimeContext,
    ):
        provider = context.provider(self.params.provider_key)
        governed_columns: GovernedColumns = provider.materialize(
            df,
            context=context,
            params=self.params,
        )
        if governed_columns.frame.height != df.height:
            raise DataValidationError(
                "Macro provider must return row-aligned governed columns",
                details={"provider": self.params.provider_key},
            )
        lineage = dict(governed_columns.lineage)
        if context.governance_mode.value == "governed" and "pit_identity" not in lineage:
            raise DataValidationError(
                "Governed macro providers must emit PIT lineage",
                details={"provider": self.params.provider_key, "lineage": lineage},
            )
        columns_frame = governed_columns.frame
        if self.params.output_columns:
            columns_frame = columns_frame.select(list(self.params.output_columns))
        cleaned = df.hstack(columns_frame)
        return self._result(
            cleaned,
            state,
            warnings=list(governed_columns.warnings),
            provider_lineage={self.params.provider_key: lineage},
            metrics={
                "provider": self.params.provider_key,
                "output_columns": list(columns_frame.columns),
            },
            mutation=self._additive_mutation(
                df.height,
                cleaned.height,
                added_columns=columns_frame.width,
            ),
        )

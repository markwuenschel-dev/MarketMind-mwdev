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


class AlternativeDataParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_key: str = "altdata"
    output_columns: tuple[str, ...] = ()


@register_cleaning_step(
    step_type="feature.altdata",
    version="1",
    params_model=AlternativeDataParams,
    provider_requirements=("altdata",),
)
class AlternativeDataNormalizerStep(CleaningStep):
    def _apply(
        self,
        df: pl.DataFrame,
        state: CleaningPipelineState,
        context: CleaningRuntimeContext,
    ):
        provider = context.provider(self.params.provider_key)
        try:
            governed_columns: GovernedColumns = provider.materialize(
                df,
                context=context,
                params=self.params,
            )
        except Exception as exc:
            fallback_policy = dict(self.spec.fallback_policy)
            if fallback_policy.get("mode") != "constant":
                raise
            values = fallback_policy.get("values", {})
            fallback_frame = pl.DataFrame(
                {
                    str(column): [values[column]] * df.height
                    for column in self.params.output_columns
                    if column in values
                }
            )
            if fallback_frame.width == 0:
                raise DataValidationError(
                    "Explicit altdata fallback requires constant values for output columns",
                    details={"output_columns": list(self.params.output_columns)},
                ) from exc
            cleaned = df.hstack(fallback_frame)
            return self._result(
                cleaned,
                state,
                warnings=[f"explicit_fallback:{self.params.provider_key}"],
                fallback_events=[{"provider": self.params.provider_key, "mode": "constant"}],
                metrics={"provider": self.params.provider_key, "fallback": True},
                mutation=self._additive_mutation(
                    df.height,
                    cleaned.height,
                    added_columns=fallback_frame.width,
                ),
            )
        if governed_columns.frame.height != df.height:
            raise DataValidationError("Alternative-data provider must return row-aligned columns")
        lineage = dict(governed_columns.lineage)
        if context.governance_mode.value == "governed" and "pit_identity" not in lineage:
            raise DataValidationError(
                "Governed alternative-data providers must emit PIT lineage",
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

from __future__ import annotations

from typing import Literal

import polars as pl
from pydantic import BaseModel, ConfigDict

from pysrc.core.errors import DataValidationError
from pysrc.pipeline.stages.cleaning.core.base import CleaningStep
from pysrc.pipeline.stages.cleaning.core.contracts import (
    CleaningPipelineState,
    CleaningRuntimeContext,
)
from pysrc.pipeline.stages.cleaning.core.registry import register_cleaning_step


class DenoiseParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    method: Literal["ewm", "minmax"] = "ewm"
    span: int = 5


@register_cleaning_step(
    step_type="impute.denoise",
    version="1",
    params_model=DenoiseParams,
)
class DenoiseNormalizerStep(CleaningStep):
    def _apply(
        self,
        df: pl.DataFrame,
        state: CleaningPipelineState,
        context: CleaningRuntimeContext,
    ):
        del context
        numeric_columns = df.select(pl.selectors.numeric()).columns
        if not numeric_columns:
            return self._result(
                df,
                state,
                metrics={"denoised_columns": 0},
                mutation=self._cell_mutation(df.height, df.height),
            )

        cleaned = df
        if self.params.method == "ewm":
            cleaned = cleaned.with_columns(
                [
                    pl.col(column).ewm_mean(span=self.params.span, adjust=False).alias(column)
                    for column in numeric_columns
                ]
            )
        elif self.params.method == "minmax":
            expressions = []
            for column in numeric_columns:
                column_min = cleaned[column].min()
                column_max = cleaned[column].max()
                if column_max == column_min:
                    continue
                expressions.append(
                    ((pl.col(column) - column_min) / (column_max - column_min)).alias(column)
                )
            cleaned = cleaned.with_columns(expressions) if expressions else cleaned
        else:
            raise DataValidationError(
                "Unsupported denoise method", details={"method": self.params.method}
            )

        return self._result(
            cleaned,
            state,
            metrics={"method": self.params.method, "denoised_columns": len(numeric_columns)},
            mutation=self._cell_mutation(
                df.height,
                cleaned.height,
                rows_with_mutations=cleaned.height if numeric_columns else 0,
                cells_mutated=cleaned.height * len(numeric_columns),
            ),
        )

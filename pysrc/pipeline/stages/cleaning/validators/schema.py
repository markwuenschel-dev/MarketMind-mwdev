from __future__ import annotations

import polars as pl
from pydantic import BaseModel, ConfigDict

from pysrc.core.errors import DataValidationError
from pysrc.core.validation import validate_dataframe, validate_ohlcv
from pysrc.pipeline.stages.cleaning.core.base import CleaningStep
from pysrc.pipeline.stages.cleaning.core.contracts import (
    CleaningPipelineState,
    CleaningRuntimeContext,
)
from pysrc.pipeline.stages.cleaning.core.registry import register_cleaning_step


class ValidationParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ohlcv_mode: bool = False
    strict: bool = True


@register_cleaning_step(
    step_type="validate.schema",
    version="1",
    params_model=ValidationParams,
)
class ValidationStep(CleaningStep):
    def _apply(
        self,
        df: pl.DataFrame,
        state: CleaningPipelineState,
        context: CleaningRuntimeContext,
    ):
        del context
        validate_dataframe(df)
        if self.params.ohlcv_mode:
            validate_ohlcv(df)
        price_columns = [
            column for column in ("open", "high", "low", "close") if column in df.columns
        ]
        if price_columns:
            has_negative = df.select(
                pl.any_horizontal([pl.col(column) < 0 for column in price_columns])
                .fill_null(False)
                .any()
            ).item()
            if has_negative:
                raise DataValidationError("Negative price values found during validation")
        return self._result(
            df,
            state,
            metrics={"validated_rows": df.height},
            mutation=self._cell_mutation(df.height, df.height),
        )

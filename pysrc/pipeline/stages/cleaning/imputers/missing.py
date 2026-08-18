from __future__ import annotations

from typing import Any, Literal

import polars as pl
from pydantic import BaseModel, ConfigDict

from pysrc.core.errors import DataValidationError
from pysrc.ops.mm_logkit import get_logger
from pysrc.pipeline.stages.cleaning.core.base import CleaningStep
from pysrc.pipeline.stages.cleaning.core.contracts import (
    CleaningPipelineState,
    CleaningRuntimeContext,
)
from pysrc.pipeline.stages.cleaning.core.registry import register_cleaning_step

logger = get_logger(__name__)

KalmanFilter = None


def _require_kalman() -> Any:
    global KalmanFilter
    if KalmanFilter is None:
        try:
            from pykalman import KalmanFilter as _KalmanFilter  # type: ignore[import-not-found]
        except ImportError as exc:
            raise DataValidationError(
                "Kalman imputation requires pykalman",
                details={"dependency": "pykalman"},
            ) from exc
        KalmanFilter = _KalmanFilter
    return KalmanFilter


class MissingValueParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    method: Literal["forward_fill", "backward_fill", "interpolate", "median", "kalman"] = (
        "forward_fill"
    )
    backward_fill: bool = False


@register_cleaning_step(
    step_type="impute.missing",
    version="1",
    params_model=MissingValueParams,
)
class MissingValueNormalizerStep(CleaningStep):
    def _apply(
        self,
        df: pl.DataFrame,
        state: CleaningPipelineState,
        context: CleaningRuntimeContext,
    ):
        del context
        if df.width == 0:
            return self._result(
                df,
                state,
                metrics={"values_imputed": 0},
                mutation=self._cell_mutation(df.height, df.height),
            )

        before = int(sum(df.null_count().row(0)))
        method = self.params.method
        if method == "forward_fill":
            cleaned = df.with_columns(pl.all().forward_fill())
            if self.params.backward_fill:
                cleaned = cleaned.with_columns(pl.all().backward_fill())
        elif method == "backward_fill":
            cleaned = df.with_columns(pl.all().backward_fill())
        elif method == "interpolate":
            cleaned = df.with_columns(pl.all().interpolate())
        elif method == "median":
            numeric_columns = df.select(pl.selectors.numeric()).columns
            expressions = [
                pl.col(column).fill_null(df[column].median()).alias(column)
                for column in numeric_columns
            ]
            cleaned = df.with_columns(expressions) if expressions else df
        elif method == "kalman":
            numeric_columns = df.select(pl.selectors.numeric()).columns
            cleaned = df
            kalman_filter = _require_kalman()
            for column in numeric_columns:
                series = cleaned[column].to_numpy()
                non_null = cleaned[column].drop_nulls().to_numpy()
                if len(non_null) == 0:
                    continue
                kf = kalman_filter(initial_state_mean=float(non_null[0]), n_dim_obs=1)
                kf = kf.em(non_null)
                imputed = kf.smooth(series)[0].flatten()
                cleaned = cleaned.with_columns(pl.Series(column, imputed))
        else:
            raise DataValidationError(
                "Unsupported missing value method", details={"method": method}
            )

        after = int(sum(cleaned.null_count().row(0)))
        values_imputed = before - after
        rows_with_nulls_before = int(
            df.select(pl.any_horizontal(pl.all().is_null()).alias("has_null"))
            .get_column("has_null")
            .sum()
        )
        return self._result(
            cleaned,
            state,
            metrics={
                "method": method,
                "nulls_before": before,
                "nulls_after": after,
                "values_imputed": values_imputed,
            },
            mutation=self._cell_mutation(
                df.height,
                cleaned.height,
                rows_with_mutations=rows_with_nulls_before,
                cells_mutated=max(values_imputed, 0),
            ),
        )

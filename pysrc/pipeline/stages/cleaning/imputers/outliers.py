from __future__ import annotations

from typing import Literal

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict

from pysrc.core.errors import DataValidationError
from pysrc.pipeline.stages.cleaning.core.base import CleaningStep
from pysrc.pipeline.stages.cleaning.core.contracts import (
    CleaningPipelineState,
    CleaningRuntimeContext,
)
from pysrc.pipeline.stages.cleaning.core.registry import register_cleaning_step

try:
    import numba as nb
except ImportError:  # pragma: no cover - exercised in dependency-light envs

    class _NumbaCompat:
        @staticmethod
        def njit(*args, **kwargs):
            del args, kwargs

            def _decorator(func):
                return func

            return _decorator

        prange = range

    nb = _NumbaCompat()


@nb.njit(parallel=True)
def _robust_zscore_mask(values: np.ndarray, threshold: float) -> np.ndarray:
    rows, columns = values.shape
    out = np.zeros((rows, columns), dtype=np.bool_)
    for column_index in nb.prange(columns):
        column = values[:, column_index]
        median = np.nanmedian(column)
        mad = np.nanmedian(np.abs(column - median))
        if mad == 0:
            continue
        robust_zscore = 0.6745 * np.abs(column - median) / mad
        out[:, column_index] = robust_zscore > threshold
    return out


class OutlierParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    method: Literal["zscore", "iqr"] = "zscore"
    threshold: float = 3.0
    factor: float = 1.5


@register_cleaning_step(
    step_type="impute.outliers",
    version="1",
    params_model=OutlierParams,
)
class OutlierNormalizerStep(CleaningStep):
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
                metrics={"outlier_rows": 0},
                mutation=self._cell_mutation(df.height, df.height),
            )

        numeric_values = df.select(numeric_columns).to_numpy()
        if self.params.method == "zscore":
            mask = _robust_zscore_mask(numeric_values, float(self.params.threshold))
        elif self.params.method == "iqr":
            q1 = np.nanquantile(numeric_values, 0.25, axis=0)
            q3 = np.nanquantile(numeric_values, 0.75, axis=0)
            iqr = q3 - q1
            lower = q1 - float(self.params.factor) * iqr
            upper = q3 + float(self.params.factor) * iqr
            mask = (numeric_values < lower) | (numeric_values > upper)
        else:
            raise DataValidationError(
                "Unsupported outlier method", details={"method": self.params.method}
            )

        cleaned = df
        for index, column in enumerate(numeric_columns):
            column_mask = pl.Series(f"{column}_mask", mask[:, index].tolist())
            cleaned = cleaned.with_columns(
                pl.when(column_mask).then(None).otherwise(pl.col(column)).alias(column)
            ).with_columns(pl.col(column).forward_fill())

        outlier_rows = int(mask.any(axis=1).sum())
        return self._result(
            cleaned,
            state,
            metrics={
                "method": self.params.method,
                "outlier_rows": outlier_rows,
                "outlier_values": int(mask.sum()),
            },
            mutation=self._cell_mutation(
                df.height,
                cleaned.height,
                rows_with_mutations=outlier_rows,
                cells_mutated=int(mask.sum()),
            ),
        )

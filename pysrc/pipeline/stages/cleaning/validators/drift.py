from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict

from pysrc.core.errors import DataDriftError
from pysrc.core.runtime.optional_imports import require_attr
from pysrc.pipeline.stages.cleaning.core.base import CleaningStep
from pysrc.pipeline.stages.cleaning.core.contracts import (
    CleaningPipelineState,
    CleaningRuntimeContext,
)
from pysrc.pipeline.stages.cleaning.core.registry import register_cleaning_step


class BaseDriftTest:
    @abstractmethod
    def compute(self, current: np.ndarray, reference: np.ndarray) -> tuple[float, float]: ...


class KSTest(BaseDriftTest):
    def compute(self, current: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
        ks_2samp = require_attr("scipy.stats", "ks_2samp", purpose="governed drift detection")
        return ks_2samp(current, reference)


class DriftDetectionParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    enabled: bool = True
    threshold: float = 0.05
    columns: tuple[str, ...] = ()
    strict: bool = True
    reference_frame: Any | None = None


@register_cleaning_step(
    step_type="validate.drift",
    version="1",
    params_model=DriftDetectionParams,
)
class DriftDetectionStep(CleaningStep):
    def _apply(
        self,
        df: pl.DataFrame,
        state: CleaningPipelineState,
        context: CleaningRuntimeContext,
    ):
        del context
        if not self.params.enabled or self.params.reference_frame is None:
            return self._result(
                df,
                state,
                metrics={"drift_columns": []},
                mutation=self._cell_mutation(df.height, df.height),
            )

        reference_frame = self.params.reference_frame
        if not isinstance(reference_frame, pl.DataFrame):
            raise DataDriftError("Drift reference frame must be a Polars DataFrame")

        columns = list(self.params.columns) or df.select(pl.selectors.numeric()).columns
        drift_columns: list[str] = []
        for column in columns:
            if column not in df.columns or column not in reference_frame.columns:
                continue
            current = df[column].drop_nulls().to_numpy()
            reference = reference_frame[column].drop_nulls().to_numpy()
            if len(current) == 0 or len(reference) == 0:
                continue
            _, p_value = KSTest().compute(current, reference)
            if p_value < self.params.threshold:
                drift_columns.append(column)
        if drift_columns and self.params.strict:
            raise DataDriftError(
                "Data drift detected in governed cleaning path",
                details={"columns": drift_columns, "threshold": self.params.threshold},
            )
        warnings = []
        if drift_columns:
            warnings.append(f"drift_detected:{','.join(drift_columns)}")
        return self._result(
            df,
            state,
            warnings=warnings,
            metrics={"drift_columns": drift_columns},
            mutation=self._cell_mutation(df.height, df.height),
        )

from __future__ import annotations

import polars as pl
from pydantic import BaseModel, ConfigDict

from pysrc.core.errors import DataValidationError
from pysrc.pipeline.stages.cleaning.core.base import CleaningStep
from pysrc.pipeline.stages.cleaning.core.contracts import (
    CleaningPipelineState,
    CleaningRuntimeContext,
)
from pysrc.pipeline.stages.cleaning.core.registry import register_cleaning_step


class BatchAnomalyParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    contamination: float = 0.1


@register_cleaning_step(
    step_type="anomaly.batch",
    version="1",
    params_model=BatchAnomalyParams,
)
class AnomalyNormalizerStep(CleaningStep):
    def _apply(
        self,
        df: pl.DataFrame,
        state: CleaningPipelineState,
        context: CleaningRuntimeContext,
    ):
        numeric_columns = df.select(pl.selectors.numeric()).columns
        if not numeric_columns or df.height < 2:
            return self._result(
                df,
                state,
                metrics={"anomaly_rows": 0},
                mutation=self._cell_mutation(df.height, df.height),
            )
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError as exc:
            raise DataValidationError("Batch anomaly detection requires scikit-learn") from exc
        model = IsolationForest(
            contamination=self.params.contamination,
            random_state=context.seed_for(self.spec.step_id),
        )
        mask = model.fit_predict(df.select(numeric_columns).to_numpy()) == 1
        if not mask.any():
            return self._result(
                df,
                state,
                warnings=["all_rows_flagged_as_anomaly"],
                metrics={"anomaly_rows": df.height},
                mutation=self._cell_mutation(df.height, df.height),
            )
        cleaned = df.filter(pl.Series("anomaly_mask", mask.tolist()))
        removed_rows = df.height - cleaned.height
        return self._result(
            cleaned,
            state,
            metrics={"anomaly_rows": removed_rows, "contamination": self.params.contamination},
            mutation=self._cell_mutation(
                df.height,
                cleaned.height,
                rows_with_mutations=removed_rows,
                cells_mutated=removed_rows * len(numeric_columns),
            ),
        )

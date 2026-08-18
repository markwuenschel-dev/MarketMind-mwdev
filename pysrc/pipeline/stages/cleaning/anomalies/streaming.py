from __future__ import annotations

from collections import deque

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


class StreamingIsolationForest:
    def __init__(
        self,
        contamination: float,
        refit_every: int,
        window_size: int = 1000,
        *,
        random_state: int = 42,
    ):
        self.contamination = contamination
        self.refit_every = refit_every
        self.window_size = window_size
        self.random_state = random_state
        self.buffer = deque(maxlen=window_size)
        self.counter = 0
        self._since_refit = 0
        self.model = None
        self._fitted = False

    def predict(self, df: pl.DataFrame) -> np.ndarray:
        numeric_columns = df.select(pl.selectors.numeric()).columns
        if not numeric_columns or df.height == 0:
            return np.ones(df.height, dtype=bool)
        try:
            from sklearn.ensemble import IsolationForest
        except ImportError as exc:
            raise DataValidationError("Streaming anomaly detection requires scikit-learn") from exc

        values = df.select(numeric_columns).to_numpy()
        for row in values:
            self.buffer.append(row)
            self.counter += 1
            self._since_refit += 1
        if self._since_refit >= self.refit_every and len(self.buffer) >= 2:
            self.model = IsolationForest(
                contamination=self.contamination,
                random_state=self.random_state,
                n_estimators=50,
            )
            self.model.fit(np.array(list(self.buffer)))
            self._fitted = True
            self._since_refit = 0
        if not self._fitted or self.model is None:
            return np.ones(df.height, dtype=bool)
        return (self.model.predict(values) == 1).astype(bool)


class StreamingAnomalyParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    contamination: float = 0.1
    refit_every: int = 1000
    window_size: int = 1000


@register_cleaning_step(
    step_type="anomaly.streaming",
    version="1",
    params_model=StreamingAnomalyParams,
    stateful=True,
)
class StreamingAnomalyNormalizerStep(CleaningStep):
    def _apply(
        self,
        df: pl.DataFrame,
        state: CleaningPipelineState,
        context: CleaningRuntimeContext,
    ):
        detector = state.step_state.get(self.spec.step_id)
        if detector is None:
            detector = StreamingIsolationForest(
                contamination=self.params.contamination,
                refit_every=self.params.refit_every,
                window_size=self.params.window_size,
                random_state=context.seed_for(self.spec.step_id),
            )
            state.step_state[self.spec.step_id] = detector
        mask = detector.predict(df)
        cleaned = df.filter(pl.Series("stream_anomaly_mask", mask.tolist())) if mask.any() else df
        removed_rows = df.height - cleaned.height
        return self._result(
            cleaned,
            state,
            metrics={"anomaly_rows": removed_rows, "streaming": True},
            mutation=self._cell_mutation(
                df.height,
                cleaned.height,
                rows_with_mutations=removed_rows,
                cells_mutated=removed_rows * len(df.select(pl.selectors.numeric()).columns),
            ),
        )

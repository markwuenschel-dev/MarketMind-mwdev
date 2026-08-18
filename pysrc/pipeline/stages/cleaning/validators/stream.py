from __future__ import annotations

import polars as pl
from pydantic import BaseModel, ConfigDict

from pysrc.core.validation import validate_stream_chunk
from pysrc.pipeline.stages.cleaning.core.base import CleaningStep
from pysrc.pipeline.stages.cleaning.core.contracts import (
    CleaningPipelineState,
    CleaningRuntimeContext,
)
from pysrc.pipeline.stages.cleaning.core.registry import register_cleaning_step


class StreamValidationParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    strict: bool = True
    sampling_rate: float = 1.0


@register_cleaning_step(
    step_type="validate.stream",
    version="1",
    params_model=StreamValidationParams,
    stateful=True,
)
class StreamValidationStep(CleaningStep):
    def _apply(
        self,
        df: pl.DataFrame,
        state: CleaningPipelineState,
        context: CleaningRuntimeContext,
    ):
        del context
        counter = int(state.step_state.get(self.spec.step_id, 0))
        state.step_state[self.spec.step_id] = counter + 1
        should_validate = True
        if self.params.sampling_rate < 1.0:
            interval = max(int(round(1.0 / self.params.sampling_rate)), 1)
            should_validate = (counter % interval) == 0
        if should_validate:
            validate_stream_chunk(df)
        return self._result(
            df,
            state,
            metrics={"validated": should_validate, "stream_chunk_rows": df.height},
            mutation=self._cell_mutation(df.height, df.height),
        )

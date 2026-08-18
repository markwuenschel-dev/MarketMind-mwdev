from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pysrc.pipeline.stages.cleaning.core.contracts import (
    BuiltCleaningPipeline,
    CleaningPipelineSpec,
    CleaningRuntimeContext,
    CleaningStepResult,
)
from pysrc.pipeline.stages.cleaning.core.factory import build_cleaning_pipeline
from pysrc.pipeline.stages.cleaning.execution.runtime import CleaningPipelineRunner


class CleanerPipeline:
    def __init__(
        self,
        pipeline: BuiltCleaningPipeline | CleaningPipelineSpec | Mapping[str, Any],
    ) -> None:
        self.pipeline = (
            pipeline
            if isinstance(pipeline, BuiltCleaningPipeline)
            else build_cleaning_pipeline(pipeline)
        )
        self.runner = CleaningPipelineRunner(self.pipeline)

    def run(
        self,
        df: Any,
        *,
        context: CleaningRuntimeContext | None = None,
        return_result: bool = False,
    ) -> Any:
        result = self.runner.run(df, context=context)
        if return_result:
            return result
        return result.frame

    @property
    def last_result(self) -> CleaningStepResult | None:
        return self.runner.last_result

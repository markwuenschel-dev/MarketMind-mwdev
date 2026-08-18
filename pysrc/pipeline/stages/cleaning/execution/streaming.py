from __future__ import annotations

import dataclasses
from collections import deque
from collections.abc import AsyncGenerator, Mapping
from typing import Any

from pysrc.core.runtime.optional_imports import optional_import

pl = optional_import("polars")
from pysrc.pipeline.stages.cleaning.core.contracts import (
    BuiltCleaningPipeline,
    CleaningPipelineSpec,
    CleaningRuntimeContext,
)
from pysrc.pipeline.stages.cleaning.core.factory import build_cleaning_pipeline
from pysrc.pipeline.stages.cleaning.execution.runtime import (
    CleaningPipelineRunner,
    _to_polars,
)


class StreamingCleanerPipeline:
    def __init__(
        self,
        pipeline: BuiltCleaningPipeline | CleaningPipelineSpec | Mapping[str, Any],
        *,
        buffer_size: int = 1,
        window: int | None = None,
    ) -> None:
        self.pipeline = (
            pipeline
            if isinstance(pipeline, BuiltCleaningPipeline)
            else build_cleaning_pipeline(pipeline)
        )
        self.runner = CleaningPipelineRunner(self.pipeline)
        maxlen = window if window is not None else max(int(buffer_size), 1)
        self.buffer = deque(maxlen=maxlen)
        self.buffer_size = max(int(buffer_size), 1)

    async def process_stream(
        self,
        stream_gen: AsyncGenerator[Any, None],
        *,
        context: CleaningRuntimeContext | None = None,
        return_result: bool = False,
    ) -> AsyncGenerator[Any, None]:
        streaming_context = context or self.runner.default_context(streaming=True)
        if not streaming_context.streaming:
            streaming_context = dataclasses.replace(streaming_context, streaming=True)

        async for chunk in stream_gen:
            self.buffer.append(_to_polars(chunk))
            if len(self.buffer) < self.buffer_size:
                continue
            yield self._flush(streaming_context, return_result=return_result)

        if self.buffer:
            yield self._flush(streaming_context, return_result=return_result)

    def _flush(
        self,
        context: CleaningRuntimeContext,
        *,
        return_result: bool,
    ) -> Any:
        if pl is None:
            raise RuntimeError("polars is required for streaming cleaning execution")
        frame = pl.concat(list(self.buffer), rechunk=True)
        self.buffer.clear()
        result = self.runner.run(frame, context=context)
        if return_result:
            return result
        return result.frame

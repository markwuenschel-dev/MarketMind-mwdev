from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any

import numpy as np
import polars as pl
import pytest

from pysrc.pipeline.core.pipeline_core_base import PipelineStep
from pysrc.pipeline.core.pipeline_core_context import PipelineContext
from pysrc.pipeline.execution.streaming import _SENTINEL, StreamingPipeline
from pysrc.pipeline.stages.cleaning import (
    CleaningRuntimeContext,
    build_cleaning_pipeline,
)
from pysrc.pipeline.stages.cleaning.anomalies.streaming import StreamingIsolationForest
from pysrc.pipeline.stages.cleaning.execution import StreamingCleanerPipeline

pytestmark = pytest.mark.determinism("d1")


class MockStep(PipelineStep):
    def __init__(self, name: str) -> None:
        super().__init__(name=name)
        self.call_count = 0

    async def execute(self, data: Any, context: PipelineContext) -> Any:
        del context
        self.call_count += 1
        if isinstance(data, pl.DataFrame):
            return data.with_columns(pl.lit(self.call_count).alias(f"step_{self.name}"))
        return data


async def _frame_stream(rows: list[dict[str, Any]]) -> AsyncGenerator[pl.DataFrame, None]:
    for row in rows:
        yield pl.DataFrame([row])
        await asyncio.sleep(0)


def _cleaning_streaming_pipeline() -> StreamingCleanerPipeline:
    pipeline = build_cleaning_pipeline(
        {
            "seed_lineage": "tests.streaming",
            "steps": [
                {
                    "step_id": "feature.technical.rsi",
                    "step_type": "feature.technical.rsi",
                    "version": "1",
                    "params": {
                        "window": 3,
                        "close_column": "close",
                        "output_column": "rsi",
                        "fillna_method": "ffill",
                    },
                },
                {
                    "step_id": "feature.technical.macd",
                    "step_type": "feature.technical.macd",
                    "version": "1",
                    "params": {
                        "fast": 2,
                        "slow": 4,
                        "signal": 2,
                        "close_column": "close",
                        "macd_column": "macd",
                        "signal_column": "macd_signal",
                    },
                },
            ],
        }
    )
    return StreamingCleanerPipeline(pipeline, buffer_size=1)


@pytest.mark.asyncio
async def test_streaming_pipeline_batches_without_exposing_sentinel(deterministic_seed: int):
    _ = deterministic_seed
    steps = [MockStep("alpha"), MockStep("beta")]
    pipeline = StreamingPipeline(steps, {"queue_size": 8, "batch_size": 2})
    context = PipelineContext(frequency="min")

    async def source() -> AsyncGenerator[dict[str, int], None]:
        for idx in range(5):
            yield {"id": idx, "value": idx}
            await asyncio.sleep(0)

    results = []
    async for result in pipeline.run(source(), context):
        results.append(result)
        assert result is not _SENTINEL

    assert len(results) == 3
    assert steps[0].call_count == 3
    assert steps[1].call_count == 3


@pytest.mark.asyncio
async def test_cleaning_streaming_pipeline_preserves_indicator_state(deterministic_seed: int):
    _ = deterministic_seed
    pipeline = _cleaning_streaming_pipeline()
    context = CleaningRuntimeContext(
        run_id="stream-test",
        determinism_tier=pipeline.pipeline.spec.determinism_tier,
        seed_lineage=pipeline.pipeline.spec.seed_lineage,
        pit_boundary="2026-04-08",
        governance_mode=pipeline.pipeline.spec.governance_mode,
        providers={},
        streaming=True,
        registry_state_hash=pipeline.pipeline.registry_state_hash,
    )
    rows = [
        {"timestamp": "2026-01-01T00:00:00", "close": 100.0},
        {"timestamp": "2026-01-01T00:01:00", "close": 101.0},
        {"timestamp": "2026-01-01T00:02:00", "close": 102.0},
        {"timestamp": "2026-01-01T00:03:00", "close": 103.5},
        {"timestamp": "2026-01-01T00:04:00", "close": 102.5},
    ]

    outputs = []
    async for batch in pipeline.process_stream(_frame_stream(rows), context=context):
        outputs.append(batch)

    assert len(outputs) == len(rows)
    assert outputs[-1].columns == ["timestamp", "close", "rsi", "macd", "macd_signal"]
    assert outputs[-1]["macd"].item() != 0.0


@pytest.mark.determinism("d2")
def test_streaming_isolation_forest_buffer_and_shape(deterministic_seed: int):
    _ = deterministic_seed
    forest = StreamingIsolationForest(
        contamination=0.1,
        refit_every=4,
        window_size=6,
        random_state=17,
    )
    data = np.arange(15, dtype=float).reshape(5, 3)
    frame = pl.DataFrame(data, schema=["a", "b", "c"], orient="row")

    predictions = forest.predict(frame)

    assert predictions.shape == (5,)
    assert predictions.dtype == bool
    assert len(forest.buffer) <= forest.window_size
    assert forest.counter == 5

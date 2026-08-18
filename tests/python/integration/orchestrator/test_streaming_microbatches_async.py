# tests/python/integration/orchestrator/test_streaming_microbatches_async.py
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

pytest_plugins = ("tests.python.plugins.torture_plugin",)


@pytest.mark.streaming
@pytest.mark.asyncio
async def test_micro_batches_stream(tmp_path_factory):
    from tests.python.plugins.torture_plugin import _auto_data_dir

    data_dir = _auto_data_dir(Path(__file__))
    stream_dir = data_dir / "micro_batches_stream"
    if not stream_dir.exists():
        pytest.skip("No micro_batches_stream/ directory found")

    # Example: pretend we iterate files and ensure recovery on poison
    # Replace with your real streaming engine
    batches = sorted(stream_dir.glob("*.csv"))
    poisoned = any("poison" in p.name for p in batches)
    # simulate: if poison exists, engine should keep going and report it
    await asyncio.sleep(0)  # placeholder async path
    assert batches, "No micro-batches found"
    if poisoned:
        # placeholder check — wire to your engine’s telemetry
        assert poisoned is True

"""Verify marketmind.preproc_steps entry points resolve without import errors."""

from __future__ import annotations

import pytest


@pytest.mark.determinism("d1")
def test_research_indicator_engine_entry_point_loads(deterministic_seed: int) -> None:
    _ = deterministic_seed
    from importlib.metadata import entry_points

    from pysrc.pipeline.stages.preprocessing import StepFactory

    eps = {ep.name: ep for ep in entry_points(group="marketmind.preproc_steps")}
    cls_obj = eps["IndicatorEngineStep"].load()
    assert cls_obj.__name__ == "IndicatorEngineStep"
    assert StepFactory.get("indicator_engine").__name__ == "IndicatorEngineStep"

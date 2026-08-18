from __future__ import annotations

import pytest


@pytest.mark.determinism("d1")
def test_preprocessing_step_factory_import_is_lazy(deterministic_seed: int) -> None:
    _ = deterministic_seed

    from pysrc.pipeline.stages.preprocessing import AVAILABLE_STEPS, StepFactory

    assert "indicator_engine" in AVAILABLE_STEPS
    assert StepFactory.get("indicator_engine").__name__ == "IndicatorEngineStep"


@pytest.mark.determinism("d1")
def test_preprocessing_step_factory_create_passes_keyword_params(deterministic_seed: int) -> None:
    _ = deterministic_seed

    from pysrc.pipeline.stages.preprocessing import StepFactory

    step = StepFactory.create("indicator_engine", {"workers": 1})

    assert step.cfg == {"workers": 1}

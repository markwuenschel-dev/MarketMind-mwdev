"""
Contract tests for the canonical pysrc.tuning surface.

The old pysrc.autotune.api.AutoTuner has been deleted as part of the tuning
consolidation (see tuning surface migration).  This file now tests that the
canonical tune() facade satisfies the same behavioural contract: the tuner
must find parameters whose score is no worse than the baseline.

Migration: pysrc.autotune.api -> pysrc.tuning (tune, tune_objective)
"""

from __future__ import annotations

import pytest

from pysrc.tuning import TuningResult, tune

pytestmark = pytest.mark.contract


@pytest.mark.determinism("d1")
def test_tune_finds_non_worse_params() -> None:
    """tune() with random engine must find params no worse than a fixed baseline."""

    # Simple objective: score is the value of 'x'.  Baseline score at x=0.
    def objective(params: dict) -> float:
        return float(params["x"])

    baseline_score = objective({"x": 0})
    result: TuningResult = tune(
        objective,
        {"x": list(range(10))},
        engine="random",
        direction="maximize",
        budget=10,
        seed=42,
    )
    assert result.best_score >= baseline_score

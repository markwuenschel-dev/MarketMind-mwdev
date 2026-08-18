"""MarketMind canonical tuning subsystem.

All external callers should import from this package.  Do not import from
sub-modules directly — the sub-module layout is an implementation detail.

Quick start::

    from pysrc.tuning import tune, TunerSpec, TuningResult

    result = tune(
        my_objective,
        {"lr": (1e-4, 1e-1), "depth": [3, 5, 7]},
        engine="random",
        direction="maximize",
        budget=20,
        seed=42,
    )
"""

from __future__ import annotations

from pysrc.tuning._facade import create_tuner, tune
from pysrc.tuning.registry import EngineRegistry
from pysrc.tuning.result import EngineNotAvailableError, TuningError
from pysrc.tuning.space import SearchSpace
from pysrc.tuning.specs import (
    ObjectiveDirection,
    ParamPoint,
    ParamValue,
    TrialRecord,
    TunerSpec,
    TuningResult,
)

__all__ = [
    # Public facade
    "tune",
    "create_tuner",
    # Registry (for custom engine registration)
    "EngineRegistry",
    # Error hierarchy
    "TuningError",
    "EngineNotAvailableError",
    # Typed boundary models
    "TunerSpec",
    "TuningResult",
    "TrialRecord",
    "SearchSpace",
    "ObjectiveDirection",
    "ParamPoint",
    "ParamValue",
]

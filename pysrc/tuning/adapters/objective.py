"""
Pure objective-function adapter for the canonical tuning subsystem.

Migration path for ``pysrc.autotune.api.autotune`` and ``AutoTuner.run()``.

The old autotune API returned a non-canonical ``{"params": ..., "score": ...}``
dict.  This adapter accepts the same flexible input types as the old callers
and returns a canonical ``TuningResult``.

Invariants:
- No engine logic lives here — this is a thin boundary-translation layer.
- All search mechanics are delegated to ``tune()`` from ``pysrc.tuning.api``.
- No print() — structured logging only.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import structlog

from pysrc.tuning._facade import tune
from pysrc.tuning.space import SearchSpace
from pysrc.tuning.specs import ObjectiveDirection, ParamPoint, TuningResult

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


def tune_objective(
    objective_fn: Callable[[ParamPoint], float],
    space: SearchSpace | dict[str, Any] | list[dict[str, Any]],
    *,
    engine: str = "random",
    direction: ObjectiveDirection = "maximize",
    budget: int = 10,
    seed: int | None = None,
) -> TuningResult:
    """Tune a raw objective function over a search space.

    This is a thin wrapper around ``tune()`` that accepts the flexible input
    formats used by legacy autotune callers and returns a canonical
    ``TuningResult`` rather than the old non-canonical
    ``{"params": ..., "score": ...}`` dict.

    Boundary translations performed:
    - ``space`` is accepted as a SearchSpace, flat dict, or list-of-dicts;
      normalisation is delegated to ``tune()`` via ``normalize_space()``.
    - All other arguments map directly to ``tune()`` keyword arguments.

    Args:
        objective_fn: Callable that accepts a :class:`ParamPoint` and returns
                      a scalar float.  The tuner maximises or minimises this
                      value according to ``direction``.
        space:        Search space in any format accepted by
                      ``pysrc.tuning.space.normalize_space()``.
        engine:       Engine key.  Defaults to "random" (matches the old
                      autotune random-sampling behaviour).
        direction:    "maximize" or "minimize".  Defaults to "maximize" (matches
                      the old autotune convention of tracking highest score).
        budget:       Maximum evaluations.  Defaults to 10.
        seed:         Optional integer seed for reproducible sampling.

    Returns:
        A fully-populated :class:`TuningResult`.

    Raises:
        TuningError:  If the underlying engine raises.
        TypeError:    If space cannot be normalised.
        ValueError:   If space contains invalid parameter definitions.
    """
    log = logger.bind(
        adapter="objective",
        engine=engine,
        direction=direction,
        budget=budget,
        seed=seed,
    )
    log.info("tuning.adapters.objective.starting")

    result: TuningResult = tune(
        objective_fn,
        space,
        engine=engine,
        direction=direction,
        budget=budget,
        seed=seed,
    )

    log.info(
        "tuning.adapters.objective.complete",
        best_score=result.best_score,
        n_trials=len(result.trials),
    )

    return result

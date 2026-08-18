"""
Random-sampling search engine.

Draws ``spec.budget`` candidate points from SearchSpace using the standard-library
``random.Random`` seeded from ``spec.seed``.  No sklearn dependency.

Invariants:
- spec.seed is the single source of reproducibility; we construct random.Random(seed)
  locally so global RNG state is never touched.
- spec.direction is the single source of truth for best-score tracking.
- Every evaluation produces a TrialRecord.
- No print() — structured logging only.
"""

from __future__ import annotations

import random as _random
from collections.abc import Callable

import structlog

from pysrc.tuning.space import SearchSpace
from pysrc.tuning.specs import (
    ObjectiveDirection,
    ParamPoint,
    TrialRecord,
    TunerSpec,
    TuningResult,
)

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

_NEG_INF = float("-inf")
_POS_INF = float("inf")


def _is_better(score: float, best: float, direction: ObjectiveDirection) -> bool:
    """Return True iff *score* is strictly better than *best* under *direction*."""
    if direction == "maximize":
        return score > best
    return score < best


def run(
    spec: TunerSpec,
    space: SearchSpace,  # noqa: F821 — forward ref
    objective_fn: Callable[[ParamPoint], float],
) -> TuningResult:
    """Sample ``spec.budget`` candidates uniformly and return the best result.

    Args:
        spec:         Frozen engine configuration.  ``spec.seed`` seeds the
                      local ``random.Random`` instance; pass ``None`` for a
                      non-deterministic run.  ``spec.budget`` controls the
                      number of evaluations.
        space:        Canonical SearchSpace instance.  ``space.sample(n, rng)``
                      must accept an ``int`` count and a ``random.Random``
                      instance.
        objective_fn: Callable that accepts a :class:`ParamPoint` and returns a
                      scalar float.

    Returns:
        A fully-populated :class:`TuningResult`.  ``best_model`` is ``None``.

    Raises:
        TuningError: If no candidates are returned by the space.
    """
    from pysrc.tuning.result import TuningError  # noqa: PLC0415

    # Construct a local RNG — never touch global random/np.random state.
    rng: _random.Random = _random.Random(spec.seed)

    candidates = space.sample(spec.budget, rng)

    if not candidates:
        raise TuningError(
            "random engine received an empty candidate list — "
            "SearchSpace.sample() returned no candidates."
        )

    log = logger.bind(
        engine="random",
        direction=spec.direction,
        budget=spec.budget,
        seed=spec.seed,
        n_candidates=len(candidates),
    )
    log.info("tuning.random.starting")

    trials: list[TrialRecord] = []
    best_score: float = _NEG_INF if spec.direction == "maximize" else _POS_INF
    best_params: ParamPoint = {}

    for idx, params in enumerate(candidates):
        score = objective_fn(params)
        trials.append(TrialRecord(params=params, score=score))

        if _is_better(score, best_score, spec.direction):
            best_score = score
            best_params = params
            log.debug("tuning.random.new_best", trial=idx, score=score, params=params)

    log.info(
        "tuning.random.complete",
        n_trials=len(trials),
        best_score=best_score,
    )

    return TuningResult(
        best_params=best_params,
        best_score=best_score,
        best_model=None,
        trials=trials,
        engine="random",
        direction=spec.direction,
        metadata={
            "budget": spec.budget,
            "seed": spec.seed,
            "n_evaluated": len(trials),
        },
    )


# ---------------------------------------------------------------------------
# Class wrapper — required by registry lazy loaders in pysrc.tuning.registry
# ---------------------------------------------------------------------------


class RandomEngine:
    """Class wrapper around ``run`` for registry compatibility."""

    def __call__(
        self,
        spec: TunerSpec,
        space: SearchSpace,  # noqa: F821
        objective_fn: Callable[[ParamPoint], float],
    ) -> TuningResult:
        return run(spec, space, objective_fn)

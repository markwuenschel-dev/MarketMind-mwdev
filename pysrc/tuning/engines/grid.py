"""
Pure cartesian-grid search engine.

No sklearn dependency — this module enumerates SearchSpace.iter_grid() exhaustively
and evaluates the objective function for every parameter combination.  The seed
field on TunerSpec is ignored because grid search is deterministic by construction.

Invariants:
- spec.direction is the single source of truth for best-score tracking.
- Every evaluation produces a TrialRecord; none are silently skipped.
- No print() — structured logging only.
"""

from __future__ import annotations

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

# SearchSpace is imported at call-time to avoid circular imports between
# engines and space modules that may not yet be fully initialised at import.
logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# Sentinels representing "no best found yet" — distinct from 0.0.
_NEG_INF = float("-inf")
_POS_INF = float("inf")


def _is_better(score: float, best: float, direction: ObjectiveDirection) -> bool:
    """Return True iff *score* is strictly better than *best* under *direction*."""
    if direction == "maximize":
        return score > best
    # direction == "minimize"
    return score < best


def run(
    spec: TunerSpec,
    space: SearchSpace,  # noqa: F821 — forward ref; imported at call site
    objective_fn: Callable[[ParamPoint], float],
) -> TuningResult:
    """Exhaustively enumerate the search space and return the best result.

    Args:
        spec:         Frozen engine configuration.  ``spec.direction`` governs
                      which score is considered better.  ``spec.budget`` and
                      ``spec.seed`` are both ignored for grid search.
        space:        Canonical SearchSpace instance.  ``space.iter_grid()``
                      must yield every parameter combination exactly once.
        objective_fn: Callable that accepts a :class:`ParamPoint` and returns a
                      scalar float.  Must be pure with respect to the tuning run
                      (i.e., the same params always produce the same score).

    Returns:
        A fully-populated :class:`TuningResult`.  ``best_model`` is ``None``
        because this engine has no sklearn dependency.

    Raises:
        TuningError: If ``space.iter_grid()`` yields no candidates.
    """
    # Import here to avoid circular import; SearchSpace may import from engines.
    from pysrc.tuning.result import TuningError  # noqa: PLC0415

    candidates = list(space.iter_grid())
    total = len(candidates)

    if total == 0:
        raise TuningError(
            "grid engine received an empty search space — "
            "SearchSpace.iter_grid() yielded no candidates."
        )

    log = logger.bind(engine="grid", direction=spec.direction, n_candidates=total)
    log.info("tuning.grid.starting")

    trials: list[TrialRecord] = []
    best_score: float = _NEG_INF if spec.direction == "maximize" else _POS_INF
    best_params: ParamPoint = {}

    for idx, params in enumerate(candidates):
        score = objective_fn(params)
        trials.append(TrialRecord(params=params, score=score))

        if _is_better(score, best_score, spec.direction):
            best_score = score
            best_params = params
            log.debug(
                "tuning.grid.new_best",
                trial=idx,
                score=score,
                params=params,
            )

    log.info(
        "tuning.grid.complete",
        n_trials=len(trials),
        best_score=best_score,
    )

    return TuningResult(
        best_params=best_params,
        best_score=best_score,
        best_model=None,
        trials=trials,
        engine="grid",
        direction=spec.direction,
        metadata={"n_candidates": total},
    )


# ---------------------------------------------------------------------------
# Class wrapper — required by registry lazy loaders in pysrc.tuning.registry
# ---------------------------------------------------------------------------


class GridEngine:
    """Class wrapper around ``run`` for registry compatibility.

    The registry's lazy loaders expect an instantiated callable with
    ``__call__(spec, space, objective_fn) -> TuningResult``.
    """

    def __call__(
        self,
        spec: TunerSpec,
        space: SearchSpace,  # noqa: F821
        objective_fn: Callable[[ParamPoint], float],
    ) -> TuningResult:
        return run(spec, space, objective_fn)

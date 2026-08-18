"""
Bayesian optimisation engine using scikit-optimize (skopt).

Uses the functional ``skopt.gp_minimize`` API — not ``BayesSearchCV`` — because
the engine accepts a generic ``objective_fn(ParamPoint) -> float`` rather than an
sklearn estimator.

Invariants:
- skopt is an optional dependency; its import is deferred to the body of ``run()``.
  If unavailable, ``EngineNotAvailableError`` is raised with an actionable message.
  No module-level import of skopt (eliminates the soft-fail anti-pattern).
- skopt always *minimises*.  When ``spec.direction == "maximize"`` we negate the
  score for skopt and un-negate it in the returned TuningResult.
- ``spec.direction`` is preserved verbatim in TuningResult so downstream consumers
  see the sign-correct score.
- Param names are extracted from skopt named dimensions (``dim.name``), not from
  a non-existent ``space.param_names`` property.
- No print() — structured logging only.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import structlog

from pysrc.tuning.space import SearchSpace
from pysrc.tuning.specs import (
    ParamPoint,
    TrialRecord,
    TunerSpec,
    TuningResult,
)

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


def run(
    spec: TunerSpec,
    space: SearchSpace,  # noqa: F821 — forward ref
    objective_fn: Callable[[ParamPoint], float],
) -> TuningResult:
    """Run Gaussian-process Bayesian optimisation and return the best result.

    Args:
        spec:         Frozen engine configuration.  ``spec.direction`` determines
                      sign convention (scores negated for skopt when maximising).
                      ``spec.budget`` is the number of GP iterations.
                      ``spec.seed`` seeds the skopt random state.
        space:        Canonical SearchSpace instance.  ``space.to_skopt_space()``
                      must return a list of skopt Dimension objects, each with a
                      ``.name`` attribute matching the parameter name.
        objective_fn: Callable that accepts a :class:`ParamPoint` and returns a
                      scalar float.

    Returns:
        A fully-populated :class:`TuningResult`.  ``best_model`` is ``None``.
        ``best_score`` is in the caller's sign convention (un-negated if the
        direction was "maximize").

    Raises:
        EngineNotAvailableError: If ``scikit-optimize`` is not installed.
        TuningError: On skopt invocation failure or empty search space.
    """
    # --- optional dependency guard (must be inside function body) ----------
    try:
        import skopt  # type: ignore[import-not-found]  # noqa: PLC0415
        from skopt import gp_minimize  # noqa: PLC0415
    except ImportError as exc:
        from pysrc.tuning.result import EngineNotAvailableError  # noqa: PLC0415

        raise EngineNotAvailableError("bayes", "scikit-optimize (skopt)") from exc

    from pysrc.tuning.result import TuningError  # noqa: PLC0415

    # Retrieve named skopt dimension objects; extract param names from .name
    # attribute so we never depend on a non-existent param_names property.
    skopt_dimensions: list[Any] = space.to_skopt_space()

    if not skopt_dimensions:
        raise TuningError(
            "bayes engine received an empty search space — "
            "SearchSpace.to_skopt_space() returned no dimensions."
        )

    # Extract ordered param names from the skopt Dimension objects.
    param_names: list[str] = [dim.name for dim in skopt_dimensions]

    maximize = spec.direction == "maximize"
    trials: list[TrialRecord] = []

    def _skopt_objective(values: list[Any]) -> float:
        """Wrap objective_fn for skopt's list-of-values calling convention."""
        params: ParamPoint = {name: values[i] for i, name in enumerate(param_names)}
        score = objective_fn(params)
        # Record the trial in the caller's sign convention before returning.
        trials.append(TrialRecord(params=params, score=score))
        # skopt minimises; negate if the caller wants to maximise.
        return -score if maximize else score

    log = logger.bind(
        engine="bayes",
        direction=spec.direction,
        budget=spec.budget,
        seed=spec.seed,
    )
    log.info("tuning.bayes.starting")

    try:
        result = gp_minimize(
            func=_skopt_objective,
            dimensions=skopt_dimensions,
            n_calls=spec.budget,
            random_state=spec.seed,
            **spec.engine_kwargs,
        )
    except Exception as exc:
        raise TuningError(f"skopt.gp_minimize failed: {exc}") from exc

    # result.x is the list of best parameter values in dimension order.
    best_params: ParamPoint = {name: result.x[i] for i, name in enumerate(param_names)}
    # result.fun is the minimum value seen by skopt.
    # Un-negate if we were maximising so the result carries the true score.
    best_score: float = -float(result.fun) if maximize else float(result.fun)

    log.info(
        "tuning.bayes.complete",
        n_trials=len(trials),
        best_score=best_score,
    )

    return TuningResult(
        best_params=best_params,
        best_score=best_score,
        best_model=None,
        trials=trials,
        engine="bayes",
        direction=spec.direction,
        metadata={
            "budget": spec.budget,
            "seed": spec.seed,
            "n_evaluated": len(trials),
            "skopt_version": getattr(skopt, "__version__", "unknown"),
        },
    )


# ---------------------------------------------------------------------------
# Class wrapper — required by registry lazy loaders in pysrc.tuning.registry
# ---------------------------------------------------------------------------


class BayesEngine:
    """Class wrapper around ``run`` for registry compatibility."""

    def __call__(
        self,
        spec: TunerSpec,
        space: SearchSpace,  # noqa: F821
        objective_fn: Callable[[ParamPoint], float],
    ) -> TuningResult:
        return run(spec, space, objective_fn)

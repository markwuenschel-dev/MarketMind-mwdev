"""
Optuna-based hyperparameter search engine.

Migrates and canonicalises the legacy hyperparameter-search helper.
Uses ``space.to_optuna_space()`` to obtain the space definition and dispatches
each parameter to either ``trial.suggest_categorical`` (for lists with > 2 entries
or non-numeric values) or ``trial.suggest_float`` (for [low, high] continuous
ranges detected by the two-numeric-element convention in space.py).

Invariants:
- optuna is an optional dependency; its import is deferred to ``run()``.
  If unavailable, ``EngineNotAvailableError`` is raised with an actionable message.
- ``spec.direction`` is passed directly to ``optuna.create_study`` (optuna
  supports "maximize" and "minimize" natively).
- ``spec.seed`` seeds the TPE sampler if provided.
- ``spec.budget`` maps to ``n_trials``.
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


def _is_continuous_spec(candidates: list[Any]) -> bool:
    """Return True if *candidates* encodes a continuous [low, high] range.

    space.to_optuna_space() represents a continuous param as a two-element
    list ``[low: float, high: float]`` where both elements are numeric and
    not bool.  All other lists are treated as categorical candidates.
    """
    if len(candidates) != 2:
        return False
    low, high = candidates
    return (
        isinstance(low, (int, float))
        and not isinstance(low, bool)
        and isinstance(high, (int, float))
        and not isinstance(high, bool)
        # Distinguish two-element numeric categorical lists from ranges:
        # ranges must satisfy low < high; a list like [3, 3] is categorical.
        and float(low) < float(high)
    )


def run(
    spec: TunerSpec,
    space: SearchSpace,  # noqa: F821 — forward ref
    objective_fn: Callable[[ParamPoint], float],
) -> TuningResult:
    """Run Optuna TPE search and return the best result.

    Args:
        spec:         Frozen engine configuration.  ``spec.direction`` is passed
                      to ``optuna.create_study``.  ``spec.budget`` is ``n_trials``.
                      ``spec.seed`` seeds the TPE sampler when provided.
        space:        Canonical SearchSpace instance.  ``space.to_optuna_space()``
                      returns ``Dict[str, List[Any]]`` where each value is either
                      a list of categorical candidates or a two-element
                      ``[low, high]`` float list representing a continuous range.
        objective_fn: Callable that accepts a :class:`ParamPoint` and returns a
                      scalar float.

    Returns:
        A fully-populated :class:`TuningResult`.  ``best_model`` is ``None``.

    Raises:
        EngineNotAvailableError: If ``optuna`` is not installed.
        TuningError: On study-creation or optimisation failure.
    """
    # --- optional dependency guard (must be inside function body) ----------
    try:
        import optuna  # type: ignore[import-not-found]  # noqa: PLC0415
    except ImportError as exc:
        from pysrc.tuning.result import EngineNotAvailableError  # noqa: PLC0415

        raise EngineNotAvailableError("optuna", "optuna") from exc

    from pysrc.tuning.result import TuningError  # noqa: PLC0415

    # Retrieve the space descriptor: {param_name: [values] or [low, high]}.
    optuna_space: dict[str, list[Any]] = space.to_optuna_space()

    if not optuna_space:
        raise TuningError(
            "optuna engine received an empty search space — "
            "SearchSpace.to_optuna_space() returned no parameters."
        )

    trials: list[TrialRecord] = []

    def _objective(trial: Any) -> float:
        """Map an Optuna trial to a canonical ParamPoint and evaluate it."""
        params: ParamPoint = {}
        for name, candidates in optuna_space.items():
            if _is_continuous_spec(candidates):
                # Continuous range: suggest_float(name, low, high)
                params[name] = trial.suggest_float(name, float(candidates[0]), float(candidates[1]))
            else:
                # Categorical: suggest_categorical(name, choices)
                params[name] = trial.suggest_categorical(name, candidates)

        score = objective_fn(params)
        trial.report(score, step=0)
        trials.append(
            TrialRecord(
                params=params,
                score=score,
                metadata={"trial_number": trial.number},
            )
        )
        return score

    log = logger.bind(
        engine="optuna",
        direction=spec.direction,
        budget=spec.budget,
        seed=spec.seed,
    )
    log.info("tuning.optuna.starting")

    # Suppress optuna's own verbose logging — MarketMind uses structlog.
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = optuna.samplers.TPESampler(seed=spec.seed)

    try:
        study = optuna.create_study(
            direction=spec.direction,
            sampler=sampler,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=min(10, max(2, spec.budget // 5))),
        )
        study.optimize(_objective, n_trials=spec.budget, gc_after_trial=True)
    except Exception as exc:
        raise TuningError(f"optuna study failed: {exc}") from exc

    best_params: ParamPoint = dict(study.best_params)
    best_score: float = float(study.best_value)

    log.info(
        "tuning.optuna.complete",
        n_trials=len(trials),
        best_score=best_score,
    )

    return TuningResult(
        best_params=best_params,
        best_score=best_score,
        best_model=None,
        trials=trials,
        engine="optuna",
        direction=spec.direction,
        metadata={
            "budget": spec.budget,
            "seed": spec.seed,
            "n_evaluated": len(trials),
            "optuna_version": getattr(optuna, "__version__", "unknown"),
        },
    )


# ---------------------------------------------------------------------------
# Class wrapper — required by registry lazy loaders in pysrc.tuning.registry
# ---------------------------------------------------------------------------


class OptunaEngine:
    """Class wrapper around ``run`` for registry compatibility."""

    def __call__(
        self,
        spec: TunerSpec,
        space: SearchSpace,  # noqa: F821
        objective_fn: Callable[[ParamPoint], float],
    ) -> TuningResult:
        return run(spec, space, objective_fn)

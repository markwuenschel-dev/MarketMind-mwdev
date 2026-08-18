"""
sklearn estimator adapter for the canonical tuning subsystem.

Migration path for the three retired functions:
- ``pysrc.tuning.grid_search.run_grid_search``
- ``pysrc.tuning.random_search.run_random_search``
- ``pysrc.tuning.bayesian_optimization.run_bayes_search``

All three are replaced by a single ``tune_estimator()`` function that delegates
to the canonical ``tune()`` engine.

Invariants:
- sklearn is required for this adapter; it is an explicit dependency.
  ``cross_val_score`` and ``clone`` are used from sklearn.
- ``best_model`` in the returned TuningResult is the estimator *refitted* on
  the full training data with the best parameters found.
- ``spec.direction`` governs score interpretation; all other engine behaviour
  is encapsulated by ``tune()``.
- No print() — structured logging only.
"""

from __future__ import annotations

from typing import Any

import structlog

from pysrc.tuning._facade import tune
from pysrc.tuning.space import SearchSpace, normalize_space
from pysrc.tuning.specs import ObjectiveDirection, TuningResult

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


def tune_estimator(
    estimator: Any,  # sklearn estimator — no common Protocol in sklearn's typing
    space: SearchSpace | dict[str, Any] | list[dict[str, Any]],
    X_train: Any,
    y_train: Any,
    *,
    engine: str = "grid",
    direction: ObjectiveDirection = "maximize",
    budget: int = 10,
    cv: int = 5,
    scoring: str | None = None,
    seed: int | None = None,
) -> TuningResult:
    """Tune an sklearn estimator over a parameter search space.

    This is the single replacement for the retired ``run_grid_search``,
    ``run_random_search``, and ``run_bayes_search`` functions.  The caller
    selects the underlying engine via the ``engine`` keyword argument.

    Workflow:
    1. Normalise ``space`` into a canonical SearchSpace.
    2. Build a ``cross_val_score`` closure that clones the estimator, fits on
       ``(X_train, y_train)``, and returns the mean CV score.
    3. Dispatch to ``tune(objective_fn, space, engine=engine, ...)`` — the
       canonical engine handles all search logic.
    4. Refit the best estimator on the full ``(X_train, y_train)`` and attach
       it to ``result.best_model``.
    5. Return the result.

    Args:
        estimator:   Any sklearn-compatible estimator implementing ``fit`` and
                     ``set_params``.
        space:       Search space in any format accepted by normalize_space().
        X_train:     Training features (numpy array, pandas DataFrame, etc.).
        y_train:     Training labels/targets.
        engine:      Tuning engine key.  Defaults to "grid".
        direction:   "maximize" or "minimize" for the CV score.  Most sklearn
                     scoring metrics are already negated for minimisation (e.g.
                     "neg_mean_squared_error"); pass direction="maximize" when
                     using accuracy/F1/AUC-style metrics.  Defaults to "maximize".
        budget:      Maximum evaluations.  Grid engine ignores this.
        cv:          Cross-validation folds.  Defaults to 5.
        scoring:     sklearn scoring string or None (uses estimator default).
        seed:        Optional integer seed for reproducible sampling.

    Returns:
        A TuningResult with ``best_model`` populated as the refitted estimator.

    Raises:
        ImportError: If sklearn is not installed.
        TuningError: If the underlying engine raises during the search.
    """
    # sklearn import is non-optional for this adapter; fail clearly if absent.
    try:
        from sklearn.base import clone  # noqa: PLC0415
        from sklearn.model_selection import cross_val_score  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "tune_estimator() requires scikit-learn. Install it with: pip install scikit-learn"
        ) from exc

    # Normalise the space once so we share a single SearchSpace object.
    normalised_space: SearchSpace = normalize_space(space)

    log = logger.bind(
        adapter="sklearn",
        engine=engine,
        direction=direction,
        cv=cv,
        scoring=scoring,
    )
    log.info("tuning.adapters.sklearn.starting")

    def _objective(params: dict[str, Any]) -> float:
        """Clone estimator, fit with params, return mean CV score."""
        candidate = clone(estimator)
        candidate.set_params(**params)
        scores = cross_val_score(
            candidate,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
        )
        mean_score: float = float(scores.mean())
        return mean_score

    # Delegate to canonical tune() — engine, direction, budget, seed all
    # flow through TunerSpec inside tune().
    result: TuningResult = tune(
        _objective,
        normalised_space,
        engine=engine,
        direction=direction,
        budget=budget,
        cv=cv,
        scoring=scoring,
        seed=seed,
    )

    # Refit the best estimator on the full training set.
    best_estimator = clone(estimator)
    best_estimator.set_params(**result.best_params)
    best_estimator.fit(X_train, y_train)

    log.info(
        "tuning.adapters.sklearn.complete",
        best_score=result.best_score,
        n_trials=len(result.trials),
    )

    # TuningResult is frozen; replace best_model via dataclasses.replace.
    import dataclasses  # noqa: PLC0415

    return dataclasses.replace(result, best_model=best_estimator)

from __future__ import annotations

"""
Canonical public facade for the tuning subsystem.

Callers interact with the tuning system exclusively through this module.
Internal engine mechanics (space normalisation, registry lookup, factory
dispatch) are encapsulated here and invisible to callers.

Public API:
    tune()         — run a full hyperparameter search and return TuningResult.
    create_tuner() — build a reusable (spec, objective) → TuningResult callable.

No sklearn / skopt / optuna imports at module level.
"""

from collections.abc import Callable
from typing import Any

import structlog

from pysrc.tuning.registry import EngineRegistry
from pysrc.tuning.result import TuningError
from pysrc.tuning.space import SearchSpace, normalize_space
from pysrc.tuning.specs import (
    ObjectiveDirection,
    ParamPoint,
    TunerSpec,
    TuningResult,
)

_log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# tune() — one-shot hyperparameter search
# ---------------------------------------------------------------------------


def tune(
    objective_fn: Callable[[ParamPoint], float],
    space: SearchSpace | dict[str, Any] | list[dict[str, Any]],
    *,
    engine: str = "random",
    direction: ObjectiveDirection = "maximize",
    budget: int = 10,
    cv: int = 5,
    scoring: str | None = None,
    seed: int | None = None,
    **engine_kwargs: Any,
) -> TuningResult:
    """Run a hyperparameter search and return a TuningResult.

    This is the single canonical entry-point for all engine types.  Callers
    should not import or instantiate engine classes directly.

    Steps:
    1. Normalise ``space`` into a SearchSpace via normalize_space().
    2. Construct an immutable TunerSpec from the keyword arguments.
    3. Look up the engine factory from EngineRegistry (raises TuningError
       if the engine key is not recognised).
    4. Invoke the factory with (spec, normalised_space, objective_fn).
    5. Return the TuningResult produced by the engine.

    Args:
        objective_fn:    Callable that accepts a ParamPoint and returns a float.
                         The tuner maximises or minimises this value according to
                         ``direction``.  The function is called up to ``budget``
                         times (or, for the grid engine, as many times as there
                         are grid points).
        space:           Search space in any format accepted by normalize_space():
                         - SearchSpace (already normalised)
                         - Dict[str, List[...]] (categorical candidates)
                         - Dict[str, Tuple[float, float]] (continuous ranges)
                         - List[Dict] (legacy list-of-dicts YAML format)
        engine:          Engine key.  One of "grid", "random", "bayes", "optuna",
                         or any key registered via EngineRegistry.register().
                         Defaults to "random".
        direction:       "maximize" or "minimize".  Defaults to "maximize".
        budget:          Maximum number of objective evaluations.  The grid engine
                         ignores this and evaluates all grid points.  Defaults to 10.
        cv:              Cross-validation folds for sklearn-backed engines.
                         Defaults to 5.
        scoring:         sklearn scoring string (e.g. "neg_mean_squared_error").
                         Passed through to sklearn adapters; ignored by others.
        seed:            Optional integer seed for reproducible runs.  Callers
                         MUST supply this for deterministic results.  Engines
                         derive sub-seeds from this value; they never call
                         random.seed() or numpy.random.seed() directly.
        **engine_kwargs: Arbitrary keyword arguments forwarded to the engine via
                         TunerSpec.engine_kwargs.  Consult individual engine
                         documentation for supported keys.

    Returns:
        A TuningResult containing the best params, best score, all trial records,
        and run-level metadata.

    Raises:
        TuningError:  If engine is not registered.
        TuningError:  If the engine raises during the run (engine errors are
                      re-raised as TuningError with context).
        TypeError:    If space cannot be normalised (propagated from normalize_space).
        ValueError:   If space contains invalid parameter definitions.
    """
    _log.info(
        "tuning.api.tune.start",
        engine=engine,
        direction=direction,
        budget=budget,
        cv=cv,
        seed=seed,
    )

    # Step 1: normalise the space.
    normalised_space: SearchSpace = normalize_space(space)

    # Step 2: build the immutable spec.
    spec = TunerSpec(
        engine=engine,
        direction=direction,
        budget=budget,
        cv=cv,
        scoring=scoring,
        seed=seed,
        engine_kwargs=dict(engine_kwargs),
    )

    # Step 3: look up the factory (raises TuningError for unknown engine).
    factory = EngineRegistry.get(engine)

    # Step 4: run the engine.
    try:
        result = factory(spec, normalised_space, objective_fn)
    except TuningError:
        # Re-raise TuningErrors without wrapping — they already carry context.
        raise
    except Exception as exc:
        # Wrap unexpected engine errors so callers have a single catch point.
        raise TuningError(f"Engine {engine!r} raised an unexpected error: {exc!r}") from exc

    _log.info(
        "tuning.api.tune.complete",
        engine=engine,
        best_score=result.best_score,
        n_trials=len(result.trials),
    )
    return result


# ---------------------------------------------------------------------------
# create_tuner() — reusable spec-bound tuner factory
# ---------------------------------------------------------------------------


def create_tuner(
    spec: TunerSpec,
) -> Callable[[SearchSpace, Callable[[ParamPoint], float]], TuningResult]:
    """Return a partially-applied callable bound to spec.

    Use this when you want to reuse the same TunerSpec across multiple spaces
    or objective functions — for example when running ablations or when the
    space is constructed at a later point than the spec.

    Args:
        spec: Immutable TunerSpec describing the engine and its configuration.

    Returns:
        A callable with signature:
            (space: SearchSpace, objective_fn: Callable[[ParamPoint], float])
                -> TuningResult

        Calling the returned callable is equivalent to:
            tune(objective_fn, space, engine=spec.engine, ...)

    Raises:
        TuningError: If spec.engine is not registered (raised eagerly at
                     create_tuner() call time so the error surfaces before
                     the tuner is dispatched to a worker).

    Example:
        tuner = create_tuner(TunerSpec(engine="random", direction="maximize",
                                       budget=50, cv=5, scoring=None, seed=42))
        result = tuner(my_space, my_objective)
    """
    # Validate the engine key eagerly — fail fast before dispatching.
    factory = EngineRegistry.get(spec.engine)

    _log.debug("tuning.api.create_tuner", engine=spec.engine)

    def _bound_tuner(
        space: SearchSpace,
        objective_fn: Callable[[ParamPoint], float],
    ) -> TuningResult:
        normalised_space: SearchSpace = normalize_space(space)
        try:
            result = factory(spec, normalised_space, objective_fn)
        except TuningError:
            raise
        except Exception as exc:
            raise TuningError(
                f"Engine {spec.engine!r} raised an unexpected error: {exc!r}"
            ) from exc

        _log.info(
            "tuning.api.bound_tuner.complete",
            engine=spec.engine,
            best_score=result.best_score,
            n_trials=len(result.trials),
        )
        return result

    return _bound_tuner


__all__ = [
    "tune",
    "create_tuner",
]

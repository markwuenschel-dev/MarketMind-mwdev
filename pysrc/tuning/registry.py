from __future__ import annotations

"""
Registry-driven engine resolution for the tuning subsystem.

The module-level EngineRegistry singleton maps string keys to engine factory
callables.  The four canonical engines ("grid", "random", "bayes", "optuna")
are pre-registered as lazy loaders that import their implementation modules on
first access, keeping optional deps (skopt, optuna) from being loaded at import
time.

No sklearn / skopt / optuna imports at module level.
"""

from collections.abc import Callable

import structlog

from pysrc.tuning.result import TuningError
from pysrc.tuning.space import SearchSpace
from pysrc.tuning.specs import ParamPoint, TunerSpec, TuningResult

_log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Type alias for engine factories
# ---------------------------------------------------------------------------

# An engine factory is any callable that accepts (TunerSpec, SearchSpace,
# Callable[[ParamPoint], float]) and returns a TuningResult.
EngineFactory = Callable[
    [TunerSpec, SearchSpace, Callable[[ParamPoint], float]],
    TuningResult,
]


# ---------------------------------------------------------------------------
# EngineRegistry
# ---------------------------------------------------------------------------


class _EngineRegistry:
    """Registry mapping engine name keys to factory callables.

    This is an internal class; the module exposes a singleton instance named
    ``EngineRegistry``.  Callers use:

        EngineRegistry.register("my_engine", my_factory)
        factory = EngineRegistry.get("random")
    """

    def __init__(self) -> None:
        self._factories: dict[str, EngineFactory] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, name: str, factory: EngineFactory) -> None:
        """Register an engine factory under name.

        If an engine with the same name is already registered, the existing
        registration is silently replaced.  This allows downstream code to
        override built-in engines for testing or customisation.

        Args:
            name:    Engine key (e.g. "grid", "random", "bayes", "optuna").
            factory: A callable satisfying the EngineFactory signature.
        """
        if name in self._factories:
            _log.debug("tuning.registry.overriding_engine", engine=name)
        self._factories[name] = factory
        _log.debug("tuning.registry.registered_engine", engine=name)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> EngineFactory:
        """Return the factory registered under name.

        Args:
            name: Engine key to look up.

        Returns:
            The registered engine factory callable.

        Raises:
            TuningError: If name is not registered.  The message includes the
                         full list of available engine keys so the caller can
                         correct the typo without inspecting source code.
        """
        factory = self._factories.get(name)
        if factory is None:
            available = ", ".join(sorted(self._factories.keys()))
            raise TuningError(f"Unknown tuning engine {name!r}. Available: {available}")
        return factory

    def available(self) -> list[str]:
        """Return a sorted list of registered engine names."""
        return sorted(self._factories.keys())


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

EngineRegistry: _EngineRegistry = _EngineRegistry()


# ---------------------------------------------------------------------------
# Lazy engine loader helpers
# These functions are registered as factories.  They import the actual engine
# implementation only when called, so that optional dependencies (skopt, optuna)
# are not required at import time.
# ---------------------------------------------------------------------------


def _grid_factory(
    spec: TunerSpec,
    space: SearchSpace,
    objective_fn: Callable[[ParamPoint], float],
) -> TuningResult:
    """Lazy loader for the grid search engine."""
    from pysrc.tuning.engines.grid import GridEngine

    return GridEngine()(spec, space, objective_fn)


def _random_factory(
    spec: TunerSpec,
    space: SearchSpace,
    objective_fn: Callable[[ParamPoint], float],
) -> TuningResult:
    """Lazy loader for the random search engine."""
    from pysrc.tuning.engines.random import RandomEngine

    return RandomEngine()(spec, space, objective_fn)


def _bayes_factory(
    spec: TunerSpec,
    space: SearchSpace,
    objective_fn: Callable[[ParamPoint], float],
) -> TuningResult:
    """Lazy loader for the Bayesian optimisation engine (requires skopt)."""
    from pysrc.tuning.engines.bayes import BayesEngine

    return BayesEngine()(spec, space, objective_fn)


def _optuna_factory(
    spec: TunerSpec,
    space: SearchSpace,
    objective_fn: Callable[[ParamPoint], float],
) -> TuningResult:
    """Lazy loader for the Optuna engine (requires optuna)."""
    from pysrc.tuning.engines.optuna import OptunaEngine

    return OptunaEngine()(spec, space, objective_fn)


# ---------------------------------------------------------------------------
# Pre-register the four canonical engines
# ---------------------------------------------------------------------------

EngineRegistry.register("grid", _grid_factory)
EngineRegistry.register("random", _random_factory)
EngineRegistry.register("bayes", _bayes_factory)
EngineRegistry.register("optuna", _optuna_factory)

_log.debug(
    "tuning.registry.initialised",
    engines=EngineRegistry.available(),
)


__all__ = [
    "EngineRegistry",
    "EngineFactory",
]

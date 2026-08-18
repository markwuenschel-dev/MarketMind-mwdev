from __future__ import annotations

"""
Typed boundary contracts for the tuning subsystem.

Schema-first design: all engine interactions are expressed through these types.
No ad-hoc dicts cross subsystem boundaries.

Invariants:
- TunerSpec and TuningResult are frozen dataclasses — immutable after construction.
- TuningEngine Protocol is the sole contract for engine implementations.
- No sklearn/skopt/optuna imports here; this module has no optional deps.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Union

# ---------------------------------------------------------------------------
# Primitive type aliases
# ---------------------------------------------------------------------------

# Direction of optimisation: maximise (e.g. Sharpe) or minimise (e.g. loss).
ObjectiveDirection = Literal["maximize", "minimize"]

# A single scalar hyperparameter value.  Union covers all common cases without
# resorting to Any, keeping mypy --strict happysrc.
ParamValue = Union[int, float, str, bool]

# A concrete point in parameter space: {param_name: value}.
ParamPoint = dict[str, ParamValue]


# ---------------------------------------------------------------------------
# Search-space forward reference
# SearchSpace is defined in pysrc.tuning.space and re-exported from __init__.
# We declare it here only as a Protocol-level type alias so that specs.py
# remains importable without importing space.py (which may import skopt).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# TunerSpec — immutable engine configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TunerSpec:
    """Immutable configuration for a single tuning run.

    Attributes:
        engine:         Engine key ("grid", "random", "bayes", "optuna").
        direction:      Whether the objective should be maximised or minimised.
        budget:         Maximum number of evaluations.  Grid engine ignores this
                        and evaluates the full Cartesian product.
        cv:             Number of cross-validation folds (used by sklearn adapter).
        scoring:        sklearn scoring string (optional; engine may default).
        seed:           Optional integer seed for reproducible runs.  Engines
                        MUST derive all sub-seeds from this value via HMAC-SHA256
                        rather than calling random.seed() or np.random.seed()
                        directly.
        engine_kwargs:  Arbitrary engine-specific keyword arguments.  Engines are
                        responsible for documenting and validating their own keys.
    """

    engine: str
    direction: ObjectiveDirection
    budget: int
    cv: int
    scoring: str | None
    seed: int | None
    # engine_kwargs is mutable by design (callers may need to pass mutable
    # structures such as lists of callbacks), so we use Dict[str, Any] with a
    # factory default.  Any is unavoidable here because engine kwargs are
    # heterogeneous and opaque at this layer.
    engine_kwargs: dict[str, Any] = field(
        default_factory=dict,
        # field is technically mutable, but the dataclass is frozen — the dict
        # reference cannot be replaced, only the dict contents can be mutated.
        # This is intentional: callers who need full immutability should pass an
        # immutable mapping and treat it as read-only.
        compare=False,
    )


# ---------------------------------------------------------------------------
# TrialRecord — single evaluation record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrialRecord:
    """Record of a single objective-function evaluation.

    Attributes:
        params:   The hyperparameter point that was evaluated.
        score:    The scalar returned by the objective function.
        metadata: Optional engine-level annotations (e.g. wall time, fold
                  scores, convergence info).  Opaque at this layer.
    """

    params: ParamPoint
    score: float
    # metadata values are engine-specific; Any is unavoidable here.
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)


# ---------------------------------------------------------------------------
# TuningResult — immutable aggregate result of a tuning run
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TuningResult:
    """Immutable aggregate result returned by every engine.

    Attributes:
        best_params: The parameter point that achieved the best score.
        best_score:  The score at best_params.
        best_model:  Populated by sklearn adapter only; None for all other
                     engines.  Type is Any because sklearn estimators have no
                     shared base class in the typing sense.
        trials:      Ordered list of all evaluated TrialRecords.
        engine:      Key of the engine that produced this result.
        direction:   Direction used during optimisation (preserved for
                     downstream consumers that need to know sign convention).
        metadata:    Run-level annotations (total wall time, convergence flag,
                     etc.).  Opaque at this layer.
    """

    best_params: ParamPoint
    best_score: float
    # best_model is intentionally Any: sklearn estimators have no common Protocol.
    best_model: Any | None  # noqa: ANN401 — heterogeneous sklearn object
    trials: list[TrialRecord]
    engine: str
    direction: ObjectiveDirection
    # metadata values are run-level and engine-specific.
    metadata: dict[str, Any] = field(default_factory=dict, compare=False)


# ---------------------------------------------------------------------------
# TuningEngine Protocol — contract for all engine implementations
# ---------------------------------------------------------------------------


class TuningEngine:
    """Protocol satisfied by every engine implementation.

    Engine implementations must be callable with the signature below and return
    a fully-populated TuningResult.

    Example (skeleton):
        class MyEngine:
            def __call__(
                self,
                spec: TunerSpec,
                space: "SearchSpace",
                objective_fn: Callable[[ParamPoint], float],
            ) -> TuningResult:
                ...

    The import of SearchSpace is deferred to avoid a circular import between
    specs.py and space.pysrc.  Engine implementations should import SearchSpace
    directly from pysrc.tuning.space.
    """

    def __call__(  # pragma: no cover — Protocol stub, not executed directly
        self,
        spec: TunerSpec,
        space: Any,  # SearchSpace — deferred to avoid circular import
        objective_fn: Callable[[ParamPoint], float],
    ) -> TuningResult:
        """Run the engine and return results.

        Args:
            spec:         Frozen engine configuration.
            space:        Normalised SearchSpace instance.
            objective_fn: Callable that accepts a ParamPoint and returns a float.

        Returns:
            A fully-populated TuningResult.
        """
        raise NotImplementedError  # Subclasses / Protocol implementations provide this.


__all__ = [
    "ObjectiveDirection",
    "ParamValue",
    "ParamPoint",
    "TunerSpec",
    "TrialRecord",
    "TuningResult",
    "TuningEngine",
]

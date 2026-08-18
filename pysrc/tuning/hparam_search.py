"""Hyperparameter search coordination.

Typed contracts for search space definition, trial execution,
and result aggregation. Integrates with autotune API and
artifact registry for full lineage tracking.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from pysrc.tuning.result import TuningResult
    from pysrc.tuning.space import SearchSpace


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Trial:
    """Individual hyperparameter trial result.

    Attributes:
        trial_id: Unique trial identifier
        params: Hyperparameter values for this trial
        metric: Objective metric value
        metadata: Optional additional measurements
    """

    trial_id: str
    params: Mapping[str, Any]
    metric: float
    metadata: Mapping[str, float] | None = None


@dataclass(frozen=True, slots=True)
class SearchState:
    """Mutable search state snapshot (immutable container).

    Attributes:
        completed: Number of completed trials
        best_trial: Best trial seen so far
        history: All trial results in order
    """

    completed: int
    best_trial: Trial | None
    history: Sequence[Trial]


class SearchAlgorithm[T](ABC):
    """Abstract search algorithm contract.

    Implementations provide Bayesian, grid, random, or custom
    search strategies with deterministic outcomes.

    TODO: Registry hook for algorithm registration.
    TODO: Integration with tuning.space for domain validation.
    """

    @abstractmethod
    def suggest(self, state: SearchState, space: SearchSpace) -> Mapping[str, T]:
        """Suggest next hyperparameter configuration.

        Args:
            state: Current search state
            space: Valid search space definition

        Returns:
            Suggested parameter configuration
        """
        ...

    @abstractmethod
    def update(self, state: SearchState, trial: Trial) -> SearchState:
        """Update search state with trial result.

        Args:
            state: Current search state
            trial: Completed trial result

        Returns:
            Updated search state
        """
        ...


class HparamSearch(ABC):
    """Hyperparameter search coordinator.

    High-level interface for executing searches with
    consistent result handling and artifact tracking.

    TODO: Factory integration for search instantiation.
    TODO: Telemetry integration for progress reporting.
    """

    @abstractmethod
    def execute(
        self,
        objective: Callable[..., float],
        space: SearchSpace,
        algorithm: SearchAlgorithm[Any],
        *,
        max_trials: int,
        early_stop_patience: int = 0,
    ) -> TuningResult:
        """Execute hyperparameter search.

        Args:
            objective: Callable to minimize
            space: Parameter search space
            algorithm: Search strategy implementation
            max_trials: Maximum number of trials
            early_stop_patience: Trials to wait for improvement

        Returns:
            Complete tuning result with best parameters
        """
        ...

    @abstractmethod
    def resume(self, run_id: str) -> TuningResult:
        """Resume interrupted search from artifact.

        Args:
            run_id: Artifact registry run identifier

        Returns:
            Completed or partial tuning result
        """
        ...

from __future__ import annotations

"""
Centralised search-space normalisation for the tuning subsystem.

Responsibilities:
- Define the canonical SearchSpace container.
- Expose normalize_space() that accepts three raw input formats and returns
  a normalised SearchSpace.
- Provide iteration helpers (iter_grid, sample) and adapter converters
  (to_optuna_space, to_skopt_space).

Dependencies:
- stdlib only (itertools, random, copy).
- Optional: skopt (for to_skopt_space).  If not installed, calling that method
  raises ImportError with a clear message naming the missing package.
- No sklearn, no numpy, no pandas imported here.
"""

import copy
import itertools
import random as _random_module
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, Union

import structlog

from pysrc.tuning.specs import ParamPoint, ParamValue

_log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Internal parameter definition types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CategoricalParam:
    """A parameter defined by an explicit list of candidate values."""

    name: str
    candidates: tuple[ParamValue, ...]


@dataclass(frozen=True)
class _ContinuousParam:
    """A parameter defined by a closed real-valued interval [low, high]."""

    name: str
    range_low: float
    range_high: float


# Union of the two concrete kinds understood by SearchSpace.
_ParamDef = Union[_CategoricalParam, _ContinuousParam]


# ---------------------------------------------------------------------------
# SearchSpace
# ---------------------------------------------------------------------------


@dataclass
class SearchSpace:
    """Normalised container for hyperparameter search-space definitions.

    Construct via normalize_space() rather than directly to ensure input
    validation and format normalisation.

    Internal representation stores each parameter as either a _CategoricalParam
    (list of explicit candidates) or a _ContinuousParam (real-valued range).
    """

    # Internal ordered list of parameter definitions.
    _param_defs: list[_ParamDef] = field(default_factory=list, repr=False)

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def params(self) -> list[_ParamDef]:
        """Return a shallow copy of the internal parameter definitions.

        Returns a list of _CategoricalParam and _ContinuousParam instances.
        Callers should treat the list as read-only.
        """
        return list(self._param_defs)

    # ------------------------------------------------------------------
    # Iteration helpers
    # ------------------------------------------------------------------

    def iter_grid(self) -> Iterable[ParamPoint]:
        """Yield all points in the Cartesian product of categorical parameters.

        Continuous parameters are skipped (they have infinite support).
        This is the enumeration used by the "grid" engine.

        Yields:
            ParamPoint dicts covering every combination of categorical values.
        """
        categorical: list[_CategoricalParam] = [
            p for p in self._param_defs if isinstance(p, _CategoricalParam)
        ]
        if not categorical:
            # Edge case: space has only continuous params — yield one empty point
            # so callers always get at least one iteration.
            yield {}
            return

        names = [p.name for p in categorical]
        value_lists: list[tuple[ParamValue, ...]] = [p.candidates for p in categorical]
        for combo in itertools.product(*value_lists):
            yield dict(zip(names, combo, strict=False))

    def sample(self, n: int, rng: _random_module.Random) -> list[ParamPoint]:
        """Draw n random ParamPoints from the space.

        For categorical params, a value is chosen uniformly from the candidates.
        For continuous params, a value is drawn uniformly from [low, high].

        Args:
            n:   Number of samples to draw.
            rng: A seeded random.Random instance.  Callers must supply this;
                 the method never seeds or creates its own RNG so that
                 determinism is fully controlled by the caller.

        Returns:
            List of n ParamPoint dicts (may contain duplicates for small spaces).
        """
        if n < 1:
            return []

        results: list[ParamPoint] = []
        for _ in range(n):
            point: ParamPoint = {}
            for p in self._param_defs:
                if isinstance(p, _CategoricalParam):
                    point[p.name] = rng.choice(list(p.candidates))
                else:
                    # uniform float sample from continuous range
                    point[p.name] = rng.uniform(p.range_low, p.range_high)
            results.append(point)
        return results

    # ------------------------------------------------------------------
    # Adapter converters
    # ------------------------------------------------------------------

    def to_optuna_space(self) -> dict[str, list[Any]]:
        """Return a merged categorical dict compatible with optuna suggest_categorical.

        Continuous params are represented by their [low, high] pair so the
        optuna engine adapter can route them to suggest_float instead.

        Returns:
            Dict mapping param name to either a list of candidates (categorical)
            or a [float, float] pair (continuous).
        """
        result: dict[str, list[Any]] = {}
        for p in self._param_defs:
            if isinstance(p, _CategoricalParam):
                result[p.name] = list(p.candidates)
            else:
                # Represent continuous range as [low, high]; the optuna adapter
                # will call trial.suggest_float(name, low, high).
                result[p.name] = [p.range_low, p.range_high]
        return result

    def to_skopt_space(self) -> list[Any]:
        """Return a list of skopt Dimension objects.

        Raises:
            ImportError: If scikit-optimize is not installed.  The message names
                         the missing package and the engine that requires it.
        """
        try:
            from skopt.space import Categorical, Real  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "Engine 'bayes' requires scikit-optimize (skopt), which is not "
                "installed.  Install it with: pip install scikit-optimize"
            ) from exc

        dimensions: list[Any] = []
        for p in self._param_defs:
            if isinstance(p, _CategoricalParam):
                dimensions.append(Categorical(list(p.candidates), name=p.name))
            else:
                dimensions.append(Real(p.range_low, p.range_high, name=p.name))
        return dimensions

    def __repr__(self) -> str:
        names = [p.name for p in self._param_defs]
        return f"SearchSpace(params={names!r})"


# ---------------------------------------------------------------------------
# normalize_space — the single entry point for space construction
# ---------------------------------------------------------------------------


def normalize_space(raw: Any) -> SearchSpace:
    """Normalise heterogeneous raw input into a canonical SearchSpace.

    Accepted formats:

    1. Dict[str, List[ParamValue]] — categorical candidates:
       ``{"lr": [0.01, 0.001], "depth": [3, 5, 7]}``

    2. Dict[str, Tuple[float, float]] — continuous ranges:
       ``{"lr": (1e-4, 1e-1), "depth": (1, 10)}``

    3. Mixed dict — any combination of lists and (low, high) tuples:
       ``{"lr": (1e-4, 1e-1), "depth": [3, 5, 7]}``

    4. List-of-dicts YAML format:
       ``[{"lr": [0.01, 0.001]}, {"depth": [3, 5]}]``

    Args:
        raw: Raw space definition in any of the formats above.

    Returns:
        A normalised SearchSpace instance.

    Raises:
        TypeError:  If raw is not a dict or list.
        ValueError: If a parameter value cannot be interpreted as a list of
                    candidates or a (low, high) range tuple.
    """
    if isinstance(raw, SearchSpace):
        # Already normalised — return a copy to avoid aliasing surprises.
        return copy.copy(raw)

    if isinstance(raw, list):
        raw = _merge_list_of_dicts(raw)

    if not isinstance(raw, dict):
        raise TypeError(
            f"normalize_space() expects a dict or list-of-dicts, got {type(raw).__name__!r}"
        )

    param_defs: list[_ParamDef] = []
    for name, spec in raw.items():
        param_def = _parse_param(name, spec)
        param_defs.append(param_def)
        _log.debug("tuning.space.normalised_param", param_name=name, kind=type(param_def).__name__)

    return SearchSpace(_param_defs=param_defs)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _merge_list_of_dicts(raw: list[Any]) -> dict[str, Any]:
    """Flatten a list-of-dicts YAML format into a single dict.

    ``[{"lr": [0.01]}, {"depth": [3, 5]}]`` → ``{"lr": [0.01], "depth": [3, 5]}``

    Raises:
        TypeError:  If any element of the list is not a dict.
        ValueError: If a parameter name appears more than once.
    """
    merged: dict[str, Any] = {}
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise TypeError(
                f"normalize_space(): list element {i} is {type(item).__name__!r}, expected dict"
            )
        for k, v in item.items():
            if k in merged:
                raise ValueError(
                    f"normalize_space(): duplicate parameter name {k!r} in list-of-dicts input"
                )
            merged[k] = v
    return merged


def _parse_param(name: str, spec: Any) -> _ParamDef:
    """Parse a single parameter spec into a typed _ParamDef.

    A spec is either:
    - A list/tuple of values whose length != 2, or a list of non-numeric
      values → _CategoricalParam.
    - A tuple of exactly two numeric values (low, high) → _ContinuousParam.
    - A list of exactly two numeric values (low, high) → ambiguous; we treat
      it as _CategoricalParam (two explicit candidates) unless both are numeric
      AND the caller wrapped them in a tuple explicitly.  This avoids silent
      mis-classification of ``[3, 5]`` as a range; callers who want a range must
      use a tuple ``(3, 5)``.

    Args:
        name: Parameter name (used in error messages).
        spec: Raw candidate list or range tuple.

    Returns:
        _CategoricalParam or _ContinuousParam.

    Raises:
        ValueError: If spec cannot be interpreted.
    """
    if isinstance(spec, tuple):
        if len(spec) == 2 and _both_numeric(spec[0], spec[1]):
            low = float(spec[0])
            high = float(spec[1])
            if low > high:
                raise ValueError(
                    f"normalize_space(): parameter {name!r} has range_low ({low}) "
                    f"> range_high ({high})"
                )
            return _ContinuousParam(name=name, range_low=low, range_high=high)
        # Tuple of length != 2, or non-numeric — treat as categorical candidates.
        return _CategoricalParam(name=name, candidates=tuple(_cast_param_values(name, spec)))

    if isinstance(spec, list):
        return _CategoricalParam(name=name, candidates=tuple(_cast_param_values(name, spec)))

    raise ValueError(
        f"normalize_space(): parameter {name!r} spec must be a list (candidates) "
        f"or a 2-tuple (low, high range), got {type(spec).__name__!r}"
    )


def _both_numeric(a: Any, b: Any) -> bool:
    """Return True if both a and b are int or float (but not bool)."""
    # bool is a subclass of int in Python, but is not a valid range bound here.
    return (
        isinstance(a, (int, float))
        and not isinstance(a, bool)
        and isinstance(b, (int, float))
        and not isinstance(b, bool)
    )


def _cast_param_values(name: str, values: Iterable[Any]) -> Iterator[ParamValue]:
    """Validate and yield each value as a ParamValue.

    Raises:
        ValueError: If a value is not int, float, str, or bool.
    """
    for v in values:
        if not isinstance(v, (int, float, str, bool)):
            raise ValueError(
                f"normalize_space(): parameter {name!r} candidate {v!r} has type "
                f"{type(v).__name__!r}; expected int, float, str, or bool"
            )
        yield v


__all__ = [
    "SearchSpace",
    "normalize_space",
]

"""Search space constraints: feasibility checks applied before trial submission."""

from __future__ import annotations

from typing import Any


class ConstraintViolationError(ValueError):
    """Raised when a candidate violates a declared search constraint."""


def check_bounds(name: str, value: float, low: float, high: float) -> None:
    """Raise ConstraintViolationError if value is outside [low, high]."""
    if not (low <= value <= high):
        raise ConstraintViolationError(
            f"Parameter {name!r} = {value} is outside bounds [{low}, {high}]"
        )


def check_no_nan(params: dict[str, Any]) -> None:
    """Raise ConstraintViolationError if any float parameter is NaN."""
    import math

    for k, v in params.items():
        if isinstance(v, float) and math.isnan(v):
            raise ConstraintViolationError(f"Parameter {k!r} is NaN")


def validate_params(
    params: dict[str, Any],
    bounds_map: dict[str, tuple[float, float]],
) -> None:
    """Apply bounds and NaN checks to a candidate param dict."""
    check_no_nan(params)
    for name, (lo, hi) in bounds_map.items():
        if name in params and isinstance(params[name], (int, float)):
            check_bounds(name, float(params[name]), lo, hi)


__all__ = ["ConstraintViolationError", "check_bounds", "check_no_nan", "validate_params"]

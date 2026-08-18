"""
Legacy YAML grid adapter for the canonical tuning subsystem.

Migration path for the legacy list-of-dicts hyperparameter-search helpers.

These functions consumed a list-of-dicts YAML format and either yielded raw
param dicts or built a flat categorical dict for optuna.  This adapter replaces
both with a single conversion function that produces a canonical SearchSpace.

The list-of-dicts format is:
    [{"lr": [0.01, 0.001]}, {"depth": [3, 5, 7]}, ...]

Each element is a dict mapping a parameter name to its list of candidate values.
Parameter names must be unique across the list.

Invariants:
- This is a pure conversion function — no engine logic, no side effects.
- Delegating to ``normalize_space()`` ensures that validation rules (unique
  param names, valid value types) are enforced in one place.
- No print() — structured logging only.
"""

from __future__ import annotations

from typing import Any

import structlog

from pysrc.tuning.space import SearchSpace, normalize_space

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


def parse_yaml_grid(grid: list[dict[str, Any]]) -> SearchSpace:
    """Convert a list-of-dicts YAML grid into a canonical SearchSpace.

    Accepts the legacy list-of-dicts hyperparameter-search format:

    .. code-block:: python

        grid = [
            {"lr": [0.01, 0.001, 0.0001]},
            {"hidden_size": [64, 128, 256]},
            {"dropout": [0.1, 0.2, 0.5]},
        ]
        space = parse_yaml_grid(grid)

    Each element in the list is a ``{param_name: [values]}`` dict.  Parameter
    names must be unique across the list.

    This function is a pure boundary-translation layer: all normalisation and
    validation logic lives in ``normalize_space()`` (``pysrc.tuning.space``).

    Args:
        grid: List of single-key dicts, each mapping a parameter name to its
              list of candidate values.  Mixed types within a single list are
              allowed as long as each value is int, float, str, or bool.

    Returns:
        A canonical :class:`SearchSpace` instance.

    Raises:
        TypeError:  If ``grid`` is not a list or any element is not a dict.
        ValueError: If a parameter name appears more than once across the list.
        ValueError: If any candidate value is not int, float, str, or bool.
    """
    logger.debug(
        "tuning.adapters.legacy_yaml.parse_yaml_grid",
        n_params=len(grid),
    )

    # Delegate entirely to normalize_space() which already handles the
    # list-of-dicts format natively.
    return normalize_space(grid)

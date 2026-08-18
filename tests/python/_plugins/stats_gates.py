"""
Statistical gate helpers: DSR, minTRL, PBO (re-export / wrappers from py).

Provides fixtures so tests can reuse compute_dsr, compute_min_trl, compute_bootstrap_ci
without importing pysrc.backtesting.validation.statistical directly.
Optional: if scipy or the statistical module is unavailable, fixtures skip.
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest


def _get_statistical_backend() -> SimpleNamespace | None:
    """Import DSR/minTRL helpers from py when available (requires scipy)."""
    try:
        from pysrc.backtesting.validation.statistical.dsr import (
            DSRComputationError,
            DSRDataError,
            compute_bootstrap_ci,
            compute_dsr,
            compute_min_trl,
        )

        return SimpleNamespace(
            compute_dsr=compute_dsr,
            compute_min_trl=compute_min_trl,
            compute_bootstrap_ci=compute_bootstrap_ci,
            DSRDataError=DSRDataError,
            DSRComputationError=DSRComputationError,
        )
    except ImportError:
        return None


@pytest.fixture(scope="session")
def statistical_helpers() -> SimpleNamespace:
    """
    Session-scoped namespace with compute_dsr, compute_min_trl, compute_bootstrap_ci,
    DSRDataError, DSRComputationError. Skips if py statistical backend is unavailable (e.g. missing scipy).
    """
    backend = _get_statistical_backend()
    if backend is None:
        pytest.skip(
            "Statistical backend unavailable (install scipy and pysrc.backtesting.validation.statistical)"
        )
    return backend


@pytest.fixture
def dsr_validator(statistical_helpers: SimpleNamespace) -> Callable[..., dict[str, Any]]:
    """
    Returns a validator callable: compute_dsr(returns, n_trials=...) and check gate_result / p_value.
    Use for tests that need DSR without importing the statistical module directly.
    """
    return statistical_helpers.compute_dsr


@pytest.fixture
def min_trl_validator(statistical_helpers: SimpleNamespace) -> Callable[..., dict[str, Any]]:
    """Returns compute_min_trl for use as a minTRL gate helper."""
    return statistical_helpers.compute_min_trl

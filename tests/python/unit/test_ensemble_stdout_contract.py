# tests/python/unit/test_ensemble_stdout_contract.py
"""
P0: Bridge-safe stdout contract for EnsemblePipelineStrategy.generate_signal().

Ensures no print() in the hot path so Java bridge stdout parsing and benchmark
timing are not corrupted. Unit test captures stdout during generate_signal and
asserts it is empty.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pysrc.strategies.migrated_strategies import EnsemblePipelineStrategy


@pytest.fixture
def minimal_ensemble():
    """Ensemble with one RSI sub-strategy (must be registered)."""
    return EnsemblePipelineStrategy(strategy_specs=[("rsi", {"rsi_window": 14})])


@pytest.fixture
def features_for_rsi():
    """Minimal features DataFrame: one column s0__rsi14 as consumed by ensemble."""
    idx = pd.date_range("2022-01-01", periods=20, freq="B")
    return pd.DataFrame({"s0__rsi14": [50.0] * 20}, index=idx)


def test_ensemble_generate_signal_emits_no_stdout(capsys, minimal_ensemble, features_for_rsi):
    """generate_signal() must not write to stdout (bridge contract)."""
    minimal_ensemble.generate_signal(features_for_rsi)
    out, _ = capsys.readouterr()
    assert out == "", (
        "EnsemblePipelineStrategy.generate_signal() must not print to stdout (Java bridge contract)"
    )


def test_ensemble_generate_signal_returns_series(minimal_ensemble, features_for_rsi):
    """Sanity: generate_signal returns a Series indexed like features."""
    result = minimal_ensemble.generate_signal(features_for_rsi)
    assert isinstance(result, pd.Series)
    assert len(result) == len(features_for_rsi)
    assert (result.index == features_for_rsi.index).all()

"""
py/backtesting/compat/backtest_pipeline.py

Deprecation shim for the old pysrc.backtest_pipeline module.
Raises ImportError with a clear migration message so callers know
exactly which symbol to replace and where.
"""

from __future__ import annotations


def __getattr__(name: str):  # module-level __getattr__ (PEP 562)
    raise ImportError(
        f"'{name}' was removed from pysrc.backtesting.compat.backtest_pipeline. "
        "Migrate to pysrc.backtesting.orchestration.suite_runner.BacktestSuiteRunner."
    )

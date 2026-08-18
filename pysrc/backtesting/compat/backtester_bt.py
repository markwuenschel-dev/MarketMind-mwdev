"""
py/backtesting/compat/backtester_bt.py

Deprecation shim for the old backtester compatibility module.
Raises ImportError with a clear migration message so callers know
exactly which symbol to replace and where.
"""

from __future__ import annotations


def __getattr__(name: str):  # module-level __getattr__ (PEP 562)
    raise ImportError(
        f"'{name}' was removed from pysrc.backtesting.compat.backtester_bt. "
        "Migrate to pysrc.backtesting.engines.backtrader.adapter.Backtester."
    )

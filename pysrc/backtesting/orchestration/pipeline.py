"""Deprecated backtesting orchestration entry.

This module previously housed the Phase 0 ``BacktestPipeline`` stub.
It now exists solely as a compatibility shim that forwards to the
backtest suite runner in ``pysrc.backtesting.orchestration.suite_runner``.
"""

from __future__ import annotations

from typing import Any

from pysrc.backtesting.orchestration.suite_runner import BacktestSuiteRunner


class BacktestPipeline(BacktestSuiteRunner):
    """Compatibility alias for ``BacktestSuiteRunner``.

    New code should import and construct ``BacktestSuiteRunner`` from
    ``pysrc.backtesting.orchestration.suite_runner`` directly.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config=config)

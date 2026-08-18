from __future__ import annotations

from typing import Any

from pysrc.backtesting.contracts.errors import OptionalDependencyMissingError
from pysrc.backtesting.contracts.registry import register_engine


class BacktraderEngineAdapter:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}

    def run(self, plan, data, store):
        raise OptionalDependencyMissingError(
            "Backtrader is an optional backend and is not wired into this scaffold runtime."
        )


class Backtester(BacktraderEngineAdapter):
    pass


register_engine("backtrader.scaffold", lambda: BacktraderEngineAdapter())

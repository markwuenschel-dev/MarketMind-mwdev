# py/domain/__init__.py
"""
Domain package: Defines business logic interfaces and models for MarketMind.
Prioritizes Polars for data handling, with abstract factories for dynamic extensibility.
"""

import importlib
import sys
from typing import Any

__all__ = [
    "OrderExecutor",
    "PositionService",
    "MarketDataProvider",
    "ProviderFactory",
    "EconomicDataProvider",
]


# Lazy import: Defer loading heavy dependencies until first access
def __getattr__(name: str) -> Any:
    if name in __all__:
        module = importlib.import_module(".interfaces", __package__)
        value = getattr(module, name)
        setattr(sys.modules[__name__], name, value)
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

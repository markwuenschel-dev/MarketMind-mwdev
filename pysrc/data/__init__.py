from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "ib_api",
    "preprocessor",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module 'pysrc.data' has no attribute {name!r}")
    module = import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    return list(__all__)

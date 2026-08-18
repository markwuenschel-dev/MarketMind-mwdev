# --- MARKET-DATA-PACKAGE-SHIM ---
"""
Lazy loader to avoid importing heavy/broken submodules during smoke tests.
"""

from importlib import import_module
from types import ModuleType

__all__ = ["sources"]  # only expose the registry package by default


def __getattr__(name: str):
    if name == "sources":
        return import_module(f"{__name__}.sources")
    # Best-effort lazy import; if a submodule is broken, return an empty module
    try:
        return import_module(f"{__name__}.{name}")
    except Exception:
        return ModuleType(name)


# --- /MARKET-DATA-PACKAGE-SHIM ---

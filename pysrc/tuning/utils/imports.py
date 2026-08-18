"""Safe optional-import helper with actionable error messages.

Provides a single call site for guarded imports so missing extras surface clearly.
"""

from __future__ import annotations

import importlib
import types

__all__ = ["require_import"]


def require_import(module: str, pip_name: str) -> types.ModuleType:
    """Import *module*, raising ImportError with a pip install hint if absent."""
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"Optional dependency '{module}' is required for this feature. "
            f"Install it with: pip install {pip_name}"
        ) from exc

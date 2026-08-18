"""Thin structured-logging wrapper; falls back to stdlib logging if mm_logkit is absent.

Ensures the tuning sub-system never hard-depends on the internal mm_logkit package.
"""

from __future__ import annotations

from typing import Any

__all__ = ["get_logger"]

try:
    from pysrc.ops.mm_logkit import get_logger as _get_logger

    def get_logger(name: str) -> Any:
        """Return a structured logger backed by mm_logkit."""
        return _get_logger(name)

except ImportError:
    import logging as _logging

    def get_logger(name: str) -> Any:
        """Return a stdlib logger when mm_logkit is unavailable."""
        return _logging.getLogger(name)

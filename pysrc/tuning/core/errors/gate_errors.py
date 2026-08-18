"""Typed gate evaluation errors."""

from __future__ import annotations


class GateError(RuntimeError):
    """Base class for gate evaluation errors."""


class GateFailedError(GateError):
    """Raised when a candidate fails a required gate check."""

    def __init__(self, gate_name: str, reason: str) -> None:
        super().__init__(f"Gate '{gate_name}' failed: {reason}")
        self.gate_name = gate_name
        self.reason = reason


class GateConfigError(GateError):
    """Raised when a gate is misconfigured."""


__all__ = ["GateError", "GateFailedError", "GateConfigError"]

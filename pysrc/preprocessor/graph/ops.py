# py/preprocessor/graph/ops.py
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any


class OpKind(StrEnum):
    elementwise = "elementwise"  # column-wise, no inter-row deps
    rolling = "rolling"  # windowed ops with lookback
    sequence = "sequence"  # lags/targets/window construction
    scaling = "scaling"  # fit/transform style normalization
    external = "external"  # calls out to models/services


class Op(ABC):
    """Base operation node; planner/executors use metadata + IR emitted by to_ir()."""

    NAME: str | None = None  # canonical registry name; default to class name
    KIND: OpKind = OpKind.elementwise

    def __init__(self, **params: Any):
        self.params: dict[str, Any] = params
        self._requires: set[str] | None = None
        self._provides: set[str] | None = None
        self.validate_params()

    @property
    def name(self) -> str:
        return self.NAME or self.__class__.__name__

    def validate_params(self) -> None:
        """Override to enforce required keys / types."""
        pass

    @property
    def requires(self) -> set[str]:
        """Declare required input columns (planner may use for ordering)."""
        if self._requires is None:
            self._requires = self._compute_requires()
        return self._requires

    def _compute_requires(self) -> set[str]:
        """Override in subclasses to compute required columns."""
        return set()

    @property
    def provides(self) -> set[str]:
        """Declare produced columns (planner may use for wiring/conflict checks)."""
        if self._provides is None:
            self._provides = self._compute_provides()
        return self._provides

    def _compute_provides(self) -> set[str]:
        """Override in subclasses to compute provided columns."""
        return set()

    def is_fittable(self) -> bool:
        """True if op needs a fit phase (scalers, encoders)."""
        return False

    def state_dict(self) -> dict[str, Any]:
        """Return learned state for serialization (for fittable ops)."""
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> Op:
        """Restore learned state."""
        return self

    def _clone_params(self) -> dict[str, Any]:
        """Returns a shallow copy of this op's params."""
        return dict(self.params)

    def clone(self) -> Op:
        """Create a new instance of this Op with the same params and state."""
        new = type(self)(**self._clone_params())
        new.load_state_dict(self.state_dict())
        return new

    @abstractmethod
    def to_ir(self) -> dict[str, Any]:
        """Return backend-agnostic IR node; planner/executors lower this further.

        Expected minimal keys: {"op": <name>, "kind": <OpKind>, "params": {...}}
        """
        raise NotImplementedError


class ElementwiseOp(Op):
    KIND = OpKind.elementwise

    def to_ir(self) -> dict[str, Any]:
        return {"op": self.name, "kind": self.KIND.value, "params": dict(self.params)}


class RollingOp(Op):
    KIND = OpKind.rolling

    def to_ir(self) -> dict[str, Any]:
        return {"op": self.name, "kind": self.KIND.value, "params": dict(self.params)}


class SequenceOp(Op):
    KIND = OpKind.sequence

    def to_ir(self) -> dict[str, Any]:
        return {"op": self.name, "kind": self.KIND.value, "params": dict(self.params)}


class ScalingOp(Op):
    KIND = OpKind.scaling

    def is_fittable(self) -> bool:
        return True

    def to_ir(self) -> dict[str, Any]:
        return {"op": self.name, "kind": self.KIND.value, "params": dict(self.params)}


class ExternalOp(Op):
    KIND = OpKind.external

    def to_ir(self) -> dict[str, Any]:
        return {"op": self.name, "kind": self.KIND.value, "params": dict(self.params)}


def get_registry() -> set:
    """Return all registered op symbols (across all backends) from the backend registry.

    Used by verification checks (E2) to confirm that required op symbols such as
    ``feature.returns``, ``feature.sma``, and ``feature.rsi`` are registered.
    """
    from pysrc.preprocessor.graph.backends.registry import list_ops as _list_ops

    all_ops: set = set()
    for backend in ("polars", "cudf"):
        all_ops.update(_list_ops(backend))
    return all_ops


__all__ = [
    "OpKind",
    "Op",
    "ElementwiseOp",
    "RollingOp",
    "SequenceOp",
    "ScalingOp",
    "ExternalOp",
    "get_registry",
]

# preprocessor/graph/expr.py
from __future__ import annotations

import functools
from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Callable, Sequence
from typing import Any

from pysrc.core.errors import UnsupportedPlan
from pysrc.ops.mm_logkit import get_logger

logger = get_logger(__name__)

__all__ = [
    "Expr",
    "Column",
    "Literal",
    "OpExpr",
    "expr_factory",
    "register_expr",
    "register_builtin_builders",
    "ExprMeta",
    "get_polars_lowering",
    "register_polars_lowering",
]

# Registry for expression builders
_BUILDERS: dict[str, Callable[..., Expr]] = {}


def register_expr(op: str, builder: Callable[..., Expr]) -> None:
    """Register an expression builder for an op name."""
    if op in _BUILDERS:
        logger.debug(f"Overwriting builder for {op}")
    _BUILDERS[op] = builder


class ExprMeta(ABCMeta):
    """Metaclass that auto-registers Expr subclasses with an 'op' attribute."""

    def __new__(mcs, name, bases, dct):
        new_cls = super().__new__(mcs, name, bases, dct)
        if getattr(new_cls, "op", None) and new_cls.op is not None:
            register_expr(new_cls.op, new_cls)
        return new_cls


class Expr(ABC, metaclass=ExprMeta):
    """Base class for expression AST nodes."""

    op: str | None = None
    args: Sequence[Expr]
    params: dict[str, Any]

    def __init__(self, args: Sequence[Expr] = (), params: dict[str, Any] | None = None):
        self.args = tuple(_ensure_expr(a) for a in args)
        self.params = params or {}
        self._hash_cache: int | None = None
        self.validate()

    def validate(self) -> None:
        """Override to add validation logic."""
        pass

    def __add__(self, other: Expr | Any) -> Expr:
        return OpExpr("add", (self, _ensure_expr(other)))

    def __sub__(self, other: Expr | Any) -> Expr:
        return OpExpr("sub", (self, _ensure_expr(other)))

    def __mul__(self, other: Expr | Any) -> Expr:
        return OpExpr("mul", (self, _ensure_expr(other)))

    def __truediv__(self, other: Expr | Any) -> Expr:
        return OpExpr("div", (self, _ensure_expr(other)))

    def __neg__(self) -> Expr:
        return OpExpr("neg", (self,))

    def __pow__(self, other: Expr | Any) -> Expr:
        return OpExpr("pow", (self, _ensure_expr(other)))

    def __radd__(self, other: Expr | Any) -> Expr:
        return OpExpr("add", (_ensure_expr(other), self))

    def __rsub__(self, other: Expr | Any) -> Expr:
        return OpExpr("sub", (_ensure_expr(other), self))

    def __rmul__(self, other: Expr | Any) -> Expr:
        return OpExpr("mul", (_ensure_expr(other), self))

    def __rtruediv__(self, other: Expr | Any) -> Expr:
        return OpExpr("div", (_ensure_expr(other), self))

    def __rpow__(self, other: Expr | Any) -> Expr:
        return OpExpr("pow", (_ensure_expr(other), self))

    def __hash__(self) -> int:
        if self._hash_cache is None:
            self._hash_cache = hash(
                (self.op, tuple(hash(a) for a in self.args), frozenset(self.params.items()))
            )
        return self._hash_cache

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Expr):
            return False
        return self.op == other.op and self.args == other.args and self.params == other.params

    @abstractmethod
    def to_ir(self) -> dict[str, Any]:
        """Convert expression to intermediate representation."""
        pass

    def optimize(self) -> Expr:
        """Return optimized version of this expression."""
        return self


def _ensure_expr(x: Any) -> Expr:
    """Convert value to Expr if needed."""
    if isinstance(x, Expr):
        return x
    if isinstance(x, str):
        return Column(x)
    return Literal(x)


class Column(Expr):
    """Reference to a column by name."""

    op = "col"

    def __init__(self, name: str):
        self.name = name
        super().__init__(args=(), params={"name": name})

    def validate(self) -> None:
        if not isinstance(self.name, str):
            raise ValueError("Column name must be str")

    def to_ir(self) -> dict[str, Any]:
        return {"op": "col", "params": {"name": self.name}}

    def __repr__(self) -> str:
        return f"Column({self.name!r})"


class Literal(Expr):
    """A literal constant value."""

    op = "lit"

    def __init__(self, value: Any):
        self.value = value
        super().__init__(args=(), params={"value": value})

    def to_ir(self) -> dict[str, Any]:
        return {"op": "lit", "params": {"value": self.value}}

    def __repr__(self) -> str:
        return f"Literal({self.value!r})"


class OpExpr(Expr):
    """An operation applied to arguments."""

    def __init__(self, op: str, args: Sequence[Expr | Any], params: dict[str, Any] | None = None):
        self.op = op
        super().__init__(args, params)

    def to_ir(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "args": [a.to_ir() for a in self.args],
            "params": dict(self.params),
        }

    def optimize(self) -> Expr:
        """Optimize by constant folding."""
        optimized_args = [a.optimize() for a in self.args]

        # Constant folding for simple arithmetic
        if all(isinstance(a, Literal) for a in optimized_args):
            vals = [a.value for a in optimized_args]
            try:
                if self.op == "add":
                    return Literal(vals[0] + vals[1])
                if self.op == "sub":
                    return Literal(vals[0] - vals[1])
                if self.op == "mul":
                    return Literal(vals[0] * vals[1])
                if self.op == "div":
                    return Literal(vals[0] / vals[1])
                if self.op == "pow":
                    return Literal(vals[0] ** vals[1])
                if self.op == "neg":
                    return Literal(-vals[0])
            except (IndexError, ArithmeticError, TypeError):
                logger.debug("Constant fold failed for op=%s", self.op)

        return OpExpr(self.op, optimized_args, self.params)

    def __repr__(self) -> str:
        return f"OpExpr({self.op!r}, {self.args!r})"


def expr_factory(op: str, *args: Expr | Any, **params: Any) -> Expr:
    """Create an expression by op name."""
    builder = _BUILDERS.get(op)
    if builder is None:
        raise UnsupportedPlan(f"No builder for {op}")
    return builder(*args, **params)


def _register_default_builders() -> None:
    """Register built-in expression builders."""
    register_expr("col", Column)
    register_expr("lit", Literal)

    for op_name in ["add", "sub", "mul", "div", "neg", "pow"]:

        def _make_builder(op: str):
            return lambda *args, **params: OpExpr(op, args, params)

        register_expr(op_name, _make_builder(op_name))


_register_default_builders()


@functools.lru_cache(maxsize=1024)
def optimize(expr: Expr) -> Expr:
    """Cached expression optimization."""
    return expr.optimize()


def register_builtin_builders() -> None:
    """Re-register default builders (idempotent)."""
    _register_default_builders()


def create_sequence_lag(col: str, k: int, prefix: str = "lag") -> Expr:
    """Create a lag expression."""
    return expr_factory("lag", [Column(col)], k=int(k), prefix=prefix)


def create_rolling_mean(col: str, window: int) -> Expr:
    """Create a rolling mean expression."""
    return expr_factory("rolling_mean", [Column(col)], window=int(window))


def create_rolling_std(col: str, window: int) -> Expr:
    """Create a rolling std expression."""
    return expr_factory("rolling_std", [Column(col)], window=int(window))


def fuse_expressions(exprs: Sequence[Expr], fused_name: str | None = None) -> Expr:
    """Fuse multiple expressions into a single composite expression."""
    if not exprs:
        raise ValueError("Cannot fuse empty expr list")
    name = fused_name or f"fused_{exprs[0].op}"
    return expr_factory(name, exprs, sub_ir=[e.to_ir() for e in exprs])


# --------- POLARS LOWERINGS ---------


_POLARS_LOWERINGS: dict[str, Callable[..., object]] = {}


def register_polars_lowering(op_name: str, fn: Callable[..., object]) -> None:
    """Register a Polars lowering function for an op."""
    _POLARS_LOWERINGS[op_name] = fn


def get_polars_lowering(op_name: str) -> Callable[..., object] | None:
    """Get the Polars lowering function for an op."""
    return _POLARS_LOWERINGS.get(op_name)

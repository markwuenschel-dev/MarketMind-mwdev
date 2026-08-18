# utils/specs.py
from __future__ import annotations

import dataclasses
import hashlib
from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable
from dataclasses import dataclass, fields
from functools import reduce
from time import perf_counter
from typing import Any

from pysrc.ops.mm_logkit import get_logger

from .errors import UnsupportedAST

logger = get_logger(__name__)


@dataclass(frozen=True)
class Spec(ABC, Hashable):
    def __post_init__(self):
        # Only forbid None when no default and not Optional[T]
        for f in fields(self):
            val = getattr(self, f.name)
            if _no_default(f) and not _is_optional(f.type) and val is None:
                raise ValueError(f"{f.name} cannot be None")

    def __hash__(self) -> int:
        m = hashlib.sha256()
        for field in fields(self):
            val = getattr(self, field.name)
            m.update(str(val).encode())
        return int(m.hexdigest(), 16)

    @abstractmethod
    def to_backend_spec(self, backend: str) -> Any:
        """Keep Spec as data-only; execution layer interprets this."""
        return {"backend": backend, **{f.name: getattr(self, f.name) for f in fields(self)}}

    def validate(self) -> None:
        pass


@dataclass(frozen=True)
class WindowSpec(Spec):
    partition_by: list[str] | None = None
    order_by: list[str] | None = None
    preceding: int | None = None
    following: int | None = None
    min_periods: int = 1

    def validate(self) -> None:
        if self.preceding is not None and self.preceding < 0:
            raise ValueError("preceding must be non-negative")
        if self.following is not None and self.following < 0:
            raise ValueError("following must be non-negative")

    # Defer real lowering to the planner/execution backend.

    def to_backend_spec(self, backend: str) -> Any:
        """Return backend-specific window specification."""
        return {
            "backend": backend,
            "partition_by": self.partition_by,
            "order_by": self.order_by,
            "preceding": self.preceding,
            "following": self.following,
            "min_periods": self.min_periods,
        }

    def __add__(self, other: WindowSpec) -> WindowSpec:
        return WindowSpec(
            partition_by=(self.partition_by or []) + (other.partition_by or []),
            order_by=(self.order_by or []) + (other.order_by or []),
            preceding=self.preceding or other.preceding,
            following=self.following or other.following,
            min_periods=max(self.min_periods, other.min_periods),
        )


@dataclass(frozen=True)
class GroupSpec(Spec):
    by: list[str]
    as_index: bool = True
    sort: bool = False

    def validate(self) -> None:
        if not self.by:
            raise ValueError("by cannot be empty")

    def to_backend_spec(self, backend: str) -> Any:
        """Return backend-specific group specification."""
        return {
            "backend": backend,
            "by": self.by,
            "as_index": self.as_index,
            "sort": self.sort,
        }

    def __add__(self, other: GroupSpec) -> GroupSpec:
        return GroupSpec(
            by=self.by + other.by,
            as_index=self.as_index and other.as_index,
            sort=self.sort or other.sort,
        )


class SpecFactory:
    registry: dict[str, Callable[..., Spec]] = {"window": WindowSpec, "group": GroupSpec}

    @classmethod
    def register(cls, name: str, builder: Callable[..., Spec]):
        cls.registry[name] = builder

    @classmethod
    def build(cls, name: str, **kwargs) -> Spec:
        if name not in cls.registry:
            raise UnsupportedAST(f"Spec {name} not registered")
        spec = cls.registry[name](**kwargs)
        spec.validate()
        return spec

    @classmethod
    def compose(cls, *specs: Spec) -> Spec:
        if not specs:
            raise ValueError("No specs to compose")
        return reduce(lambda a, b: a + b, specs)


def profile_spec(func: Callable) -> Callable:
    metrics: dict[tuple[Hashable, str], float] = {}  # Key: (hash, backend)

    def wrapper(self, *args, **kwargs):
        backend = args[0] if args else "unknown"
        key = (hash(self), backend)
        start = perf_counter()
        try:
            result = func(self, *args, **kwargs)
        except Exception as e:
            logger.error(f"Spec {type(self).__name__} failed for {backend}: {e}")
            raise
        duration = perf_counter() - start
        if key not in metrics or duration < metrics[key]:
            metrics[key] = duration
            logger.info(f"Evolved: Updated best time for spec {key} to {duration}s")
        return result

    return wrapper


# Decorate methods
WindowSpec.to_backend_spec = profile_spec(WindowSpec.to_backend_spec)
GroupSpec.to_backend_spec = profile_spec(GroupSpec.to_backend_spec)


# ---- infra ----
def _is_optional(typ) -> bool:
    import typing

    origin = getattr(typing, "get_origin", lambda x: None)(typ)
    args = getattr(typing, "get_args", lambda x: ())(typ)
    return origin is getattr(typing, "Union", None) and type(None) in args


def _no_default(f) -> bool:
    return f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING


# Register built-in specs
SpecFactory.register("window", WindowSpec)
SpecFactory.register("group", GroupSpec)

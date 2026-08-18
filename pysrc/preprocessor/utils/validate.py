# utils/validate.py
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from pysrc.ops.mm_logkit import get_logger

from .cuda_runtime import capabilities
from .errors import SchemaMismatch

logger = get_logger(__name__)

try:
    import cudf
except Exception:
    cudf = None
try:
    import polars as pl
except Exception:
    pl = None


class Validator(ABC):
    @abstractmethod
    def validate(self, obj: Any) -> None:
        pass


class SchemaValidator(Validator):
    def __init__(self, expected: Mapping[str, str], strict: bool = False):
        self.expected = expected
        self.strict = strict

    def validate(self, df: Any) -> None:
        actual = self._get_schema(df)
        missing = {k: v for k, v in self.expected.items() if k not in actual}
        extra = {k: v for k, v in actual.items() if k not in self.expected} if self.strict else {}
        mismatched = {
            k: (self.expected[k], actual[k])
            for k in self.expected
            if k in actual and str(self.expected[k]) != str(actual[k])
        }
        if missing or extra or mismatched:
            raise SchemaMismatch(
                "Schema validation failed",
                {"missing": missing, "extra": extra, "mismatched": mismatched},
            )

    def _get_schema(self, df: Any) -> dict[str, str]:
        caps = capabilities()
        if cudf and isinstance(df, cudf.DataFrame) and caps.has_cudf:
            return {c: str(df[c].dtype) for c in df.columns}
        if pl and isinstance(df, pl.DataFrame):
            schema = getattr(df, "schema", None)
            if schema is not None and hasattr(schema, "get"):
                return {c: str(schema.get(c, "unknown")) for c in df.columns}
            if hasattr(df, "dtypes") and hasattr(df, "columns"):
                return dict(zip(df.columns, [str(t) for t in df.dtypes], strict=False))
        if hasattr(df, "dtypes") and hasattr(df, "columns"):
            dtypes = df.dtypes
            if isinstance(dtypes, dict):
                return {c: str(dtypes[c]) for c in df.columns}
            return dict(zip(df.columns, [str(t) for t in dtypes], strict=False))
        raise ValueError("Unsupported DataFrame type")


class PlanValidator(Validator):
    def validate(self, graph: dict[Any, Sequence[Any]]) -> None:
        visiting, visited = set(), set()

        def dfs(n):
            if n in visiting:
                raise SchemaMismatch(f"Cycle detected at {n}")
            if n in visited:
                return
            visiting.add(n)
            for m in graph.get(n, ()):
                dfs(m)
            visiting.remove(n)
            visited.add(n)

        for node in graph:
            dfs(node)
        # Add more checks: node compatibility, etc.


class ValidatorFactory:
    @staticmethod
    def schema(expected: Mapping[str, str], strict: bool = False) -> Validator:
        return SchemaValidator(expected, strict)

    @staticmethod
    def plan() -> Validator:
        return PlanValidator()


def schema_checks(df, expected: Mapping[str, str], strict: bool = False) -> None:
    ValidatorFactory.schema(expected, strict).validate(df)


def plan_checks(graph: dict[Any, Sequence[Any]]) -> None:
    ValidatorFactory.plan().validate(graph)

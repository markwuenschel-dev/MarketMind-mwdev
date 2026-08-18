from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import polars as pl

from pysrc.core.errors import DataValidationError

__all__ = ["Bar", "Tick", "MarketDataFrameSchema"]


@dataclass(frozen=True)
class Bar:
    timestamp: pl.Datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Tick:
    timestamp: pl.Datetime
    price: float
    size: float
    side: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MarketDataFrameSchema:
    required_columns: Mapping[str, pl.DataType] = field(default_factory=dict)
    optional_columns: Mapping[str, pl.DataType] = field(default_factory=dict)
    strict: bool = False
    unknown_ok: bool = False

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> MarketDataFrameSchema:
        return cls(
            required_columns={
                str(column): _dtype_from_name(dtype_name)
                for column, dtype_name in raw.get("required_columns", {}).items()
            },
            optional_columns={
                str(column): _dtype_from_name(dtype_name)
                for column, dtype_name in raw.get("optional_columns", {}).items()
            },
            strict=bool(raw.get("strict", False)),
            unknown_ok=bool(raw.get("unknown_ok", False)),
        )

    def validate(
        self,
        df: pl.DataFrame,
        *,
        strict: bool | None = None,
        unknown_ok: bool | None = None,
    ) -> tuple[bool, list[str]]:
        use_strict = self.strict if strict is None else strict
        use_unknown_ok = self.unknown_ok if unknown_ok is None else unknown_ok
        errors: list[str] = []
        columns = set(df.columns)

        for column, dtype in self.required_columns.items():
            if column not in columns:
                errors.append(f"missing:{column}")
                continue
            actual_dtype = df[column].dtype
            if actual_dtype != dtype:
                errors.append(f"dtype:{column}:expected={dtype} got={actual_dtype}")

        for column, dtype in self.optional_columns.items():
            if column in columns and df[column].dtype != dtype:
                errors.append(f"dtype:{column}:expected={dtype} got={df[column].dtype}")

        if use_strict and not use_unknown_ok:
            allowed = set(self.required_columns) | set(self.optional_columns)
            for extra in sorted(columns - allowed):
                errors.append(f"unknown:{extra}")

        return len(errors) == 0, errors

    def assert_valid(
        self,
        df: pl.DataFrame,
        *,
        strict: bool | None = None,
        unknown_ok: bool | None = None,
        label: str = "frame",
    ) -> None:
        ok, errors = self.validate(df, strict=strict, unknown_ok=unknown_ok)
        if not ok:
            raise DataValidationError(
                f"{label} failed schema validation",
                details={"errors": errors, "label": label},
            )

    def to_payload(self) -> dict[str, Any]:
        return {
            "required_columns": {
                column: str(dtype) for column, dtype in self.required_columns.items()
            },
            "optional_columns": {
                column: str(dtype) for column, dtype in self.optional_columns.items()
            },
            "strict": self.strict,
            "unknown_ok": self.unknown_ok,
        }


def _dtype_from_name(raw: Any) -> pl.DataType:
    if isinstance(raw, pl.DataType):
        return raw
    if not isinstance(raw, str):
        raise DataValidationError(
            "Schema dtype names must be strings",
            details={"dtype_type": type(raw).__name__},
        )
    candidate = raw.split(".")[-1]
    if hasattr(pl, candidate):
        dtype = getattr(pl, candidate)
        if isinstance(dtype, pl.DataType):
            return dtype
        if callable(dtype):
            resolved = dtype()
            if isinstance(resolved, pl.DataType):
                return resolved
    raise DataValidationError(
        "Unknown Polars dtype in cleaning schema",
        details={"dtype": raw},
    )

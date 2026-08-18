from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import polars as pl

from pysrc.core.errors import DataValidationError


@dataclass(frozen=True)
class GovernedColumns:
    frame: pl.DataFrame
    lineage: dict[str, Any]
    warnings: tuple[str, ...] = ()


@runtime_checkable
class GovernedColumnProvider(Protocol):
    provider_name: str

    def materialize(
        self,
        df: pl.DataFrame,
        *,
        context: Any,
        params: Any,
    ) -> GovernedColumns: ...


class FailClosedProvider:
    def __init__(self, provider_name: str) -> None:
        self.provider_name = provider_name

    def materialize(
        self,
        df: pl.DataFrame,
        *,
        context: Any,
        params: Any,
    ) -> GovernedColumns:
        _ = (df, context, params)
        raise DataValidationError(
            "Governed cleaning provider is unavailable",
            details={"provider_name": self.provider_name},
        )


def default_cleaning_providers() -> dict[str, GovernedColumnProvider]:
    from pysrc.pipeline.stages.cleaning.core.fred_macro_provider import FredMacroGovernedProvider

    return {
        "macro": FredMacroGovernedProvider(),
        "altdata": FailClosedProvider("altdata"),
        "sentiment.vader": FailClosedProvider("sentiment.vader"),
        "sentiment.finbert": FailClosedProvider("sentiment.finbert"),
    }

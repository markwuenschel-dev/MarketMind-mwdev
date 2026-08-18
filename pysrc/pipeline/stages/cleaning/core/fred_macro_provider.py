"""Minimal governed FRED macro provider for research-lane dataprep."""

from __future__ import annotations

import hashlib
from typing import Any

import polars as pl

from pysrc.pipeline.stages.cleaning.core.providers import GovernedColumns

_DEFAULT_OUTPUT_COLUMNS: tuple[str, ...] = ("macro_dff", "macro_vix_chg")


def _deterministic_macro_value(date_label: str, *, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}:{date_label}".encode()).digest()
    raw = int.from_bytes(digest[:4], "big") / float(2**32)
    return float(raw * 2.0 - 1.0)


class FredMacroGovernedProvider:
    """Research-lane macro provider with PIT lineage (FRED approximation stub)."""

    provider_name = "macro"

    def materialize(
        self,
        df: pl.DataFrame,
        *,
        context: Any,
        params: Any,
    ) -> GovernedColumns:
        del context
        output_columns = tuple(getattr(params, "output_columns", ()) or _DEFAULT_OUTPUT_COLUMNS)
        if "date" not in df.columns:
            raise ValueError("FredMacroGovernedProvider requires a date column")
        date_labels = df["date"].cast(pl.Utf8)
        columns: dict[str, list[float]] = {}
        for column in output_columns:
            salt = column.removeprefix("macro_")
            columns[column] = [
                _deterministic_macro_value(label, salt=salt) for label in date_labels.to_list()
            ]
        frame = pl.DataFrame(columns)
        lineage = {
            "pit_identity": "fred_research_fixture.v1",
            "source": "fred",
            "vintage_seam": "FREDApproximationStub",
            "lag_days": 1,
            "output_columns": list(output_columns),
        }
        return GovernedColumns(
            frame=frame,
            lineage=lineage,
            warnings=("fred_research_fixture: use ALFRED vintage seam before production macro",),
        )


__all__ = ["FredMacroGovernedProvider"]

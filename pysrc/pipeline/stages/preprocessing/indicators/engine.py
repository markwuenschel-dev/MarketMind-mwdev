"""Unified indicator engine for pipeline materialization and W3-B."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pysrc.pipeline.stages.preprocessing.indicators.config import IndicatorLibraryConfig
from pysrc.pipeline.stages.preprocessing.indicators.pandas_ta_classic_provider import (
    IndicatorProviderResult,
    compute_pandas_ta_classic_features,
    load_pipeline_indicator_features,
)
from pysrc.pipeline.stages.preprocessing.indicators.schema import (
    W3B_INDICATOR_IDS,
    W3B_INDICATOR_SCHEMA_VERSION,
)

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True, slots=True)
class IndicatorEngine:
    """Compute or load W3-B indicator columns behind one interface."""

    library: IndicatorLibraryConfig = IndicatorLibraryConfig()

    @property
    def indicator_ids(self) -> tuple[str, ...]:
        return W3B_INDICATOR_IDS

    @property
    def schema_version(self) -> str:
        return W3B_INDICATOR_SCHEMA_VERSION

    def compute(
        self,
        panel: pd.DataFrame,
        *,
        workers: int = 1,
        copy_input: bool = False,
        ta_scratch_path: Path | None = None,
    ) -> IndicatorProviderResult:
        return compute_pandas_ta_classic_features(
            panel,
            self.library,
            workers=max(1, int(workers)),
            copy_input=copy_input,
            ta_scratch_path=ta_scratch_path,
        )

    def load(
        self,
        panel_path: Path,
        *,
        key_columns: tuple[str, ...] = ("date", "instrument"),
    ) -> IndicatorProviderResult:
        return load_pipeline_indicator_features(
            Path(panel_path),
            key_columns=key_columns,
        )


__all__ = ["IndicatorEngine", "IndicatorProviderResult"]

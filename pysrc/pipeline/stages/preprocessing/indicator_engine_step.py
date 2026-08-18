"""Research preprocessing step: W3-B indicators via IndicatorEngine."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pysrc.core.runtime.optional_imports import optional_import
from pysrc.pipeline.core.pipeline_core_base import PipelineStep
from pysrc.pipeline.core.pipeline_core_context import PipelineContext
from pysrc.pipeline.materializers.indicator_panel import attach_panel_supervision_columns
from pysrc.pipeline.stages.preprocessing.indicators.config import IndicatorLibraryConfig
from pysrc.pipeline.stages.preprocessing.indicators.engine import IndicatorEngine

pl = optional_import("polars")
pd = optional_import("pandas")

_DEFAULT_INTERVAL = "daily"


class IndicatorEngineStep(PipelineStep):
    STEP_NAME = "IndicatorEngineStep"
    STEP_VERSION = "1.0.0"

    def __init__(self, **cfg: Any) -> None:
        super().__init__()
        self.cfg = dict(cfg)

    def fit_transform(self, df: Any, ctx: PipelineContext) -> Any:  # noqa: ARG002
        if pd is None:
            raise RuntimeError("pandas is required for IndicatorEngineStep")

        if pl is not None and isinstance(df, pl.DataFrame):
            frame = df.to_pandas()
            return_polars = True
        elif pd is not None and isinstance(df, pd.DataFrame):
            frame = df
            return_polars = False
        else:
            frame = pd.DataFrame(df)
            return_polars = False

        library = IndicatorLibraryConfig()
        workers = max(1, int(self.cfg.get("workers", 1)))
        scratch_path = self.cfg.get("ta_scratch_path")
        scratch = Path(scratch_path) if scratch_path else None

        engine = IndicatorEngine(library)
        result = engine.compute(
            frame,
            workers=workers,
            copy_input=False,
            ta_scratch_path=scratch,
        )
        out = attach_panel_supervision_columns(frame, result.features.copy())
        if "interval" not in out.columns:
            out["interval"] = _DEFAULT_INTERVAL

        if return_polars and pl is not None:
            return pl.from_pandas(out)
        return out

    def apply_batch_pandas(self, df: Any, ctx: PipelineContext) -> Any:  # noqa: ARG002
        return self.fit_transform(df, ctx)


__all__ = ["IndicatorEngineStep"]

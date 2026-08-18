from __future__ import annotations

import pandas as pd

from pysrc.core.runtime.optional_imports import optional_import
from pysrc.pipeline.core.pipeline_core_base import PipelineStep
from pysrc.pipeline.core.pipeline_core_context import PipelineContext

pl = optional_import("polars")


class BacktestingSplitNormalizerStep(PipelineStep):
    is_fast = True

    def __init__(self, cfg):
        super().__init__(name="BacktestingSplitNormalizerStep")
        self.enabled = cfg.enabled
        self.split_ratio = cfg.split_ratio

    def apply_batch(self, lf: pl.LazyFrame, ctx: PipelineContext) -> pl.LazyFrame:
        del ctx
        if not self.enabled:
            return lf
        df = lf.collect()
        train_size = int(len(df) * self.split_ratio)
        if train_size > 0:
            df = df.with_columns(pl.lit(train_size).alias("bt_split_idx"))
        return df.lazy()

    def apply_batch_pandas(self, df: pd.DataFrame, ctx: PipelineContext) -> pd.DataFrame:
        del ctx
        if not self.enabled:
            return df
        out = df.copy()
        train_size = int(len(out) * self.split_ratio)
        out.attrs["bt_split_idx"] = train_size
        if train_size > 0:
            out.attrs["bt_train_end"] = out.index[train_size - 1]
        return out

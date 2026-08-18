# py/pipeline/stages/preprocessing/_base_plan_step.py
from __future__ import annotations

from pysrc.pipeline.core.pipeline_core_base import PipelineStep
from pysrc.preprocessor.api import PlanSpec, Preprocessor


class PlanStep(PipelineStep):
    STEP_VERSION = "1.0.0"

    def __init__(self, *, backend: str | None = "auto", device: int = 0, nvtx: bool = False, **cfg):
        self.backend, self.device, self.nvtx = backend, device, nvtx
        self.cfg = cfg  # raw step pipeline_config; subclasses shape it into PlanSpec

    def _build_spec(self) -> PlanSpec:
        raise NotImplementedError

    def fit_transform(self, df, ctx):
        spec = self._build_spec()
        pp = Preprocessor(backend=self.backend or "auto", device=self.device, nvtx=self.nvtx)
        X, y, meta = pp.materialize(df, spec)
        return X, y, meta

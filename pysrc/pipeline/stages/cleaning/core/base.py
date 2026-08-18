from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from pysrc.core.runtime.optional_imports import optional_import

pd = optional_import("pandas")
pl = optional_import("polars")
from pysrc.pipeline.stages.cleaning.core.contracts import (
    CleaningMutationSummary,
    CleaningPipelineState,
    CleaningRuntimeContext,
    CleaningStepResult,
    CleaningStepSpec,
)


def _to_polars(df: Any) -> pl.DataFrame:
    if pl is None:
        raise RuntimeError("polars is required for cleaning execution")
    if isinstance(df, pl.DataFrame):
        return df
    if pd is not None and isinstance(df, pd.DataFrame):
        return pl.from_pandas(df)
    raise TypeError(f"Unsupported cleaning frame type: {type(df)!r}")


class CleaningStep(ABC):
    STEP_TYPE = ""
    STEP_VERSION = "1"

    def __init__(
        self, *, spec: CleaningStepSpec, params: BaseModel, registration: Any | None = None
    ) -> None:
        self.spec = spec
        self.params = params
        self.registration = registration

    def apply(
        self,
        df: Any,
        *,
        state: CleaningPipelineState | None = None,
        context: CleaningRuntimeContext | None = None,
    ) -> CleaningStepResult:
        runtime_state = state or CleaningPipelineState()
        runtime_context = context or CleaningRuntimeContext(
            run_id="adhoc-cleaning",
            determinism_tier=self.spec.determinism_tier,
            seed_lineage="adhoc",
            pit_boundary="",
            governance_mode=self.spec.governance_mode,
        )
        result = self._apply(_to_polars(df), runtime_state, runtime_context)
        result.apply_to_state()
        return result

    @abstractmethod
    def _apply(
        self,
        df: pl.DataFrame,
        state: CleaningPipelineState,
        context: CleaningRuntimeContext,
    ) -> CleaningStepResult: ...

    def _result(
        self,
        df: pl.DataFrame,
        state: CleaningPipelineState,
        *,
        warnings: list[str] | None = None,
        metrics: dict[str, Any] | None = None,
        provider_lineage: dict[str, Any] | None = None,
        validation_failures: list[str] | None = None,
        fallback_events: list[dict[str, Any]] | None = None,
        mutation: CleaningMutationSummary | None = None,
    ) -> CleaningStepResult:
        return CleaningStepResult(
            frame=df,
            state=state,
            warnings=list(warnings or []),
            metrics=dict(metrics or {}),
            provider_lineage=dict(provider_lineage or {}),
            validation_failures=list(validation_failures or []),
            fallback_events=list(fallback_events or []),
            mutation=mutation or CleaningMutationSummary(rows_in=df.height, rows_out=df.height),
        )

    def _cell_mutation(
        self,
        before_rows: int,
        after_rows: int,
        *,
        rows_with_mutations: int = 0,
        cells_mutated: int = 0,
    ) -> CleaningMutationSummary:
        return CleaningMutationSummary(
            rows_in=before_rows,
            rows_out=after_rows,
            rows_removed=max(before_rows - after_rows, 0),
            rows_with_mutations=max(rows_with_mutations, 0),
            cells_mutated=max(cells_mutated, 0),
        )

    def _additive_mutation(
        self,
        before_rows: int,
        after_rows: int,
        *,
        added_columns: int,
    ) -> CleaningMutationSummary:
        active_rows = min(before_rows, after_rows) if added_columns > 0 else 0
        return self._cell_mutation(
            before_rows,
            after_rows,
            rows_with_mutations=active_rows,
            cells_mutated=active_rows * max(added_columns, 0),
        )

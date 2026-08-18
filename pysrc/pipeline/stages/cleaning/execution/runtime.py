from __future__ import annotations

from typing import Any

from pysrc.core.runtime.optional_imports import optional_import

pd = optional_import("pandas")
pl = optional_import("polars")
from pysrc.core.errors import DataValidationError
from pysrc.pipeline.stages.cleaning.core.contracts import (
    BuiltCleaningPipeline,
    CleaningMutationSummary,
    CleaningPipelineState,
    CleaningRuntimeContext,
    CleaningStepResult,
    _normalize_jsonable,
)
from pysrc.pipeline.stages.cleaning.core.providers import default_cleaning_providers


def _to_polars(df: Any) -> pl.DataFrame:
    if pl is None:
        raise RuntimeError("polars is required for cleaning execution")
    if isinstance(df, pl.DataFrame):
        return df
    if pd is not None and isinstance(df, pd.DataFrame):
        return pl.from_pandas(df)
    raise TypeError(f"Unsupported cleaning frame type: {type(df)!r}")


class CleaningPipelineRunner:
    def __init__(
        self,
        pipeline: BuiltCleaningPipeline,
        *,
        state: CleaningPipelineState | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.state = state or CleaningPipelineState()
        self.last_context: CleaningRuntimeContext | None = None
        self.last_result: CleaningStepResult | None = None

    def default_context(self, *, streaming: bool = False) -> CleaningRuntimeContext:
        return CleaningRuntimeContext(
            run_id="cleaning-local",
            determinism_tier=self.pipeline.spec.determinism_tier,
            seed_lineage=self.pipeline.spec.seed_lineage,
            pit_boundary=self.pipeline.spec.pit_boundary,
            governance_mode=self.pipeline.spec.governance_mode,
            providers=default_cleaning_providers(),
            streaming=streaming,
            registry_state_hash=self.pipeline.registry_state_hash,
        )

    def run(
        self,
        df: Any,
        *,
        context: CleaningRuntimeContext | None = None,
    ) -> CleaningStepResult:
        current = _to_polars(df)
        runtime_context = context or self.default_context()
        runtime_context.registry_state_hash = self.pipeline.registry_state_hash
        step_reports: list[dict[str, Any]] = []
        total_mutation = CleaningMutationSummary(rows_in=current.height, rows_out=current.height)

        for step in self.pipeline.steps:
            registration = step.registration
            if registration is None:
                raise DataValidationError(
                    "Cleaning step is missing registration metadata",
                    details={"step_id": step.spec.step_id},
                )
            missing_providers = [
                provider
                for provider in registration.provider_requirements
                if provider not in runtime_context.providers
            ]
            if missing_providers:
                raise DataValidationError(
                    "Cleaning step requires unavailable providers",
                    details={
                        "step_id": step.spec.step_id,
                        "providers": missing_providers,
                    },
                )

            step.spec.input_contract.validate(current, label=f"{step.spec.step_id}.input")
            result = step._apply(current, self.state, runtime_context)
            result.apply_to_state()
            step.spec.output_contract.validate(
                result.frame,
                label=f"{step.spec.step_id}.output",
            )
            step_mutation = result.mutation
            total_mutation = CleaningMutationSummary(
                rows_in=total_mutation.rows_in,
                rows_out=step_mutation.rows_out,
                rows_removed=total_mutation.rows_removed + step_mutation.rows_removed,
                rows_with_mutations=total_mutation.rows_with_mutations
                + step_mutation.rows_with_mutations,
                cells_mutated=total_mutation.cells_mutated + step_mutation.cells_mutated,
            )
            step_reports.append(
                {
                    "step_id": step.spec.step_id,
                    "step_type": step.spec.step_type,
                    "version": step.spec.version,
                    "enabled": step.spec.enabled,
                    "determinism_tier": step.spec.determinism_tier.value,
                    "governance_mode": step.spec.governance_mode.value,
                    "stateful": registration.stateful,
                    "provider_requirements": list(registration.provider_requirements),
                    "mutation_summary": step_mutation.to_payload(),
                    "warnings": list(result.warnings),
                    "validation_failures": list(result.validation_failures),
                    "fallback_events": _normalize_jsonable(result.fallback_events),
                    "metrics": _normalize_jsonable(result.metrics),
                }
            )
            current = result.frame

        final_result = CleaningStepResult(
            frame=current,
            state=self.state,
            warnings=list(self.state.warnings),
            metrics={
                "step_reports": step_reports,
                "final_contract_status": {
                    "ok": True,
                    "columns": list(current.columns),
                    "row_count": current.height,
                },
            },
            provider_lineage=dict(self.state.provider_lineage),
            fallback_events=list(self.state.fallback_events),
            mutation=total_mutation,
        )
        self.last_context = runtime_context
        self.last_result = final_result
        return final_result

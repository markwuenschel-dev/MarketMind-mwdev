from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from typing import Any

from pysrc.pipeline.stages.cleaning.core.config_models import (
    pipeline_spec_from_external_cleaning_config,
    pipeline_spec_from_external_pipeline_spec,
)
from pysrc.pipeline.stages.cleaning.core.contracts import (
    BuiltCleaningPipeline,
    CleaningPipelineSpec,
    CleaningStepSpec,
    FrameContract,
    _normalize_jsonable,
)
from pysrc.pipeline.stages.cleaning.core.registry import (
    registry_state_hash,
    resolve_cleaning_step,
)


def build_cleaning_pipeline(
    spec: CleaningPipelineSpec | Mapping[str, Any],
) -> BuiltCleaningPipeline:
    pipeline_spec = (
        spec
        if isinstance(spec, CleaningPipelineSpec)
        else (
            pipeline_spec_from_external_pipeline_spec(spec)
            if "steps" in spec and "combos" not in spec
            else pipeline_spec_from_external_cleaning_config(spec)
        )
    )
    built_steps: list[Any] = []
    normalized_specs: list[CleaningStepSpec] = []
    current_registry_state_hash = registry_state_hash()

    for step_spec in pipeline_spec.steps:
        registration = resolve_cleaning_step(step_spec.step_type, step_spec.version)
        params_model = registration.params_model.model_validate(step_spec.params)
        normalized_params = _normalize_jsonable(params_model.model_dump(mode="python"))
        normalized_spec = dataclasses.replace(
            step_spec,
            params=normalized_params,
            input_contract=(
                registration.input_contract
                if step_spec.input_contract == FrameContract()
                else step_spec.input_contract
            ),
            output_contract=(
                registration.output_contract
                if step_spec.output_contract == FrameContract()
                else step_spec.output_contract
            ),
        )
        normalized_specs.append(normalized_spec)
        if normalized_spec.enabled:
            built_steps.append(
                registration.step_cls(
                    spec=normalized_spec,
                    params=params_model,
                    registration=registration,
                )
            )

    normalized_pipeline_spec = dataclasses.replace(pipeline_spec, steps=tuple(normalized_specs))
    plan_hash = _compute_plan_hash(normalized_pipeline_spec, current_registry_state_hash)
    return BuiltCleaningPipeline(
        spec=normalized_pipeline_spec,
        steps=tuple(built_steps),
        plan_hash=plan_hash,
        registry_state_hash=current_registry_state_hash,
    )


def _compute_plan_hash(spec: CleaningPipelineSpec, current_registry_state_hash: str) -> str:
    payload = json.dumps(
        {
            "registry_state_hash": current_registry_state_hash,
            "spec": spec.to_payload(),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

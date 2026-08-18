from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, StrEnum
from typing import Any

import polars as pl

from pysrc.core.errors import DataValidationError
from pysrc.pipeline.stages.cleaning.validators.contracts import MarketDataFrameSchema


class CleaningDeterminismTier(StrEnum):
    D0 = "d0"
    D1 = "d1"
    D2 = "d2"
    D3 = "d3"


class GovernanceMode(StrEnum):
    GOVERNED = "governed"
    NONGOVERNED = "nongoverned"


@dataclass(frozen=True)
class FrameContract:
    required_columns: tuple[str, ...] = ()
    optional_columns: tuple[str, ...] = ()
    schema: MarketDataFrameSchema | None = None
    strict: bool = False
    unknown_ok: bool = True

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> FrameContract:
        if raw is None:
            return cls()
        if isinstance(raw, FrameContract):
            return raw
        schema = raw.get("schema")
        if isinstance(schema, Mapping):
            schema = MarketDataFrameSchema.from_mapping(schema)
        required_columns = tuple(str(col) for col in raw.get("required_columns", ()))
        optional_columns = tuple(str(col) for col in raw.get("optional_columns", ()))
        return cls(
            required_columns=required_columns,
            optional_columns=optional_columns,
            schema=schema,
            strict=bool(raw.get("strict", False)),
            unknown_ok=bool(raw.get("unknown_ok", True)),
        )

    def validate(self, df: pl.DataFrame, *, label: str) -> None:
        errors: list[str] = []
        columns = set(df.columns)
        for column in self.required_columns:
            if column not in columns:
                errors.append(f"missing:{column}")

        if self.schema is not None:
            ok, schema_errors = self.schema.validate(
                df,
                strict=self.strict,
                unknown_ok=self.unknown_ok,
            )
            if not ok:
                errors.extend(schema_errors)
        elif self.strict and not self.unknown_ok:
            allowed = set(self.required_columns) | set(self.optional_columns)
            for extra in sorted(columns - allowed):
                errors.append(f"unknown:{extra}")

        if errors:
            raise DataValidationError(
                f"{label} failed frame contract validation",
                details={"errors": errors, "label": label},
            )

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "required_columns": list(self.required_columns),
            "optional_columns": list(self.optional_columns),
            "strict": self.strict,
            "unknown_ok": self.unknown_ok,
        }
        if self.schema is not None:
            payload["schema"] = self.schema.to_payload()
        return payload


@dataclass(frozen=True)
class CleaningStepSpec:
    step_id: str
    step_type: str
    version: str
    enabled: bool = True
    params: Mapping[str, Any] = field(default_factory=dict)
    input_contract: FrameContract = field(default_factory=FrameContract)
    output_contract: FrameContract = field(default_factory=FrameContract)
    determinism_tier: CleaningDeterminismTier = CleaningDeterminismTier.D1
    governance_mode: GovernanceMode = GovernanceMode.GOVERNED
    fallback_policy: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        default_governance_mode: GovernanceMode = GovernanceMode.GOVERNED,
        default_determinism_tier: CleaningDeterminismTier = CleaningDeterminismTier.D1,
    ) -> CleaningStepSpec:
        if not isinstance(raw, Mapping):
            raise DataValidationError(
                "Cleaning step specifications must be mappings",
                details={"step_spec_type": type(raw).__name__},
            )
        missing = [
            key
            for key in ("step_id", "step_type", "version")
            if key not in raw or not str(raw[key]).strip()
        ]
        if missing:
            raise DataValidationError(
                "Cleaning step specification is missing required keys",
                details={"missing": missing, "raw": dict(raw)},
            )
        determinism_raw = str(raw.get("determinism_tier", default_determinism_tier.value)).lower()
        governance_raw = str(raw.get("governance_mode", default_governance_mode.value)).lower()
        return cls(
            step_id=str(raw["step_id"]),
            step_type=str(raw["step_type"]),
            version=str(raw["version"]),
            enabled=bool(raw.get("enabled", True)),
            params=dict(raw.get("params", {})),
            input_contract=FrameContract.from_mapping(raw.get("input_contract")),
            output_contract=FrameContract.from_mapping(raw.get("output_contract")),
            determinism_tier=CleaningDeterminismTier(determinism_raw),
            governance_mode=GovernanceMode(governance_raw),
            fallback_policy=dict(raw.get("fallback_policy", {})),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "step_type": self.step_type,
            "version": self.version,
            "enabled": self.enabled,
            "params": _normalize_jsonable(self.params),
            "input_contract": self.input_contract.to_payload(),
            "output_contract": self.output_contract.to_payload(),
            "determinism_tier": self.determinism_tier.value,
            "governance_mode": self.governance_mode.value,
            "fallback_policy": _normalize_jsonable(self.fallback_policy),
        }


@dataclass(frozen=True)
class CleaningPipelineSpec:
    steps: tuple[CleaningStepSpec, ...]
    determinism_tier: CleaningDeterminismTier = CleaningDeterminismTier.D1
    seed_lineage: str = ""
    pit_boundary: str = ""
    governance_mode: GovernanceMode = GovernanceMode.GOVERNED
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> CleaningPipelineSpec:
        steps = tuple(
            CleaningStepSpec.from_mapping(
                step,
                default_governance_mode=GovernanceMode(
                    str(raw.get("governance_mode", GovernanceMode.GOVERNED.value)).lower()
                ),
                default_determinism_tier=CleaningDeterminismTier(
                    str(raw.get("determinism_tier", CleaningDeterminismTier.D1.value)).lower()
                ),
            )
            for step in raw.get("steps", ())
        )
        determinism_tier = CleaningDeterminismTier(
            str(raw.get("determinism_tier", CleaningDeterminismTier.D1.value)).lower()
        )
        governance_mode = GovernanceMode(
            str(raw.get("governance_mode", GovernanceMode.GOVERNED.value)).lower()
        )
        return cls(
            steps=steps,
            determinism_tier=determinism_tier,
            seed_lineage=str(raw.get("seed_lineage", "")),
            pit_boundary=str(raw.get("pit_boundary", "")),
            governance_mode=governance_mode,
            metadata=dict(raw.get("metadata", {})),
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "steps": [step.to_payload() for step in self.steps],
            "determinism_tier": self.determinism_tier.value,
            "seed_lineage": self.seed_lineage,
            "pit_boundary": self.pit_boundary,
            "governance_mode": self.governance_mode.value,
            "metadata": _normalize_jsonable(self.metadata),
        }


@dataclass
class CleaningPipelineState:
    step_state: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    fallback_events: list[dict[str, Any]] = field(default_factory=list)
    provider_lineage: dict[str, dict[str, Any]] = field(default_factory=dict)
    validation_failures: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class CleaningMutationSummary:
    rows_in: int = 0
    rows_out: int = 0
    rows_removed: int = 0
    rows_with_mutations: int = 0
    cells_mutated: int = 0

    def to_payload(self) -> dict[str, int]:
        return {
            "rows_in": int(self.rows_in),
            "rows_out": int(self.rows_out),
            "rows_removed": int(self.rows_removed),
            "rows_with_mutations": int(self.rows_with_mutations),
            "cells_mutated": int(self.cells_mutated),
        }


@dataclass
class CleaningRuntimeContext:
    run_id: str
    determinism_tier: CleaningDeterminismTier
    seed_lineage: str
    pit_boundary: str
    governance_mode: GovernanceMode
    providers: MutableMapping[str, Any] = field(default_factory=dict)
    streaming: bool = False
    registry_state_hash: str = ""

    def provider(self, key: str) -> Any:
        if key not in self.providers:
            raise DataValidationError(
                "Required cleaning provider is not configured",
                details={"provider": key, "run_id": self.run_id},
            )
        return self.providers[key]

    def seed_for(self, material: str) -> int:
        seed_source = f"{self.seed_lineage}|{self.run_id}|{material}"
        digest = hashlib.sha256(seed_source.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % (2**31 - 1)


@dataclass
class CleaningStepResult:
    frame: pl.DataFrame
    state: CleaningPipelineState
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    provider_lineage: dict[str, Any] = field(default_factory=dict)
    validation_failures: list[str] = field(default_factory=list)
    fallback_events: list[dict[str, Any]] = field(default_factory=list)
    mutation: CleaningMutationSummary = field(default_factory=CleaningMutationSummary)

    def apply_to_state(self) -> None:
        self.state.warnings.extend(self.warnings)
        self.state.fallback_events.extend(self.fallback_events)
        self.state.validation_failures.extend(
            {"error": error} for error in self.validation_failures
        )
        for key, lineage in self.provider_lineage.items():
            self.state.provider_lineage[key] = dict(lineage)


@dataclass(frozen=True)
class BuiltCleaningPipeline:
    spec: CleaningPipelineSpec
    steps: tuple[Any, ...]
    plan_hash: str
    registry_state_hash: str

    def run(self, df: Any, *, context: CleaningRuntimeContext | None = None) -> CleaningStepResult:
        from pysrc.pipeline.stages.cleaning.execution.runtime import CleaningPipelineRunner

        runner = CleaningPipelineRunner(self)
        return runner.run(df, context=context)

    def to_plan_payload(self) -> dict[str, Any]:
        from pysrc.pipeline.stages.cleaning.core.registry import resolve_cleaning_step

        provider_contracts = {
            step.step_id: list(
                resolve_cleaning_step(step.step_type, step.version).provider_requirements
            )
            for step in self.spec.steps
        }
        return {
            "schema_version": "1.0.0",
            "plan_hash": self.plan_hash,
            "registry_state_hash": self.registry_state_hash,
            "provider_contracts": provider_contracts,
            **self.spec.to_payload(),
        }

    def to_report_payload(
        self,
        result: CleaningStepResult,
        *,
        context: CleaningRuntimeContext,
    ) -> dict[str, Any]:
        step_reports = list(result.metrics.get("step_reports", []))
        pit_identity = sorted(
            {
                str(lineage["pit_identity"])
                for lineage in result.provider_lineage.values()
                if isinstance(lineage, Mapping) and "pit_identity" in lineage
            }
        )
        return {
            "schema_version": "1.0.0",
            "run_id": context.run_id,
            "plan_hash": self.plan_hash,
            "registry_state_hash": self.registry_state_hash,
            "determinism_tier": context.determinism_tier.value,
            "seed_lineage": context.seed_lineage,
            "pit_boundary": context.pit_boundary,
            "pit_identity": pit_identity,
            "governance_mode": context.governance_mode.value,
            "step_order": [report["step_id"] for report in step_reports],
            "step_versions": {report["step_id"]: report["version"] for report in step_reports},
            "steps": step_reports,
            "warnings": list(result.warnings),
            "validation_failures": list(result.state.validation_failures),
            "fallback_policy": {
                step.step_id: _normalize_jsonable(step.fallback_policy) for step in self.spec.steps
            },
            "fallback_events": list(result.fallback_events),
            "provider_lineage": _normalize_jsonable(result.provider_lineage),
            "final_contract_status": _normalize_jsonable(
                result.metrics.get("final_contract_status", {})
            ),
            "mutation_summary": result.mutation.to_payload(),
        }


def _normalize_jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _normalize_jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_jsonable(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_jsonable(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "model_dump"):
        return _normalize_jsonable(value.model_dump(mode="json"))
    if isinstance(value, pl.DataType):
        return str(value)
    return value

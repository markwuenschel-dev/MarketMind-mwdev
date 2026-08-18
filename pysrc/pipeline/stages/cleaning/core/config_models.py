from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from pysrc.core.errors import DataValidationError
from pysrc.pipeline.core import PipelineContext
from pysrc.pipeline.stages.cleaning.core.contracts import (
    CleaningDeterminismTier,
    CleaningPipelineSpec,
    CleaningStepSpec,
    FrameContract,
    GovernanceMode,
)


class ExternalFrameContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    required_columns: tuple[str, ...] = ()
    optional_columns: tuple[str, ...] = ()
    schema_payload: Mapping[str, Any] | None = Field(default=None, alias="schema")
    strict: bool = False
    unknown_ok: bool = True

    def to_internal(self) -> FrameContract:
        payload = self.model_dump(mode="python", by_alias=True)
        return FrameContract.from_mapping(payload)


class ExternalCleaningStepSpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    step_id: str
    step_type: str
    version: str
    enabled: bool = True
    params: Mapping[str, Any] = Field(default_factory=dict)
    input_contract: ExternalFrameContractModel = Field(default_factory=ExternalFrameContractModel)
    output_contract: ExternalFrameContractModel = Field(default_factory=ExternalFrameContractModel)
    determinism_tier: CleaningDeterminismTier | None = None
    governance_mode: GovernanceMode | None = None
    fallback_policy: Mapping[str, Any] = Field(default_factory=dict)

    def to_internal(
        self,
        *,
        default_governance_mode: GovernanceMode,
        default_determinism_tier: CleaningDeterminismTier,
    ) -> CleaningStepSpec:
        return CleaningStepSpec(
            step_id=self.step_id,
            step_type=self.step_type,
            version=self.version,
            enabled=self.enabled,
            params=dict(self.params),
            input_contract=self.input_contract.to_internal(),
            output_contract=self.output_contract.to_internal(),
            determinism_tier=self.determinism_tier or default_determinism_tier,
            governance_mode=self.governance_mode or default_governance_mode,
            fallback_policy=dict(self.fallback_policy),
        )


class ExternalCleaningComboModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str | None = None
    when: Mapping[str, Any] = Field(default_factory=dict)
    steps: tuple[ExternalCleaningStepSpecModel, ...] = ()
    order: Mapping[str, Mapping[str, tuple[str, ...]]] = Field(default_factory=dict)


class ExternalCleaningPipelineSpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    steps: tuple[ExternalCleaningStepSpecModel, ...] = ()
    determinism_tier: CleaningDeterminismTier = CleaningDeterminismTier.D1
    governance_mode: GovernanceMode = GovernanceMode.GOVERNED
    seed_lineage: str = ""
    pit_boundary: str = ""
    metadata: Mapping[str, Any] = Field(default_factory=dict)

    def to_internal(self) -> CleaningPipelineSpec:
        return CleaningPipelineSpec(
            steps=tuple(
                step.to_internal(
                    default_governance_mode=self.governance_mode,
                    default_determinism_tier=self.determinism_tier,
                )
                for step in self.steps
            ),
            determinism_tier=self.determinism_tier,
            seed_lineage=self.seed_lineage,
            pit_boundary=self.pit_boundary,
            governance_mode=self.governance_mode,
            metadata=dict(self.metadata),
        )


class ExternalCleaningConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    use: str | None = None
    determinism_tier: CleaningDeterminismTier = CleaningDeterminismTier.D1
    governance_mode: GovernanceMode = GovernanceMode.GOVERNED
    seed_lineage: str = ""
    pit_boundary: str = ""
    metadata: Mapping[str, Any] = Field(default_factory=dict)
    combos: Mapping[str, ExternalCleaningComboModel] | tuple[ExternalCleaningComboModel, ...] = (
        Field(default_factory=tuple)
    )


def pipeline_spec_from_external_cleaning_config(
    raw: Mapping[str, Any],
    *,
    context: PipelineContext | None = None,
    metadata: Mapping[str, Any] | None = None,
    name: str | None = None,
) -> CleaningPipelineSpec:
    config = ExternalCleaningConfigModel.model_validate(raw)
    combo = _select_combo(config, context=context, name=name)
    if combo is None:
        return CleaningPipelineSpec(
            steps=(),
            determinism_tier=config.determinism_tier,
            seed_lineage=config.seed_lineage,
            pit_boundary=config.pit_boundary,
            governance_mode=config.governance_mode,
            metadata={**dict(config.metadata), **dict(metadata or {})},
        )

    ordered_step_ids = _topo_order(
        [step.step_id for step in combo.steps],
        combo.order,
    )
    by_id = {step.step_id: step for step in combo.steps}
    missing = [step_id for step_id in ordered_step_ids if step_id not in by_id]
    if missing:
        raise DataValidationError(
            "Cleaning combo order references unknown step ids",
            details={"missing": missing},
        )
    steps = tuple(
        by_id[step_id].to_internal(
            default_governance_mode=config.governance_mode,
            default_determinism_tier=config.determinism_tier,
        )
        for step_id in ordered_step_ids
    )
    merged_metadata = {**dict(config.metadata), **dict(metadata or {})}
    if combo.name:
        merged_metadata.setdefault("combo", combo.name)
    return CleaningPipelineSpec(
        steps=steps,
        determinism_tier=config.determinism_tier,
        seed_lineage=config.seed_lineage,
        pit_boundary=config.pit_boundary,
        governance_mode=config.governance_mode,
        metadata=merged_metadata,
    )


def pipeline_spec_from_external_pipeline_spec(raw: Mapping[str, Any]) -> CleaningPipelineSpec:
    return ExternalCleaningPipelineSpecModel.model_validate(raw).to_internal()


def _select_combo(
    config: ExternalCleaningConfigModel,
    *,
    context: PipelineContext | None,
    name: str | None,
) -> ExternalCleaningComboModel | None:
    combos = _iter_combos(config.combos)
    if not combos:
        return None
    if name is not None:
        for combo in combos:
            if combo.name == name:
                return combo
        raise DataValidationError(
            "Unknown cleaning combo",
            details={"requested": name, "available": _combo_names(combos)},
        )
    if config.use is not None:
        for combo in combos:
            if combo.name == config.use:
                return combo
        raise DataValidationError(
            "Unknown cleaning combo",
            details={"requested": config.use, "available": _combo_names(combos)},
        )
    if context is not None:
        for combo in combos:
            if combo.when and _matches(combo.when, context):
                return combo
    if len(combos) == 1:
        return combos[0]
    default = next((combo for combo in combos if combo.name == "default"), None)
    if default is not None:
        return default
    raise DataValidationError(
        "Cleaning config requires an explicit combo selector",
        details={"available": _combo_names(combos)},
    )


def _iter_combos(
    raw_combos: Mapping[str, ExternalCleaningComboModel] | tuple[ExternalCleaningComboModel, ...],
) -> list[ExternalCleaningComboModel]:
    if isinstance(raw_combos, Mapping):
        out: list[ExternalCleaningComboModel] = []
        for combo_name, combo in raw_combos.items():
            if combo.name == combo_name:
                out.append(combo)
            else:
                out.append(combo.model_copy(update={"name": combo_name}))
        return out
    return list(raw_combos)


def _combo_names(combos: list[ExternalCleaningComboModel]) -> list[str | None]:
    return [combo.name for combo in combos]


def _matches(rule_when: Mapping[str, Any], ctx: PipelineContext) -> bool:
    for key, expected in rule_when.items():
        actual = getattr(ctx, key, None)
        if isinstance(expected, (list, tuple, set)):
            if actual not in expected:
                return False
            continue
        if isinstance(expected, Mapping):
            if "in" in expected and actual not in set(expected["in"]):
                return False
            if "not_in" in expected and actual in set(expected["not_in"]):
                return False
            if "ge" in expected and not (actual is not None and actual >= expected["ge"]):
                return False
            if "gt" in expected and not (actual is not None and actual > expected["gt"]):
                return False
            if "le" in expected and not (actual is not None and actual <= expected["le"]):
                return False
            if "lt" in expected and not (actual is not None and actual < expected["lt"]):
                return False
            continue
        if actual != expected:
            return False
    return True


def _topo_order(
    step_ids: list[str],
    order_cfg: Mapping[str, Mapping[str, tuple[str, ...]]] | None,
) -> list[str]:
    seen: set[str] = set()
    ordered_step_ids = [
        step_id for step_id in step_ids if not (step_id in seen or seen.add(step_id))
    ]
    edges = _edges_from_order(ordered_step_ids, order_cfg)
    _check_conflicts(edges)

    predecessors: dict[str, set[str]] = {step_id: set() for step_id in ordered_step_ids}
    successors: dict[str, set[str]] = {step_id: set() for step_id in ordered_step_ids}
    for before, after in edges:
        if before not in predecessors or after not in predecessors:
            continue
        predecessors[after].add(before)
        successors[before].add(after)

    ready = [step_id for step_id in ordered_step_ids if not predecessors[step_id]]
    out: list[str] = []
    while ready:
        current = ready.pop(0)
        out.append(current)
        for successor in list(successors[current]):
            predecessors[successor].discard(current)
            if not predecessors[successor]:
                ready.append(successor)
    if len(out) != len(ordered_step_ids):
        raise DataValidationError("Cyclic cleaning step constraints detected")
    return out


def _edges_from_order(
    step_ids: list[str],
    order_cfg: Mapping[str, Mapping[str, tuple[str, ...]]] | None,
) -> list[tuple[str, str]]:
    if order_cfg is None:
        return []
    step_set = set(step_ids)
    edges: list[tuple[str, str]] = []
    for before, after_steps in (order_cfg.get("before") or {}).items():
        for after in after_steps or ():
            if before in step_set and after in step_set:
                edges.append((before, after))
    for after, before_steps in (order_cfg.get("after") or {}).items():
        for before in before_steps or ():
            if before in step_set and after in step_set:
                edges.append((before, after))
    return edges


def _check_conflicts(edges: list[tuple[str, str]]) -> None:
    edge_set = set(edges)
    for before, after in edges:
        if (after, before) in edge_set:
            raise DataValidationError(
                "Conflicting cleaning step constraints detected",
                details={"before": before, "after": after},
            )

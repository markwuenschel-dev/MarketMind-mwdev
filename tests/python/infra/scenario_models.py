# tests/python/infra/scenario_models.py
from __future__ import annotations

# This module defines the contract for all scenario objects that end up in pytest
# parametrization. We validate every scenario dict (learned or static) against
# these models at collection time so bad shapes never make it into the matrix.
from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

_PYDANTIC_V2 = hasattr(BaseModel, "model_validate")
_KIND_ALIASES = {
    "ensemble": "ensemble_loop",  # legacy alias → canonical prefix form
    # add more if needed, e.g.:
    # "strategy": "strategy_basic",
    # "module": "module_switches",
}


def _validate(cls, data):
    # v2 uses model_validate; v1 uses parse_obj (kept for safety)
    if _PYDANTIC_V2:
        return cls.model_validate(data)
    return cls.parse_obj(data)


def _reraise_validation_error(e: ValidationError, cls, data):
    # Never pass model=... in v2; reconstruct with v2 fields if needed,
    # otherwise just re-raise the original error.
    try:
        # v2 signature: errors list + optional title/input
        raise ValidationError(e.errors(), title=getattr(cls, "__name__", "Model"), input=data)
    except TypeError:
        # v1 signature: (errors, model_class)
        raise ValidationError(e.errors(), cls)


# Strict base model for scenario families that must not silently drift.
# extra="forbid" means: if engine/static authors start adding random fields
# without updating this schema, we blow up loudly in test collection instead
# of producing silently-misaligned tests.
class _StrictScenario(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ParallelThresholdScenario(_StrictScenario):
    kind: Literal["parallel_threshold_minus", "parallel_threshold_plus"]
    rows: int
    expect_parallel: bool
    requires: dict[str, Any] = {}


class HighRiskPatternScenario(_StrictScenario):
    kind: Literal["high_risk_pattern"]
    shape: tuple[int, ...]
    num_ops: int
    risk_count: int
    expect_stability: bool
    requires: dict[str, Any] = {}


class OptimalSequenceScenario(_StrictScenario):
    kind: Literal["optimal_sequence"]
    ops: list[str]
    expect_reorder: bool
    requires: dict[str, Any] = {}


# Ensemble-style / ensemble_* scenarios can evolve quickly and may carry
# arbitrary hints like "pattern". We allow forward-compatible extra keys here.
class EnsembleScenario(BaseModel):
    kind: str
    pattern: Any = None  # loose shape, engine- / test-writer-defined
    requires: dict[str, Any] = {}

    @field_validator("kind")
    @classmethod
    def _must_be_ensemble(cls, v: str) -> str:
        if not (isinstance(v, str) and v.startswith("ensemble_")):
            raise ValueError("EnsembleScenario.kind must start with 'ensemble_'")
        return v

    model_config = ConfigDict(extra="allow")  # forward-compatible for new ensemble_* data


# Union of all allowed scenario model types. We keep this simple (plain Union)
# so IDEs/mypy/etc. see a normal type, instead of Annotated[...] with Field().
AnyScenario = Union[
    ParallelThresholdScenario,
    HighRiskPatternScenario,
    OptimalSequenceScenario,
    EnsembleScenario,
]


# Map explicit kinds -> model class. This enforces that known kinds are strict.
# ensemble_* is handled separately via prefix match.
_KIND_TO_MODEL: dict[str, type[BaseModel]] = {
    "parallel_threshold_minus": ParallelThresholdScenario,
    "parallel_threshold_plus": ParallelThresholdScenario,
    "high_risk_pattern": HighRiskPatternScenario,
    "optimal_sequence": OptimalSequenceScenario,
}


def coerce_scenario(raw: dict[str, Any]) -> AnyScenario:
    kind = raw.get("kind")

    # Normalize legacy aliases to canonical kinds first
    if isinstance(kind, str) and kind in _KIND_ALIASES:
        kind = _KIND_ALIASES[kind]
        raw = {**raw, "kind": kind}

    # Forward-compatible: allow extra keys for ensemble_* kinds
    if isinstance(kind, str) and kind.startswith("ensemble_"):
        return EnsembleScenario.model_validate(raw)

    # Strict kinds must map to a concrete model
    model_cls = _KIND_TO_MODEL.get(kind)
    if model_cls is None:
        # Pydantic v2 line error (ctx['error'] required for value_error)
        line_errors = [
            {
                "type": "value_error",
                "loc": ("kind",),
                "msg": "Unknown scenario kind",
                "input": kind,
                "ctx": {"error": f"Unknown scenario kind '{kind}'"},
            }
        ]
        # v2: do not pass input= (your build rejects it)
        raise ValidationError.from_exception_data(
            title=_StrictScenario.__name__,
            line_errors=line_errors,
        )

    # Validate against the concrete strict model (v2)
    return model_cls.model_validate(raw)


class HandlerResult(BaseModel):
    raised: bool
    exc_type: str | None = None
    exc_msg: str = ""
    logs: list[str] = []

    model_config = ConfigDict(extra="forbid")  # handlers aren't allowed to invent random keys

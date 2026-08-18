from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from pysrc.core.errors import DataValidationError
from pysrc.pipeline.stages.cleaning.core.contracts import (
    CleaningDeterminismTier,
    FrameContract,
)


@dataclass(frozen=True)
class CleaningStepRegistration:
    step_type: str
    version: str
    step_cls: type[Any]
    params_model: type[BaseModel]
    input_contract: FrameContract
    output_contract: FrameContract
    determinism_tier: CleaningDeterminismTier
    provider_requirements: tuple[str, ...] = field(default_factory=tuple)
    stateful: bool = False

    def descriptor(self) -> dict[str, Any]:
        return {
            "step_type": self.step_type,
            "version": self.version,
            "step_class": f"{self.step_cls.__module__}.{self.step_cls.__name__}",
            "params_model": f"{self.params_model.__module__}.{self.params_model.__name__}",
            "input_contract": self.input_contract.to_payload(),
            "output_contract": self.output_contract.to_payload(),
            "determinism_tier": self.determinism_tier.value,
            "provider_requirements": list(self.provider_requirements),
            "stateful": self.stateful,
        }


_REGISTRY: dict[tuple[str, str], CleaningStepRegistration] = {}
_BOOTSTRAPPED = False

_DEFAULT_MODULES: tuple[str, ...] = (
    "pysrc.pipeline.stages.cleaning.imputers.missing",
    "pysrc.pipeline.stages.cleaning.imputers.outliers",
    "pysrc.pipeline.stages.cleaning.imputers.denoise",
    "pysrc.pipeline.stages.cleaning.validators.schema",
    "pysrc.pipeline.stages.cleaning.validators.io",
    "pysrc.pipeline.stages.cleaning.validators.stream",
    "pysrc.pipeline.stages.cleaning.validators.drift",
    "pysrc.pipeline.stages.cleaning.features.technical",
    "pysrc.pipeline.stages.cleaning.features.calendar",
    "pysrc.pipeline.stages.cleaning.features.macro",
    "pysrc.pipeline.stages.cleaning.features.altdata",
    "pysrc.pipeline.stages.cleaning.features.sentiment",
    "pysrc.pipeline.stages.cleaning.anomalies.batch",
    "pysrc.pipeline.stages.cleaning.anomalies.streaming",
)


def register_cleaning_step(
    *,
    step_type: str,
    version: str,
    params_model: type[BaseModel],
    input_contract: FrameContract | None = None,
    output_contract: FrameContract | None = None,
    determinism_tier: CleaningDeterminismTier = CleaningDeterminismTier.D1,
    provider_requirements: tuple[str, ...] = (),
    stateful: bool = False,
) -> Callable[[type[Any]], type[Any]]:
    def _decorator(step_cls: type[Any]) -> type[Any]:
        key = (step_type, version)
        if key in _REGISTRY:
            raise DataValidationError(
                "Cleaning step registration already exists",
                details={"step_type": step_type, "version": version},
            )
        registration = CleaningStepRegistration(
            step_type=step_type,
            version=version,
            step_cls=step_cls,
            params_model=params_model,
            input_contract=input_contract or FrameContract(),
            output_contract=output_contract or FrameContract(),
            determinism_tier=determinism_tier,
            provider_requirements=provider_requirements,
            stateful=stateful,
        )
        _REGISTRY[key] = registration
        step_cls.STEP_TYPE = step_type
        step_cls.STEP_VERSION = version
        return step_cls

    return _decorator


def bootstrap_default_cleaning_registry() -> None:
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    for module_name in _DEFAULT_MODULES:
        importlib.import_module(module_name)
    _BOOTSTRAPPED = True


def resolve_cleaning_step(step_type: str, version: str) -> CleaningStepRegistration:
    bootstrap_default_cleaning_registry()
    key = (step_type, version)
    if key not in _REGISTRY:
        available = sorted(
            f"{registered.step_type}@{registered.version}" for registered in _REGISTRY.values()
        )
        raise DataValidationError(
            "Unknown cleaning step registration",
            details={
                "requested_step_type": step_type,
                "requested_version": version,
                "available": available,
            },
        )
    return _REGISTRY[key]


def list_registered_cleaning_steps() -> list[CleaningStepRegistration]:
    bootstrap_default_cleaning_registry()
    return sorted(
        _REGISTRY.values(), key=lambda registration: (registration.step_type, registration.version)
    )


def registry_state_hash() -> str:
    bootstrap_default_cleaning_registry()
    payload = [registration.descriptor() for registration in list_registered_cleaning_steps()]
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

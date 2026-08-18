from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "DriftDetectionStep": "pysrc.pipeline.stages.cleaning.validators.drift",
    "IOValidationStep": "pysrc.pipeline.stages.cleaning.validators.io",
    "StreamValidationStep": "pysrc.pipeline.stages.cleaning.validators.stream",
    "ValidationStep": "pysrc.pipeline.stages.cleaning.validators.schema",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return list(__all__)

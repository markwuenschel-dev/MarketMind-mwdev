from __future__ import annotations

from functools import cache
from importlib import import_module
from types import ModuleType
from typing import Any, Final

__all__ = ["optional_import", "require", "require_attr"]

_PIP_NAMES: Final[dict[str, str]] = {
    "aiohttp": "aiohttp",
    "blake3": "blake3",
    "cudf": "cudf-cu13",
    "cupy": "cupy-cuda13x",
    "cuml": "cuml-cu13",
    "dask": "dask",
    "dask.dataframe": "dask[dataframe]",
    "grpc": "grpcio",
    "google.cloud.logging": "google-cloud-logging",
    "jsonschema": "jsonschema",
    "influxdb_client": "influxdb-client",
    "mlflow": "mlflow",
    "numpy": "numpy",
    "opentelemetry.sdk.metrics.aggregation": "opentelemetry-sdk",
    "opentelemetry.sdk.metrics.exemplar": "opentelemetry-sdk",
    "pandas": "pandas",
    "polars": "polars",
    "prometheus_client": "prometheus-client",
    "psutil": "psutil",
    "pydantic": "pydantic",
    "pynvml": "pynvml",
    "redis": "redis",
    "sklearn": "scikit-learn",
    "sklearn.ensemble": "scikit-learn",
    "sklearn.model_selection": "scikit-learn",
    "sklearn.preprocessing": "scikit-learn",
    "torch": "torch",
    "watchtower": "watchtower",
    "xgboost": "xgboost",
    "xxhash": "xxhash",
    "yaml": "PyYAML",
    "zstandard": "zstandard",
}


@cache
def _module_prefixes(module: str) -> tuple[str, ...]:
    parts = module.split(".")
    return tuple(".".join(parts[:i]) for i in range(1, len(parts) + 1))


def _validate_module_name(module: str) -> str:
    module = module.strip()
    if not module:
        raise ValueError("module must be a non-empty import path")
    return module


def _validate_attr_name(attr: str) -> str:
    attr = attr.strip()
    if not attr:
        raise ValueError("attr must be a non-empty attribute name")
    return attr


def _is_missing_requested_module(exc: ModuleNotFoundError, module: str) -> bool:
    missing = exc.name or ""
    return missing in _module_prefixes(module)


def _resolve_pip_name(module: str) -> str:
    for prefix in reversed(_module_prefixes(module)):
        pip_name = _PIP_NAMES.get(prefix)
        if pip_name is not None:
            return pip_name
    return module


def _import_optional(module: str) -> ModuleType | None:
    try:
        return import_module(module)
    except ModuleNotFoundError as exc:
        if _is_missing_requested_module(exc, module):
            return None
        raise


def optional_import(module: str) -> ModuleType | None:
    module = _validate_module_name(module)
    return _import_optional(module)


def require(module: str, *, purpose: str | None = None) -> ModuleType:
    module = _validate_module_name(module)
    imported = _import_optional(module)
    if imported is not None:
        return imported
    pkg = _resolve_pip_name(module)
    msg = f"Missing dependency '{pkg}'"
    if purpose:
        msg += f" required for {purpose.strip()}"
    msg += f". Install with: pip install {pkg}"
    raise ImportError(msg)


def require_attr(module: str, attr: str, *, purpose: str | None = None) -> Any:
    attr = _validate_attr_name(attr)
    imported = require(module, purpose=purpose)
    try:
        return getattr(imported, attr)
    except AttributeError as exc:
        raise ImportError(f"Dependency '{module}' does not provide '{attr}'") from exc

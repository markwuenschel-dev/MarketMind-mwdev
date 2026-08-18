"""Model family registry and factory for panel experiments.

Rule: ``registered == executable``. Families listed in ``PLANNED_MODEL_FAMILIES``
are documented for future work but fail at config resolution, not at train time.
"""

from __future__ import annotations

from typing import Any

from pysrc.models.base import PanelModel
from pysrc.models.mlp import create_mlp_model
from pysrc.models.tabular import TABULAR_FAMILIES, TabularPanelModel


def _xgboost_available() -> bool:
    try:
        import xgboost  # noqa: F401

        return True
    except ImportError:
        return False


_TABULAR_EXECUTABLE: frozenset[str] = frozenset(
    family for family in TABULAR_FAMILIES if family != "xgboost" or _xgboost_available()
)

EXECUTABLE_MODEL_FAMILIES: frozenset[str] = _TABULAR_EXECUTABLE | frozenset({"mlp"})

PLANNED_MODEL_FAMILIES: frozenset[str] = frozenset(
    {"lstm", "gru", "tcn", "transformer", "informer", "bayesian_nn", "gnn", "hybrid"}
)

# Backward-compatible alias used by CLI and configs.
SUPPORTED_MODEL_FAMILIES = EXECUTABLE_MODEL_FAMILIES


def resolve_model_family(name: str) -> str:
    """Validate a model family; raise before run allocation if not executable."""

    if name in EXECUTABLE_MODEL_FAMILIES:
        return name
    if name in PLANNED_MODEL_FAMILIES:
        raise ValueError(
            f"Model family '{name}' is planned but not yet executable via panel train-matrix. "
            f"Executable families: {sorted(EXECUTABLE_MODEL_FAMILIES)}"
        )
    if name == "xgboost" and not _xgboost_available():
        raise ValueError(
            "Model family 'xgboost' requires the xgboost package. "
            f"Executable families: {sorted(EXECUTABLE_MODEL_FAMILIES)}"
        )
    if name in {"group_lasso", "sparse_group_lasso"}:
        raise ValueError(
            f"Model family '{name}' has no panel contract yet; not registered. "
            f"Executable families: {sorted(EXECUTABLE_MODEL_FAMILIES)}"
        )
    raise ValueError(f"Unsupported model_family: {name}")


def create_panel_model(
    family: str,
    *,
    model_id: str | None = None,
    params: dict[str, Any] | None = None,
    random_seed: int = 42,
    sklearn_n_jobs: int = 1,
) -> PanelModel:
    """Instantiate a panel model for the given family."""

    resolved = resolve_model_family(family)
    mid = model_id or resolved
    if resolved in TABULAR_FAMILIES:
        return TabularPanelModel(
            model_id=mid,
            family=resolved,
            params=params,
            random_seed=random_seed,
            sklearn_n_jobs=sklearn_n_jobs,
        )
    if resolved == "mlp":
        return create_mlp_model(model_id=mid, params=params, random_seed=random_seed)
    raise ValueError(f"Unsupported model_family: {family}")


def model_entries_from_yaml(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize YAML model list entries."""

    entries: list[dict[str, Any]] = []
    for item in models:
        if not isinstance(item, dict):
            continue
        family = item.get("family")
        if not isinstance(family, str):
            continue
        resolve_model_family(family)
        entries.append(
            {
                "family": family,
                "params": dict(item.get("params") or {}),
                "sequence_length": item.get("sequence_length"),
            }
        )
    return entries


__all__ = [
    "EXECUTABLE_MODEL_FAMILIES",
    "PLANNED_MODEL_FAMILIES",
    "SUPPORTED_MODEL_FAMILIES",
    "TABULAR_FAMILIES",
    "create_panel_model",
    "model_entries_from_yaml",
    "resolve_model_family",
]

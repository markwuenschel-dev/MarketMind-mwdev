"""Model implementations and registry for panel experiments.

All model definitions, runtime helpers, and dataset contracts live under
``pysrc.models``. Training orchestration lives in ``pysrc.pipeline.panel``.
"""

from __future__ import annotations

from pysrc.models.registry import (
    EXECUTABLE_MODEL_FAMILIES,
    PLANNED_MODEL_FAMILIES,
    SUPPORTED_MODEL_FAMILIES,
    resolve_model_family,
)
from pysrc.models.tabular import get_model_instance, train_and_predict

__all__ = [
    "EXECUTABLE_MODEL_FAMILIES",
    "PLANNED_MODEL_FAMILIES",
    "SUPPORTED_MODEL_FAMILIES",
    "get_model_instance",
    "resolve_model_family",
    "train_and_predict",
]

"""Version-aware extension registries; every extension point is explicit."""

from pysrc.tuning.core.registries import (
    feature_adapter_registry,
    gate_registry,
    model_registry,
    objective_registry,
    planner_registry,
    search_registry,
    validation_registry,
)
from pysrc.tuning.core.registries.feature_adapter_registry import FeatureAdapterProtocol
from pysrc.tuning.core.registries.model_registry import ModelFactory

__all__ = [
    "FeatureAdapterProtocol",
    "ModelFactory",
    "feature_adapter_registry",
    "gate_registry",
    "model_registry",
    "objective_registry",
    "planner_registry",
    "search_registry",
    "validation_registry",
]

# pysrc/pipeline/stages/preprocessing/__init__.py
from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from typing import Any

from pysrc.ops.mm_logkit import get_logger

LOG = get_logger(__name__)

StepClass = type[Any]

_STEP_IMPORTS: Mapping[str, tuple[str, str]] = {
    "technical": ("pysrc.pipeline.stages.preprocessing.technical_step", "TechnicalFeaturesStep"),
    "indicator_engine": (
        "pysrc.pipeline.stages.preprocessing.indicator_engine_step",
        "IndicatorEngineStep",
    ),
    "scaling": ("pysrc.pipeline.stages.preprocessing.scaling_step", "ScalingStep"),
    "sentiment": ("pysrc.pipeline.stages.preprocessing.sentiment_step", "SentimentESGStep"),
    "temporal": ("pysrc.pipeline.stages.preprocessing.temporal_step", "TemporalStep"),
    "sequence": ("pysrc.pipeline.stages.preprocessing.sequence_step", "SequenceStep"),
    "explainability": (
        "pysrc.pipeline.stages.preprocessing.explainability_step",
        "ExplainabilityStep",
    ),
    "embedding": (
        "pysrc.pipeline.stages.preprocessing.text_embedding_step",
        "TextEmbeddingStep",
    ),
    "text_embedding": (
        "pysrc.pipeline.stages.preprocessing.text_embedding_step",
        "TextEmbeddingStep",
    ),
    "topic_modeling": (
        "pysrc.pipeline.stages.preprocessing.topic_modeling_step",
        "TopicModelingStep",
    ),
    "topics": ("pysrc.pipeline.stages.preprocessing.topic_modeling_step", "TopicModelingStep"),
}

_CLASS_IMPORTS: Mapping[str, tuple[str, str]] = {
    class_name: (module_name, class_name) for module_name, class_name in set(_STEP_IMPORTS.values())
}


def _load_plugins() -> dict[str, StepClass]:
    try:
        from importlib.metadata import entry_points

        eps = entry_points(group="marketmind.preproc_steps")
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        LOG.warning("preprocessing_entry_points_unavailable", error=str(exc))
        return {}

    out: dict[str, StepClass] = {}
    for ep in eps:
        try:
            cls_obj = ep.load()
        except (ImportError, AttributeError, TypeError, ValueError) as exc:
            LOG.warning("preprocessing_plugin_load_failed", plugin=ep.name, error=str(exc))
            continue
        if not isinstance(cls_obj, type):
            LOG.warning("preprocessing_plugin_not_class", plugin=ep.name)
            continue
        out[ep.name.lower()] = cls_obj
    return out


def _load_builtin(name: str) -> StepClass:
    module_name, class_name = _STEP_IMPORTS[name]
    module = import_module(module_name)
    cls_obj = getattr(module, class_name)
    if not isinstance(cls_obj, type):
        raise TypeError(f"Preprocessing step {name!r} did not resolve to a class")
    return cls_obj


class StepFactory:
    _registry: dict[str, StepClass] = _load_plugins()

    @classmethod
    def register(cls, name: str, step_cls: StepClass) -> None:
        cls._registry[name.lower()] = step_cls

    @classmethod
    def get(cls, name: str) -> StepClass:
        key = name.lower().strip()
        if key in cls._registry:
            return cls._registry[key]
        if key in _STEP_IMPORTS:
            step_cls = _load_builtin(key)
            cls._registry[key] = step_cls
            return step_cls
        available = sorted(set(_STEP_IMPORTS) | set(cls._registry))
        available_text = ", ".join(available)
        raise ValueError(f"Unknown preprocessing step {name!r}. Available: {available_text}")

    @classmethod
    def create(cls, name: str, cfg: dict[str, Any]) -> Any:
        return cls.get(name)(**cfg)


def __getattr__(name: str) -> Any:
    if name in _CLASS_IMPORTS:
        module_name, class_name = _CLASS_IMPORTS[name]
        module = import_module(module_name)
        value = getattr(module, class_name)
        globals()[name] = value
        return value
    raise AttributeError(name)


new_step = StepFactory.create
AVAILABLE_STEPS = tuple(sorted(set(_STEP_IMPORTS) | set(StepFactory._registry)))

__all__ = [
    "TechnicalFeaturesStep",
    "IndicatorEngineStep",
    "ScalingStep",
    "SentimentESGStep",
    "TemporalStep",
    "SequenceStep",
    "ExplainabilityStep",
    "TextEmbeddingStep",
    "TopicModelingStep",
    "StepFactory",
    "new_step",
    "AVAILABLE_STEPS",
]

# py/pipeline/core/pipeline_core_registry.py
from __future__ import annotations

import importlib.metadata
from typing import TYPE_CHECKING

from pysrc.ops.mm_logkit import get_logger

if TYPE_CHECKING:
    from pysrc.pipeline.core.pipeline_core_base import PipelineStep

logger = get_logger(__name__)


class StepRegistry:
    _registry: dict[str, dict[str, type[PipelineStep]]] = {
        "cleaning": {},
        "preprocessing": {},
        "market_data": {},  # anticipate future stages
        "features": {},
    }

    @classmethod
    def register(cls, *args, override: bool = False) -> None:
        if len(args) == 2:
            stage, name, step_class = "cleaning", args[0], args[1]
        elif len(args) == 3:
            stage, name, step_class = args
        else:
            raise TypeError("register() expects (name, step_class) or (stage, name, step_class)")

        if stage not in cls._registry:
            cls._registry[stage] = {}
        stage_map = cls._registry[stage]

        if not override and name in stage_map:
            # architectural choice: log warning instead of raising in production
            logger.warning(f"Overriding existing step '{name}' in stage '{stage}'")
        stage_map[name] = step_class

    @classmethod
    def get(cls, stage: str, name: str) -> type[PipelineStep]:
        # fallback chain for flexibility
        if stage in cls._registry and name in cls._registry[stage]:
            return cls._registry[stage][name]
        # try without stage prefix if name looks like full path
        if "." in name:
            short_name = name.split(".")[-1]
            if stage in cls._registry and short_name in cls._registry[stage]:
                return cls._registry[stage][short_name]
        raise KeyError(f"Unknown step '{name}' for stage '{stage}'")

    @classmethod
    def load_plugins(cls, entry_point_group: str, stage: str) -> None:
        try:
            eps = importlib.metadata.entry_points().select(group=entry_point_group)
        except AttributeError:
            try:
                eps = importlib.metadata.entry_points(group=entry_point_group)
            except TypeError:
                eps = importlib.metadata.entry_points().get(entry_point_group, [])

        for ep in eps:
            try:
                step_class = ep.load()
                cls.register(stage, ep.name, step_class, override=True)
                logger.info(f"Loaded plugin: {ep.name}")
            except Exception as e:
                logger.warning(f"Failed loading {ep.name}: {e}")

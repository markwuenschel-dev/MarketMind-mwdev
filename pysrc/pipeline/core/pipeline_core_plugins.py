# py/pipeline/core/pipeline_core_plugins.py
"""Loads stage-specific plugins via importlib.metadata entry points."""

from __future__ import annotations

from collections.abc import Iterable

from .pipeline_core_registry import StepRegistry


def load_stage_plugins(stage: str, group_prefix: str = "marketmind"):
    """
    Load plugins for a specific pipeline stage.
    Entry point group convention: f"{group_prefix}.{stage}_steps"
    """
    entry_point_group = f"{group_prefix}.{stage}_steps"
    StepRegistry.load_plugins(entry_point_group, stage)


def discover_all_plugins(stages: Iterable[str], group_prefix: str = "marketmind"):
    """Discover and load plugins for all given stages."""
    for stage in stages:
        load_stage_plugins(stage, group_prefix=group_prefix)

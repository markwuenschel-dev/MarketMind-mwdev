"""run_search_plan: execute a SearchPlan and emit trial results."""

from __future__ import annotations

from typing import Any

from pysrc.tuning.core.planning.search_plan import SearchPlan


def run_search_plan(plan: SearchPlan, context: dict[str, Any]) -> list[dict[str, Any]]:
    """Execute all steps of a SearchPlan; return a list of trial result dicts."""
    raise NotImplementedError("run_search_plan must be wired to an executor and model registry")


__all__ = ["run_search_plan"]

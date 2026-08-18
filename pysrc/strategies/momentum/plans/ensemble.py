from __future__ import annotations

from typing import Any

from pysrc.strategies.pipeline_strategy import FeaturePlan


def build_plan(params: dict[str, Any]) -> FeaturePlan:
    raise NotImplementedError(
        "momentum.plans.ensemble is a Phase II/III stub. "
        "See ResolutionLedger and Spec v1.3 §2.4 for promotion criteria."
    )

from __future__ import annotations

from math import isclose

from pysrc.backtesting.validation.mechanical.parity.tolerances import tolerance_for_tier


def compare_metrics(metrics_a: dict[str, float], metrics_b: dict[str, float], tier) -> bool:
    tolerance = tolerance_for_tier(tier)
    keys = set(metrics_a) | set(metrics_b)
    return all(
        isclose(
            float(metrics_a.get(key, 0.0)),
            float(metrics_b.get(key, 0.0)),
            abs_tol=tolerance.atol,
            rel_tol=tolerance.rtol,
        )
        for key in keys
    )

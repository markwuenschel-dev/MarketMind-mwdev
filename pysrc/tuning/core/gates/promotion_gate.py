"""Promotion gate: composite check that a candidate is safe to promote."""

from __future__ import annotations

from pysrc.tuning.core.gates.stat_validity import passes_dsr_gate, passes_harvey_tstat


def evaluate_promotion_gate(
    dsr: float,
    t_stat: float,
    pit_ok: bool,
    determinism_ok: bool,
    dsr_threshold: float = 0.0,
    t_stat_min: float = 3.0,
) -> tuple[bool, dict[str, bool]]:
    """Return (overall_passed, per-gate breakdown dict)."""
    checks: dict[str, bool] = {
        "dsr": passes_dsr_gate(dsr, dsr_threshold),
        "harvey_tstat": passes_harvey_tstat(t_stat, t_stat_min),
        "pit": pit_ok,
        "determinism": determinism_ok,
    }
    return all(checks.values()), checks


__all__ = ["evaluate_promotion_gate"]

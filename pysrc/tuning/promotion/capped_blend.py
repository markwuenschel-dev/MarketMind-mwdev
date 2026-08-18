"""Capped blend: allocate up to blend_cap weight to a new candidate alongside the live strategy."""

from __future__ import annotations

from typing import Any


def run_capped_blend(
    candidate_id: str,
    job_id: str,
    blend_cap: float,
    context: dict[str, Any],
) -> dict[str, Any]:
    """Activate capped blending; return a blend handle dict."""
    if not (0.0 < blend_cap <= 1.0):
        raise ValueError(f"blend_cap must be in (0, 1], got {blend_cap}")
    raise NotImplementedError("run_capped_blend must be wired to the live signal router")


__all__ = ["run_capped_blend"]

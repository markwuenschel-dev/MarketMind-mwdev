"""Shadow mode: run a promoted candidate alongside the live strategy without affecting signals."""

from __future__ import annotations

from typing import Any


def run_shadow_mode(
    candidate_id: str,
    job_id: str,
    duration_days: int,
    context: dict[str, Any],
) -> dict[str, Any]:
    """Start shadow-mode execution; return a shadow run handle dict."""
    raise NotImplementedError(
        "run_shadow_mode must be wired to the live inference adapter and monitoring layer"
    )


__all__ = ["run_shadow_mode"]

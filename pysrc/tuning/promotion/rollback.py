"""Rollback: revert the live strategy to a prior artifact snapshot."""

from __future__ import annotations

from typing import Any


def run_rollback(
    job_id: str,
    target_artifact_hash: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    """Roll back to target_artifact_hash; return a rollback receipt dict."""
    raise NotImplementedError(
        "run_rollback must be wired to the artifact reader and live_checkpoint_switch"
    )


__all__ = ["run_rollback"]

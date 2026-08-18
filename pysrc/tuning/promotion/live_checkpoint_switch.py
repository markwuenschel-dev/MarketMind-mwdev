"""LiveCheckpointSwitch: atomically swap the active model checkpoint in the live system."""

from __future__ import annotations

from typing import Any


def switch_live_checkpoint(
    job_id: str,
    from_artifact_hash: str,
    to_artifact_hash: str,
    context: dict[str, Any],
) -> None:
    """Atomically switch the live checkpoint from one artifact hash to another."""
    raise NotImplementedError(
        "switch_live_checkpoint must be wired to the runtime state store and artifact reader"
    )


__all__ = ["switch_live_checkpoint"]

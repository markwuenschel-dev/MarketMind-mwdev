"""JobRunner: top-level coordinator that drives a tuning job from submission to completion."""

from __future__ import annotations

from typing import Any


class JobRunner:
    """Drives a TuningJobSpec through search, validation, gating, and promotion."""

    def run(self, job_id: str, config: dict[str, Any]) -> None:
        """Execute a full tuning job; raises on unrecoverable failure."""
        raise NotImplementedError("JobRunner.run must be wired to the execution and gate layers")

    def cancel(self, job_id: str) -> None:
        """Cancel a running job by job_id."""
        raise NotImplementedError("JobRunner.cancel requires a live state-machine connection")


__all__ = ["JobRunner"]

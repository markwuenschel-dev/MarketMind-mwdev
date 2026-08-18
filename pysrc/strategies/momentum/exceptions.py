from __future__ import annotations

from typing import Any

from pysrc.strategies.pipeline_strategy import MaterializationError, PipelineError


class FeatureFlagError(PipelineError):
    """Raised when a feature-flagged momentum variant is invoked without its gate."""


class ConvergenceError(MaterializationError):
    """Raised when an iterative momentum materialization step fails to converge."""

    def __init__(
        self,
        message: str,
        *,
        n_iterations: int | None = None,
        asset_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.n_iterations = n_iterations
        self.asset_id = asset_id


class CostGateRejection(Exception):
    """Raised when governed execution cost checks reject a momentum trade intent."""

    def __init__(
        self,
        message: str,
        *,
        variant: str,
        run_id: str,
        reason_code: str,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.variant = variant
        self.run_id = run_id
        self.reason_code = reason_code


class SerializationError(Exception):
    """Raised when a governed artifact cannot be serialized safely."""


class MissingExecutionAssumptionsError(FileNotFoundError):
    """Raised when governed entry execution reaches the cost gate without assumptions."""


def exception_metadata(exc: Exception) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for field_name in ("message", "n_iterations", "asset_id", "variant", "run_id", "reason_code"):
        if hasattr(exc, field_name):
            metadata[field_name] = getattr(exc, field_name)
    return metadata

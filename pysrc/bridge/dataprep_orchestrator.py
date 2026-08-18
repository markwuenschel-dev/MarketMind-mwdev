"""Dataprep orchestration bridge.

Coordinates data preparation pipelines between Python preprocessing
and Java entry points. Provides typed contracts for cross-runtime
data handoff with deterministic lineage tracking.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class DataprepSpec:
    """Dataprep execution specification.

    Attributes:
        pipeline_id: Canonical pipeline identifier
        config_path: Path to pipeline configuration
        output_dir: Destination for prepared artifacts
        as_of: Point-in-time boundary for data leakage prevention
    """

    pipeline_id: str
    config_path: Path
    output_dir: Path
    as_of: str | None = None

    def __post_init__(self) -> None:
        if not self.pipeline_id:
            raise ValueError("pipeline_id must be non-empty")


@dataclass(frozen=True, slots=True)
class DataprepResult:
    """Dataprep execution result.

    Attributes:
        success: Execution status
        artifact_paths: Mapping of artifact name to path
        lineage_hash: Deterministic hash of inputs for reproducibility
        metrics: Execution metrics (duration, memory, etc.)
    """

    success: bool
    artifact_paths: Mapping[str, Path]
    lineage_hash: str
    metrics: Mapping[str, float]


class DataprepOrchestrator(ABC):
    """Abstract dataprep orchestrator for Python/Java bridge.

    Implementations coordinate multi-stage data preparation
    with artifact registry integration and deterministic replay.

    TODO: Registry hook for orchestrator selection.
    TODO: Factory integration for runtime-specific instantiation.
    """

    @abstractmethod
    def prepare(self, spec: DataprepSpec) -> DataprepResult:
        """Execute dataprep pipeline.

        Args:
            spec: Dataprep execution specification

        Returns:
            Execution result with artifacts and lineage
        """
        ...

    @abstractmethod
    def validate(self, result: DataprepResult) -> bool:
        """Validate dataprep output integrity.

        Args:
            result: Dataprep result to validate

        Returns:
            True if output passes all validation checks
        """
        ...

    @abstractmethod
    def replay(self, lineage_hash: str) -> DataprepResult:
        """Replay dataprep from recorded lineage.

        Args:
            lineage_hash: Deterministic lineage identifier

        Returns:
            Replayed execution result (must match original)
        """
        ...

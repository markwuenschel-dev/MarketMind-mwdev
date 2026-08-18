from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pysrc.preprocessor.contracts.state import FitStateArtifact, PreprocessingStateManifest


def normalize_preprocessing_state(raw: Mapping[str, Any]) -> PreprocessingStateManifest:
    artifacts = tuple(
        FitStateArtifact(name=str(artifact["name"]), payload=dict(artifact.get("payload", {})))
        for artifact in raw.get("artifacts", ())
    )
    return PreprocessingStateManifest(
        schema_version=str(raw.get("schema_version", "1.0")),
        plan_version=str(raw.get("plan_version", "1.0")),
        artifacts=artifacts,
        lineage=dict(raw.get("lineage", {})),
    )

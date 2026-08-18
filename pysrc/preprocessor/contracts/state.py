from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from pysrc.ops.hashing_contract import HashingContract, HashRef

CURRENT_PREPROCESSING_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class FitStateArtifact:
    name: str
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("FitStateArtifact.name must be non-empty")
        object.__setattr__(self, "payload", dict(self.payload))

    def to_payload(self) -> dict[str, Any]:
        return {"name": self.name, "payload": dict(self.payload)}


@dataclass(frozen=True, slots=True)
class PreprocessingStateManifest:
    schema_version: str
    plan_version: str
    artifacts: tuple[FitStateArtifact, ...] = field(default_factory=tuple)
    lineage: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        object.__setattr__(self, "lineage", dict(self.lineage))
        if self.schema_version != CURRENT_PREPROCESSING_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {CURRENT_PREPROCESSING_SCHEMA_VERSION}, got {self.schema_version}"
            )
        if self.plan_version != self.schema_version:
            raise ValueError(
                f"plan_version must match schema_version {self.schema_version}, got {self.plan_version}"
            )

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_version": self.plan_version,
            "artifacts": [artifact.to_payload() for artifact in self.artifacts],
            "lineage": dict(self.lineage),
        }

    @property
    def state_id(self) -> HashRef:
        payload = self.to_payload()
        HashingContract.check_banned_values(payload)
        digest = HashingContract.hash_for_identity(
            HashingContract.canonicalize_json(payload).encode("utf-8")
        ).hex()
        return HashRef(domain="cas.v1", algo="b3-256", hex_digest=digest)

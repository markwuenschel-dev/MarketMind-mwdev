from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from pysrc.ops.hashing_contract import HashingContract, HashRef


@dataclass(frozen=True, slots=True)
class MaterializationSpec:
    format: str = "polars"
    schema_signature: str = ""
    partial_allowed: bool = False


@dataclass(frozen=True, slots=True)
class CanonicalOp:
    name: str
    params: Mapping[str, Any] = field(default_factory=dict)
    provides: tuple[str, ...] = field(default_factory=tuple)
    requires: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("CanonicalOp.name must be non-empty")
        object.__setattr__(self, "params", dict(self.params))
        object.__setattr__(self, "provides", tuple(self.provides))
        object.__setattr__(self, "requires", tuple(self.requires))

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "params": dict(self.params),
            "provides": list(self.provides),
            "requires": list(self.requires),
        }


@dataclass(frozen=True, slots=True)
class PreprocessingPlan:
    version: str
    ops: tuple[CanonicalOp, ...] = field(default_factory=tuple)
    group_by: tuple[str, ...] = field(default_factory=tuple)
    materialization: MaterializationSpec = field(default_factory=MaterializationSpec)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.version:
            raise ValueError("PreprocessingPlan.version must be non-empty")
        object.__setattr__(self, "ops", tuple(self.ops))
        object.__setattr__(self, "group_by", tuple(self.group_by))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_payload(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "ops": [op.to_payload() for op in self.ops],
            "group_by": list(self.group_by),
            "materialization": {
                "format": self.materialization.format,
                "schema_signature": self.materialization.schema_signature,
                "partial_allowed": self.materialization.partial_allowed,
            },
            "metadata": dict(self.metadata),
        }

    @property
    def plan_id(self) -> HashRef:
        payload = self.to_payload()
        HashingContract.check_banned_values(payload)
        digest = HashingContract.hash_for_identity(
            HashingContract.canonicalize_json(payload).encode("utf-8")
        ).hex()
        return HashRef(domain="cas.v1", algo="b3-256", hex_digest=digest)

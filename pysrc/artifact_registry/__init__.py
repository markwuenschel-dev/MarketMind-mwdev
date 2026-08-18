from __future__ import annotations

"""
CAS-backed artifact registry primitives.

This package provides:

- LocalCAS: filesystem-backed content-addressable storage keyed by
  domain-qualified CAS identifiers (cas.v1:b3-256:<hex>).
- HashRefs: structured references that carry both CAS identity and, for JSON
  artifacts, an attestation hash compatible with mm-gate
  (attest.v1:jcs-sha256:<hex>).

High-level policy:

- JSON canonical bytes MUST come from `marketmind_gate.hashing.canonical`.
- CAS identity is always BLAKE3-256 over those bytes.
- Gate attestation is always SHA-256 over the same bytes.

Reproducibility metadata expectations for stored artifacts: see Programming Guidelines §7.2
and ``pysrc.artifact_registry.reproducibility``.
"""

from pysrc.artifact_registry.cas import HashRefs, LocalCAS
from pysrc.artifact_registry.registry import ArtifactRegistry, CleanupReport, ResolvedArtifact
from pysrc.artifact_registry.reproducibility import (
    DETERMINISM_TIER_VALUES,
    collect_bundle_reproducibility_echo,
    json_artifact_lineage_fields,
    validate_plan_reproducibility_fields,
)

__all__ = [
    "LocalCAS",
    "HashRefs",
    "ArtifactRegistry",
    "CleanupReport",
    "ResolvedArtifact",
    "DETERMINISM_TIER_VALUES",
    "collect_bundle_reproducibility_echo",
    "json_artifact_lineage_fields",
    "validate_plan_reproducibility_fields",
]

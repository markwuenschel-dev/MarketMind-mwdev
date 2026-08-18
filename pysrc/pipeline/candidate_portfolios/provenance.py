"""Promotion bundle provenance hashing helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

from pysrc.ops.hashing.primitives.blake3_impl import Blake3Hasher
from pysrc.validation.task_validity import _bundle_sha256


def compute_promotion_provenance_hashes(
    bundle_dir: Path,
    gate_result_path: Path,
) -> dict[str, str]:
    """Return bundle CAS id and gate_result content hash for manifest pinning."""

    bundle_dir = Path(bundle_dir)
    gate_result_path = Path(gate_result_path)
    bundle_sha = _bundle_sha256(bundle_dir)
    cas_ref = Blake3Hasher().hash_artifact_id(bundle_sha.encode("utf-8"))
    gate_digest = hashlib.sha256(gate_result_path.read_bytes()).hexdigest()
    return {
        "bundle_content_hash": bundle_sha,
        "bundle_cas_id": str(cas_ref),
        "gate_result_hash": f"sha256:{gate_digest}",
    }


__all__ = ["compute_promotion_provenance_hashes"]

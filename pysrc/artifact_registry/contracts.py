from __future__ import annotations

from typing import Any

from pysrc.artifact_registry import HashRefs, LocalCAS
from pysrc.artifact_registry.run_registry import RunRegistry


def store_model_snapshot_manifest(
    cas: LocalCAS,
    run_registry: RunRegistry,
    run_id: str,
    manifest: dict[str, Any],
) -> HashRefs:
    """
    Store a model snapshot manifest via CAS + RunRegistry.

    Behavior:
        - Canonicalizes and stores the manifest as JSON in CAS.
        - Registers the artifact under role "model_snapshot_manifest"
          in the provided RunRegistry.
        - Returns HashRefs containing CAS + attestation hashes and size.
    """
    hashes = cas.put_json(manifest, media_type="application/json")
    run_registry.add_artifact(run_id, "model_snapshot_manifest", hashes)
    return hashes


def store_task_manifest(
    cas: LocalCAS,
    run_registry: RunRegistry,
    run_id: str,
    manifest: dict[str, Any],
) -> HashRefs:
    """
    Store a task manifest via CAS + RunRegistry.

    Behavior:
        - Canonicalizes and stores the manifest as JSON in CAS.
        - Registers the artifact under role "task_manifest"
          in the provided RunRegistry.
        - Returns HashRefs containing CAS + attestation hashes and size.
    """
    hashes = cas.put_json(manifest, media_type="application/json")
    run_registry.add_artifact(run_id, "task_manifest", hashes)
    return hashes

"""Integration checks for the frozen RG-09 reference verifier."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from scripts.reproduce_rg09_reference import BUNDLE_PATH, MANIFEST_PATH, compute_reference_hash

pytestmark = [pytest.mark.integration, pytest.mark.determinism("d1")]


def test_rg09_reference_hash_matches_manifest(deterministic_seed: int) -> None:
    """Frozen reference payload hash matches the registered manifest hash."""
    _ = deterministic_seed
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    assert compute_reference_hash(BUNDLE_PATH) == manifest["artifact_hash"]
    assert manifest["surface_role"] == "rg09_task_validity_anchor"
    assert "NOT the Phase II allocator incumbent comparison baseline" in manifest["note"]


def test_rg09_reference_hash_detects_parameter_perturbation(
    tmp_path: Path, deterministic_seed: int
) -> None:
    """A preprocessing/hyperparameter perturbation changes the frozen payload hash."""
    _ = deterministic_seed
    copied_bundle = tmp_path / "rg09_reference_v1"
    shutil.copytree(BUNDLE_PATH, copied_bundle)
    plan_path = copied_bundle / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["config"]["hyperparameters"]["vol_window"] = 121
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest = json.loads(
        (copied_bundle / "rg09_reference_manifest.json").read_text(encoding="utf-8")
    )
    assert compute_reference_hash(copied_bundle) != manifest["artifact_hash"]

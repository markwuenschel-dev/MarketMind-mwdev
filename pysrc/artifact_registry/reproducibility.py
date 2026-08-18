"""Reproducibility and lineage helpers (Programming Guidelines §7.2).

Used by bundle writers and the gate CLI to share determinism-tier validation and to
build consistent ``gate_result.json`` metadata echoes.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

DETERMINISM_TIER_VALUES: Final[frozenset[str]] = frozenset({"D0", "D1", "D2", "D3"})


def validate_plan_reproducibility_fields(plan: Mapping[str, Any]) -> tuple[bool, str, str]:
    """Validate optional plan.json reproducibility keys.

    Returns:
        (ok, reason_code, message). When *ok* is False, *reason_code* is suitable for a gate check.
    """
    if "determinism_tier" in plan:
        tier = plan["determinism_tier"]
        if not isinstance(tier, str) or tier not in DETERMINISM_TIER_VALUES:
            return (
                False,
                "INVALID_DETERMINISM_TIER",
                f"plan.json determinism_tier must be one of {sorted(DETERMINISM_TIER_VALUES)}, got {tier!r}",
            )
    for key in ("planner_version", "feature_op_registry_version", "seed_lineage"):
        if key not in plan:
            continue
        val = plan[key]
        if not isinstance(val, str) or not val.strip():
            return (
                False,
                "INVALID_REPRODUCIBILITY_METADATA",
                f"plan.json {key} must be a non-empty string when present",
            )
    if "partition_identity" in plan:
        pid = plan["partition_identity"]
        if not isinstance(pid, (str, dict)):
            return (
                False,
                "INVALID_REPRODUCIBILITY_METADATA",
                "plan.json partition_identity must be a string or object when present",
            )
    return True, "", ""


def collect_bundle_reproducibility_echo(bundle_path: Path) -> dict[str, Any]:
    """Best-effort aggregate of bundle files for gate_result metadata (§7.2)."""
    out: dict[str, Any] = {}
    plan_path = bundle_path / "plan.json"
    if plan_path.exists():
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            plan = None
        if isinstance(plan, dict):
            out["plan_schema_version"] = plan.get("schema_version")
            out["plan_hash"] = plan.get("plan_hash")
            out["config_hash"] = plan.get("config_hash")
            out["as_of_time"] = plan.get("as_of_time")
            for opt in (
                "determinism_tier",
                "planner_version",
                "feature_op_registry_version",
                "seed_lineage",
                "partition_identity",
            ):
                if opt in plan:
                    out[opt] = plan[opt]
    env_path = bundle_path / "env_fingerprint.json"
    if env_path.exists():
        try:
            env = json.loads(env_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            env = None
        if isinstance(env, dict):
            out["git_sha"] = env.get("git_sha")
            out["python_version"] = env.get("python_version")
    ds_path = bundle_path / "dataset_manifest.json"
    if ds_path.exists():
        try:
            ds = json.loads(ds_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            ds = None
        if isinstance(ds, dict):
            out["dataset_id"] = ds.get("dataset_id")
            out["pit_compliant"] = ds.get("pit_compliant")
            out["knowledge_time_column"] = ds.get("knowledge_time_column")
            if "content_hash" in ds:
                out["dataset_content_hash"] = ds["content_hash"]
    return out


def json_artifact_lineage_fields(
    *,
    cas_id: str,
    attest_id: str | None = None,
    schema_version: str | None = None,
    determinism_tier: str | None = None,
) -> dict[str, str]:
    """Minimum lineage key-value pairs for JSON artifacts (Guidelines §7.2, CAS policy)."""
    fields: dict[str, str] = {"cas": cas_id}
    if attest_id is not None:
        fields["attest"] = attest_id
    if schema_version is not None:
        fields["schema_version"] = schema_version
    if determinism_tier is not None:
        if determinism_tier not in DETERMINISM_TIER_VALUES:
            raise ValueError(
                f"determinism_tier must be one of {sorted(DETERMINISM_TIER_VALUES)}, got {determinism_tier!r}"
            )
        fields["determinism_tier"] = determinism_tier
    return fields

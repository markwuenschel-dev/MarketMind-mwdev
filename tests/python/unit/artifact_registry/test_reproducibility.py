from __future__ import annotations

import json
from pathlib import Path

import pytest

from pysrc.artifact_registry import LocalCAS
from pysrc.artifact_registry.reproducibility import (
    collect_bundle_reproducibility_echo,
    json_artifact_lineage_fields,
    validate_plan_reproducibility_fields,
)

pytestmark = [pytest.mark.unit, pytest.mark.determinism("d1")]


def test_validate_plan_reproducibility_fields_accepts_valid_tier() -> None:
    ok, _, _ = validate_plan_reproducibility_fields({"determinism_tier": "D0"})
    assert ok is True


def test_validate_plan_reproducibility_fields_rejects_bad_tier() -> None:
    ok, code, msg = validate_plan_reproducibility_fields({"determinism_tier": "DX"})
    assert ok is False
    assert code == "INVALID_DETERMINISM_TIER"
    assert "DX" in msg


def test_collect_bundle_reproducibility_echo_merges_plan_and_env(tmp_path: Path) -> None:
    bundle = tmp_path / "b"
    bundle.mkdir()
    (bundle / "plan.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "plan_hash": "h1",
                "config_hash": "c1",
                "as_of_time": "2026-01-01T00:00:00Z",
                "determinism_tier": "D3",
            }
        ),
        encoding="utf-8",
    )
    (bundle / "env_fingerprint.json").write_text(
        json.dumps({"git_sha": "deadbeef", "python_version": "3.12.0"}),
        encoding="utf-8",
    )
    echo = collect_bundle_reproducibility_echo(bundle)
    assert echo["plan_hash"] == "h1"
    assert echo["determinism_tier"] == "D3"
    assert echo["git_sha"] == "deadbeef"


def test_hashrefs_to_lineage_dict(tmp_path: Path) -> None:
    cas = LocalCAS(tmp_path / "store")
    refs = cas.put_json({"a": 1})
    d = refs.to_lineage_dict(schema_version="1.0.0", determinism_tier="D1")
    assert d["schema_version"] == "1.0.0"
    assert d["determinism_tier"] == "D1"
    assert d["cas"].startswith("cas.v1:")


def test_json_artifact_lineage_fields_validates_tier() -> None:
    with pytest.raises(ValueError, match="determinism_tier"):
        json_artifact_lineage_fields(cas_id="cas.v1:b3-256:ab", determinism_tier="bad")

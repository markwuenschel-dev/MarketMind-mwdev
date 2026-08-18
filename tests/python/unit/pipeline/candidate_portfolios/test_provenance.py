"""Tests for promotion provenance hashing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pysrc.pipeline.candidate_portfolios.provenance import compute_promotion_provenance_hashes


@pytest.mark.determinism("d1")
def test_compute_promotion_provenance_hashes(deterministic_seed: int, tmp_path: Path) -> None:
    _ = deterministic_seed
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "plan.json").write_text(json.dumps({"plan_hash": "abc"}), encoding="utf-8")
    gate = bundle / "gate_result.json"
    gate.write_text(json.dumps({"overall_result": "PASS"}), encoding="utf-8")
    hashes = compute_promotion_provenance_hashes(bundle, gate)
    assert hashes["bundle_cas_id"].startswith("cas.v1:b3-256:")
    assert hashes["gate_result_hash"].startswith("sha256:")
    assert hashes["bundle_content_hash"].startswith("sha256:")

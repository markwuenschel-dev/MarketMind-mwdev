"""RG-09 → MLN-06 bridge: deterministic Phase II triple from harness-shaped inputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pysrc.meta.rg09_phase2_bridge import emit_mln06_triple_for_rg09_harness
from pysrc.meta.seed_policy import build_run_identity, scaffold_int_seed_from_content_tag


@pytest.mark.determinism("d1")
def test_emit_mln06_triple_replay_byte_identical(tmp_path: Path, deterministic_seed: int) -> None:
    _ = deterministic_seed
    fixture_sha256 = "sha256:" + "a" * 64
    summary = {"date_range_start": "2020-01-01", "date_range_end": "2020-06-01"}
    episodes = pd.DataFrame()
    gate_result: dict[str, object] = {
        "decision": "PASS",
        "evidence": {
            "non_exchangeability": {"structural_separability_ratio": 0.25},
            "null_collapse": {"functional_evidence": {"harvey_t": 2.0}},
        },
    }
    ts = "2026-04-11T12:00:00Z"
    common = {
        "null_seed_namespace": "rg09.test.replay",
        "summary": summary,
        "episodes": episodes,
        "fixture_sha256": fixture_sha256,
        "generation_timestamp": ts,
        "gate_result": gate_result,
        "label_horizon_bars": 5,
    }
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    out_a.mkdir()
    out_b.mkdir()
    emit_mln06_triple_for_rg09_harness(output_dir=out_a, **common)
    emit_mln06_triple_for_rg09_harness(output_dir=out_b, **common)
    seed = scaffold_int_seed_from_content_tag(fixture_sha256)
    expected_run_id = build_run_identity(seed).run_id
    for name in ("task_manifest.json", "meta_validity_report.json", "execution_assumptions.json"):
        assert (out_a / name).read_bytes() == (out_b / name).read_bytes()
        doc = json.loads((out_a / name).read_text(encoding="utf-8"))
        assert doc["run_id"] == expected_run_id

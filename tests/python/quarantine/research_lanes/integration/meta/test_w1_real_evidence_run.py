"""End-to-end W1 real-data governed evidence (integration)."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

pytest.importorskip("xgboost")

from tests.python.unit.meta.test_task_generator import _Encoder

from pysrc.meta.phase2_artifact_contract import (
    derive_phase2_task_pool_identity_hash,
    validate_phase2_artifact_triple,
)
from pysrc.meta.regime_config import BOCPDConfig
from pysrc.meta.regime_labeler import RegimeLabeler
from pysrc.meta.reptile_trainer_benchmark import run_real_w1_baseline_evidence
from pysrc.meta.w1_governed_csv_dataview import W1GovernedCsvDataView
from pysrc.meta.w1_real_task_pool import (
    W1RealPoolConfig,
    build_w1_real_task_pool,
    w1_meta_tasks_as_manifest_inputs,
)
from pysrc.meta_learning.regime_vocabulary import REGIME_CLASS_ORDER

_FIXTURE_CSV = (
    Path(__file__).resolve().parents[3] / "fixtures" / "w1" / "spy_daily_close_fixture.csv"
)


class _Universe:
    def __call__(self, d: date) -> tuple[str, ...]:
        return ("SPY",)


@pytest.mark.determinism("d1")
@pytest.mark.integration
def test_run_real_w1_baseline_emits_valid_triple(tmp_path, deterministic_seed: int) -> None:
    _ = deterministic_seed
    bcfg = BOCPDConfig(cold_start_burn_in=40, vol_window=10, trend_window=15)
    mins = dict.fromkeys(REGIME_CLASS_ORDER, 1)
    dv = W1GovernedCsvDataView.from_csv(_FIXTURE_CSV, symbol="SPY")
    pool_cfg = W1RealPoolConfig(
        data_view=dv,
        encoder=_Encoder(),
        labeler=RegimeLabeler(bcfg),
        bocpd_config=bcfg,
        universe_resolver=_Universe(),
        signal_set_version=1,
        construction_seed=202,
        bucket_minimums=mins,
        start_ts="2010-01-04T00:00:00Z",
        end_ts="2019-12-31T00:00:00Z",
        episode_stride_days=1,
    )
    ds = {
        "pit_compliant": True,
        "knowledge_time_column": "knowledge_time",
        "content_hash": "a",
        "content_hash_expected": "a",
    }
    out_dir = tmp_path / "w1_bundle"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = run_real_w1_baseline_evidence(
        output_dir=out_dir,
        pool_cfg=pool_cfg,
        dataset_manifest=ds,
        seed=4242,
        timestamp_utc="2026-04-24T12:00:00Z",
    )
    assert res.evidence_lane == "governed_real"
    assert res.challenger_aggregate is None
    assert res.evidence_source is not None
    assert res.evidence_source.market_data_source == "governed_historical_dataview"
    assert res.evidence_source.real_market_data_evidence is True
    assert res.evidence_source.w1_gate_closure_eligible is False
    assert res.target_provenance is not None
    assert res.target_provenance.get("uses_hash_proxy") is False
    assert (out_dir / "w1_baseline_report_real.json").is_file()
    assert (out_dir / "task_manifest.json").is_file()
    assert (out_dir / "meta_validity_report.json").is_file()
    assert (out_dir / "execution_assumptions.json").is_file()

    w1_report_path = out_dir / "w1_baseline_report_real.json"
    w1_blob = json.dumps(json.loads(w1_report_path.read_text(encoding="utf-8")))
    assert "test_fixture_log_returns" not in w1_blob
    assert "deterministic_synthetic_return_stream" not in w1_blob
    assert "unit_test_fixture_log_returns_v1" not in w1_blob

    pool = build_w1_real_task_pool(pool_cfg)
    rows = w1_meta_tasks_as_manifest_inputs(pool.tasks)
    assert derive_phase2_task_pool_identity_hash(rows) == pool.task_pool_hash

    tm = json.loads((out_dir / "task_manifest.json").read_text(encoding="utf-8"))
    mv = json.loads((out_dir / "meta_validity_report.json").read_text(encoding="utf-8"))
    ex = json.loads((out_dir / "execution_assumptions.json").read_text(encoding="utf-8"))
    validate_phase2_artifact_triple(task_doc=tm, meta_doc=mv, exec_doc=ex)
    triple_blob = json.dumps({"tm": tm, "mv": mv, "ex": ex})
    assert "test_fixture_log_returns" not in triple_blob
    assert "deterministic_synthetic_return_stream" not in triple_blob
    assert "unit_test_fixture_log_returns_v1" not in triple_blob
    bc = mv["baseline_comparison"]
    assert bc["data_parity"] == "governed_real_meta_task_pool"
    assert bc["net_result"] == "INSUFFICIENT_REAL_DATA"
    assert bc["evidence_source"]["market_data_source"] == "governed_historical_dataview"
    assert bc["evidence_source"]["real_market_data_evidence"] is True
    assert bc["evidence_source"]["pit_heterogeneity_governance_acknowledged"] is False
    assert "STUB_PENDING_REAL_DATA" not in json.dumps(mv)

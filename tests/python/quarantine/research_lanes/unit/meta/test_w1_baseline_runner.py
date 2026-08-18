from __future__ import annotations

# ruff: noqa: S101
import json
import math
import re
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import pysrc.meta.w1_baseline_runner as w1r
from pysrc.meta.reptile_k_sweep_errors import ArtifactImmutabilityError
from pysrc.meta.reptile_trainer_benchmark import run_default_w1_baseline_evidence
from pysrc.meta.task import MAX_SIGNALS, MetaTask
from pysrc.meta.w1_baseline_config import (
    W1BaselineConfig,
    W1BaselineConfigError,
    W1EvidenceSanityConfig,
)
from pysrc.meta.w1_baseline_io import (
    build_w1_synthetic_task_pool,
    emit_w1_baseline_example_json,
    emit_w1_baseline_report_json,
    recompute_w1_content_hash_from_document,
    w1_baseline_result_to_document,
)
from pysrc.meta.w1_baseline_runner import run_w1_baseline_evidence
from pysrc.meta_learning.regime_vocabulary import REGIME_CLASS_ORDER


def _ids() -> tuple[str, ...]:
    return tuple(f"s{i}" for i in range(MAX_SIGNALS))


def _mask() -> tuple[bool, ...]:
    return tuple(i == 0 for i in range(MAX_SIGNALS))


def _bull_only_task(i: int) -> MetaTask:
    day = datetime(2024, 7, 1, tzinfo=UTC) + timedelta(days=i * 7)
    support = tuple((day + timedelta(days=j)).isoformat() for j in range(6))
    query = tuple((day + timedelta(days=20 + j)).isoformat() for j in range(4))
    emb = np.full(4, 0.01 * float(i), dtype=np.float32)
    return MetaTask(
        task_id=f"w1-bullonly-{i:05d}",
        regime_id="trend_hi__vol_med__bocpd_stable",
        regime_class="bull",
        t0=support[0],
        t1=(day + timedelta(days=45)).isoformat(),
        pit_boundary=support[-1],
        support_set=support,
        query_set=query,
        signal_ids=_ids(),
        signal_mask=_mask(),
        signal_set_version="1",
        signal_ids_hash="sha256:w1_stub",
        horizon=1,
        active_k=1,
        regime_embedding=emb,
    )


def _no_print_calls_in_modules() -> None:
    root = Path(__file__).resolve().parents[4]
    paths = [
        root / "pysrc/meta/w1_baseline_config.py",
        root / "pysrc/meta/w1_baseline_io.py",
        root / "pysrc/meta/w1_baseline_runner.py",
        root / "pysrc/meta/w1_baseline_types.py",
        root / "pysrc/meta/w1_baseline_incumbent.py",
        root / "pysrc/meta/w1_real_task_pool.py",
        root / "pysrc/meta/w1_evidence_fingerprints.py",
        root / "pysrc/meta/w1_governed_csv_dataview.py",
        root / "pysrc/meta/w1_challenger_surface.py",
        root / "pysrc/meta/w1_evidence_sanity.py",
        root / "pysrc/meta/w1_reptile_challenger_bridge.py",
        root / "pysrc/meta/run_w1_real_challenger_evidence.py",
        root / "pysrc/meta/w1_fold_safe_meta_allocator_adapter.py",
        root / "pysrc/meta/reptile_trainer_benchmark.py",
    ]
    for p in paths:
        text = p.read_text(encoding="utf-8")
        # Word-boundary: avoid false positives on identifiers like ``...fingerprint(...)``.
        assert re.search(r"(?<![A-Za-z0-9_])print\s*\(", text) is None, p.name


@pytest.mark.determinism("d1")
def test_w1_no_print_in_harness_modules() -> None:
    _no_print_calls_in_modules()


@pytest.mark.determinism("d1")
def test_w1_default_run_returns_result(tmp_path: Path, deterministic_seed) -> None:
    """Contract: default entrypoint is ``run_default_w1_baseline_evidence``."""
    _ = deterministic_seed
    cfg = replace(W1BaselineConfig(), artifact_dir=str(tmp_path / "w1def"))
    res = run_default_w1_baseline_evidence(cfg, n_synthetic_tasks=50)
    assert res.schema_version == "w1_baseline.v2"
    assert res.baseline_id == cfg.baseline_id
    assert len(res.fold_results) == cfg.n_walk_forward_folds
    assert (tmp_path / "w1def" / "w1_baseline_report.json").is_file()
    assert (tmp_path / "w1def" / "w1_baseline_example.json").is_file()


@pytest.mark.determinism("d1")
def test_synthetic_run_emits_non_real_evidence_flags(deterministic_seed: int) -> None:
    _ = deterministic_seed
    cfg = W1BaselineConfig()
    pool = build_w1_synthetic_task_pool(50, cfg)
    res = run_w1_baseline_evidence(cfg, pool)
    doc = w1_baseline_result_to_document(res)
    assert doc["evidence_source"]["real_market_data_evidence"] is False
    assert doc["evidence_source"]["w1_gate_closure_eligible"] is False
    assert doc["evidence_source"]["pit_heterogeneity_governance_acknowledged"] is False
    assert doc["evidence_source"]["market_data_source"] == "deterministic_synthetic_return_stream"


@pytest.mark.determinism("d1")
def test_w1_fold_count(deterministic_seed) -> None:
    _ = deterministic_seed
    cfg = W1BaselineConfig(n_walk_forward_folds=5)
    pool = build_w1_synthetic_task_pool(80, cfg)
    res = run_w1_baseline_evidence(cfg, pool)
    assert len(res.fold_results) == 5
    assert {f.fold_index for f in res.fold_results} == set(range(5))


@pytest.mark.determinism("d1")
def test_w1_fold_windows_distinct(deterministic_seed) -> None:
    _ = deterministic_seed
    cfg = W1BaselineConfig()
    pool = build_w1_synthetic_task_pool(50, cfg)
    res = run_w1_baseline_evidence(cfg, pool)
    starts = [f.fold_start for f in res.fold_results]
    assert len(set(starts)) == len(starts)


@pytest.mark.determinism("d1")
def test_w1_report_and_example_differ(tmp_path: Path, deterministic_seed) -> None:
    _ = deterministic_seed
    cfg = replace(W1BaselineConfig(), artifact_dir=str(tmp_path / "wdiff"))
    run_default_w1_baseline_evidence(cfg, n_synthetic_tasks=50)
    rep = json.loads((tmp_path / "wdiff" / "w1_baseline_report.json").read_text(encoding="utf-8"))
    ex = json.loads((tmp_path / "wdiff" / "w1_baseline_example.json").read_text(encoding="utf-8"))
    assert rep["content_hash"]["value"] != ex["content_hash"]["value"]
    assert rep["n_folds"] != ex["n_folds"]


@pytest.mark.determinism("d1")
def test_w1_content_hash_determinism(deterministic_seed) -> None:
    _ = deterministic_seed
    cfg = W1BaselineConfig()
    pool = build_w1_synthetic_task_pool(50, cfg)

    def _fake_ts1() -> datetime:
        return datetime(2026, 4, 22, 12, 0, 0, tzinfo=UTC)

    def _fake_ts2() -> datetime:
        return datetime(2027, 1, 1, 0, 0, 0, tzinfo=UTC)

    with patch("pysrc.meta.w1_baseline_runner.datetime") as mdt:
        mdt.now = lambda tz=None: _fake_ts1()
        mdt.UTC = UTC
        r1 = run_w1_baseline_evidence(cfg, pool)
    with patch("pysrc.meta.w1_baseline_runner.datetime") as mdt:
        mdt.now = lambda tz=None: _fake_ts2()
        mdt.UTC = UTC
        r2 = run_w1_baseline_evidence(cfg, pool)
    assert r1.run_timestamp_utc != r2.run_timestamp_utc
    assert r1.content_hash["value"] == r2.content_hash["value"]


@pytest.mark.determinism("d1")
def test_w1_content_hash_excludes_timestamp(deterministic_seed) -> None:
    _ = deterministic_seed
    cfg = W1BaselineConfig()
    pool = build_w1_synthetic_task_pool(50, cfg)
    with patch("pysrc.meta.w1_baseline_runner.datetime") as mdt:
        mdt.now = lambda tz=None: datetime(2026, 4, 22, 12, 0, 0, tzinfo=UTC)
        mdt.UTC = UTC
        doc = w1_baseline_result_to_document(run_w1_baseline_evidence(cfg, pool))
    h = recompute_w1_content_hash_from_document(doc)
    assert h == doc["content_hash"]["value"]
    doc2 = dict(doc)
    doc2["run_timestamp_utc"] = "2099-01-01T00:00:00Z"
    assert recompute_w1_content_hash_from_document(doc2) == h


@pytest.mark.determinism("d1")
def test_w1_schema_version(deterministic_seed) -> None:
    _ = deterministic_seed
    cfg = W1BaselineConfig(schema_version="w1_baseline.v2")
    pool = build_w1_synthetic_task_pool(50, cfg)
    res = run_w1_baseline_evidence(cfg, pool)
    assert res.schema_version == "w1_baseline.v2"
    assert w1_baseline_result_to_document(res)["schema_version"] == "w1_baseline.v2"


@pytest.mark.determinism("d1")
def test_w1_cost_assumptions_present(deterministic_seed) -> None:
    _ = deterministic_seed
    cfg = W1BaselineConfig(spread_bps=5.0, slippage_bps=2.0, borrow_rate_ann=0.005)
    pool = build_w1_synthetic_task_pool(50, cfg)
    res = run_w1_baseline_evidence(cfg, pool)
    assert res.cost_assumptions.spread_bps == 5.0
    assert res.cost_assumptions.slippage_bps == 2.0
    assert res.cost_assumptions.borrow_rate_ann == 0.005
    d = w1_baseline_result_to_document(res)["cost_assumptions"]
    assert d == {"spread_bps": 5.0, "slippage_bps": 2.0, "borrow_rate_ann": 0.005}


@pytest.mark.determinism("d1")
def test_w1_threshold_references_ids(deterministic_seed) -> None:
    _ = deterministic_seed
    cfg = W1BaselineConfig()
    pool = build_w1_synthetic_task_pool(50, cfg)
    res = run_w1_baseline_evidence(cfg, pool)
    ids = [r.id for r in res.threshold_references]
    assert ids == sorted(ids)
    assert set(ids) == {"THR-W1-IC01", "THR-W1-NS01"}


@pytest.mark.determinism("d1")
def test_w1_gate_ii_posture(deterministic_seed) -> None:
    _ = deterministic_seed
    cfg = W1BaselineConfig()
    pool = build_w1_synthetic_task_pool(50, cfg)
    res = run_w1_baseline_evidence(cfg, pool)
    assert res.gate_ii_posture == "DEFERRED"
    assert w1_baseline_result_to_document(res)["gate_ii_posture"] == "DEFERRED"


@pytest.mark.determinism("d1")
def test_w1_aggregate_finite(deterministic_seed) -> None:
    _ = deterministic_seed
    cfg = W1BaselineConfig()
    pool = build_w1_synthetic_task_pool(50, cfg)
    a = run_w1_baseline_evidence(cfg, pool).aggregate
    for name, v in (
        ("mean_net_sharpe", a.mean_net_sharpe),
        ("mean_gross_ic", a.mean_gross_ic),
        ("mean_max_drawdown", a.mean_max_drawdown),
        ("mean_turnover", a.mean_turnover),
    ):
        assert math.isfinite(v), name
    if a.mean_crisis_ic is not None:
        assert math.isfinite(a.mean_crisis_ic)
    if a.mean_net_sharpe_normal_null_p is not None:
        assert math.isfinite(a.mean_net_sharpe_normal_null_p)


@pytest.mark.determinism("d1")
def test_w1_regime_breakdown_is_per_regime_ic(deterministic_seed) -> None:
    _ = deterministic_seed
    cfg = W1BaselineConfig()
    pool = build_w1_synthetic_task_pool(50, cfg)
    f0 = run_w1_baseline_evidence(cfg, pool).fold_results[0]
    rb = f0.regime_breakdown
    assert set(rb.keys()) == set(REGIME_CLASS_ORDER)
    vals = [v for v in rb.values() if v is not None]
    assert vals
    assert all(isinstance(v, float) and math.isfinite(v) for v in vals)


@pytest.mark.determinism("d1")
def test_w1_example_json_emitted(tmp_path: Path, deterministic_seed) -> None:
    _ = deterministic_seed
    cfg = W1BaselineConfig(artifact_dir=str(tmp_path / "w1"))
    pool = build_w1_synthetic_task_pool(50, cfg)
    res = run_w1_baseline_evidence(cfg, pool)
    ex = tmp_path / "w1" / "w1_baseline_example.json"
    ex.parent.mkdir(parents=True, exist_ok=True)
    emit_w1_baseline_example_json(ex, res)
    loaded = json.loads(ex.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == cfg.schema_version
    assert "fold_results" in loaded
    assert loaded["content_hash"]["algorithm"] == "sha256"


@pytest.mark.determinism("d1")
def test_w1_emit_immutability(tmp_path: Path, deterministic_seed) -> None:
    _ = deterministic_seed
    cfg = W1BaselineConfig()
    pool = build_w1_synthetic_task_pool(50, cfg)
    res = run_w1_baseline_evidence(cfg, pool)
    p = tmp_path / "rep.json"
    emit_w1_baseline_report_json(p, res)
    with pytest.raises(ArtifactImmutabilityError):
        emit_w1_baseline_report_json(p, res)
    ex = tmp_path / "ex.json"
    emit_w1_baseline_example_json(ex, res)
    with pytest.raises(ArtifactImmutabilityError):
        emit_w1_baseline_example_json(ex, res)


@pytest.mark.determinism("d1")
def test_w1_config_regime_mismatch() -> None:
    with pytest.raises(W1BaselineConfigError):
        W1BaselineConfig(regime_classes=("bull",))


@pytest.mark.determinism("d1")
def test_w1_config_invalid_folds() -> None:
    with pytest.raises(W1BaselineConfigError):
        W1BaselineConfig(n_walk_forward_folds=0)


@pytest.mark.determinism("d1")
def test_w1_config_invalid_fold_length() -> None:
    with pytest.raises(W1BaselineConfigError):
        W1BaselineConfig(fold_length_bars=0)


@pytest.mark.determinism("d1")
def test_w1_config_invalid_warmup() -> None:
    with pytest.raises(W1BaselineConfigError):
        W1BaselineConfig(warmup_bars=-1)


@pytest.mark.determinism("d1")
def test_w1_config_invalid_min_tasks() -> None:
    with pytest.raises(W1BaselineConfigError):
        W1BaselineConfig(min_tasks_per_class=0)


@pytest.mark.determinism("d1")
def test_w1_config_negative_slippage() -> None:
    with pytest.raises(W1BaselineConfigError):
        W1BaselineConfig(slippage_bps=-1.0)


@pytest.mark.determinism("d1")
def test_w1_config_whitespace_baseline_id() -> None:
    with pytest.raises(W1BaselineConfigError):
        W1BaselineConfig(baseline_id="   ")


@pytest.mark.determinism("d1")
def test_w1_config_whitespace_artifact_dir() -> None:
    with pytest.raises(W1BaselineConfigError):
        W1BaselineConfig(artifact_dir="  ")


@pytest.mark.determinism("d1")
def test_w1_config_whitespace_report_filename() -> None:
    with pytest.raises(W1BaselineConfigError):
        W1BaselineConfig(report_filename="")


@pytest.mark.determinism("d1")
def test_w1_config_whitespace_example_filename() -> None:
    with pytest.raises(W1BaselineConfigError):
        W1BaselineConfig(example_filename="\t")


@pytest.mark.determinism("d1")
def test_w1_config_whitespace_schema_version() -> None:
    with pytest.raises(W1BaselineConfigError):
        W1BaselineConfig(schema_version="")


@pytest.mark.determinism("d1")
def test_w1_config_invalid_cost() -> None:
    with pytest.raises(W1BaselineConfigError):
        W1BaselineConfig(spread_bps=float("nan"))


@pytest.mark.determinism("d1")
def test_w1_config_governed_rejects_fixture_smoke_sanity_profile() -> None:
    with pytest.raises(W1BaselineConfigError, match="governed_closure cannot use"):
        W1BaselineConfig(evidence_sanity=W1EvidenceSanityConfig.fixture_smoke())


@pytest.mark.determinism("d1")
def test_w1_config_fixture_smoke_mode_accepts_fixture_profile() -> None:
    c = W1BaselineConfig(
        evidence_sanity_mode="fixture_smoke",
        evidence_sanity=W1EvidenceSanityConfig.fixture_smoke(),
    )
    assert c.evidence_sanity_mode == "fixture_smoke"
    assert c.evidence_sanity.is_fixture_smoke_profile is True


@pytest.mark.determinism("d1")
def test_w1_pool_too_small() -> None:
    cfg = W1BaselineConfig()
    with pytest.raises(ValueError, match="below min coverage"):
        build_w1_synthetic_task_pool(10, cfg)


@pytest.mark.determinism("d1")
def test_w1_pool_zero_tasks() -> None:
    cfg = W1BaselineConfig()
    with pytest.raises(ValueError, match="n_tasks must be"):
        build_w1_synthetic_task_pool(0, cfg)


@pytest.mark.determinism("d1")
def test_w1_empty_task_pool_raises() -> None:
    with pytest.raises(ValueError, match="task pool empty"):
        run_w1_baseline_evidence(W1BaselineConfig(), ())


@pytest.mark.determinism("d1")
def test_w1_crisis_ic_sometimes_populated(deterministic_seed) -> None:
    _ = deterministic_seed
    cfg = W1BaselineConfig()
    pool = build_w1_synthetic_task_pool(50, cfg)
    res = run_w1_baseline_evidence(cfg, pool)
    assert res.aggregate.n_folds_with_crisis >= 1
    assert any(f.crisis_ic is not None for f in res.fold_results)


@pytest.mark.determinism("d1")
def test_w1_runner_helpers_edge_cases() -> None:
    x1 = np.asarray([1.0], dtype=np.float64)
    y1 = np.asarray([2.0], dtype=np.float64)
    assert w1r._pearson_ic(x1, y1) == 0.0
    z = np.zeros(0, dtype=np.float64)
    assert w1r._max_drawdown(z) == 0.0
    assert w1r._mean_sharpe_normal_null_p_value(1.0, 1) is None
    assert w1r._mean_sharpe_normal_null_p_value(float("nan"), 5) is None
    ones = np.ones(3, dtype=np.float64)
    assert w1r._pearson_ic(ones, ones) == 0.0
    with pytest.raises(TypeError):
        recompute_w1_content_hash_from_document([])  # type: ignore[arg-type]
    with patch("pysrc.meta.w1_baseline_runner.math.sqrt", return_value=float("inf")):
        assert w1r._mean_sharpe_normal_null_p_value(1.0, 3) is None


@pytest.mark.determinism("d1")
def test_w1_expanded_timeline_invalid_span() -> None:
    class _BadSpan:
        warmup_bars = -20
        n_walk_forward_folds = 1
        fold_length_bars = 2

    one = build_w1_synthetic_task_pool(5, W1BaselineConfig(min_tasks_per_class=1))[0]
    with pytest.raises(ValueError, match="invalid walk-forward span"):
        w1r._expanded_timeline([one], _BadSpan())  # type: ignore[arg-type]


@pytest.mark.determinism("d1")
def test_w1_fold_without_crisis_tasks() -> None:
    cfg = W1BaselineConfig(
        n_walk_forward_folds=1,
        fold_length_bars=2,
        warmup_bars=0,
        min_tasks_per_class=1,
    )
    pool = tuple(_bull_only_task(i) for i in range(5))
    res = run_w1_baseline_evidence(cfg, pool)
    assert res.fold_results[0].crisis_ic is None


@pytest.mark.determinism("d1")
def test_w1_single_bar_fold_turnover_zero(deterministic_seed) -> None:
    _ = deterministic_seed
    cfg = W1BaselineConfig(
        n_walk_forward_folds=1,
        fold_length_bars=1,
        warmup_bars=0,
        min_tasks_per_class=1,
    )
    pool = tuple(_bull_only_task(i) for i in range(5))
    with patch("pysrc.meta.w1_baseline_runner.np.std", return_value=0.0):
        res = run_w1_baseline_evidence(cfg, pool)
    assert res.fold_results[0].net_sharpe == 0.0
    assert res.fold_results[0].turnover == 0.0


@pytest.mark.determinism("d1")
def test_run_default_w1_smoke(tmp_path: Path, deterministic_seed) -> None:
    """``run_default_w1_baseline_evidence`` writes under a temp artifact root."""
    _ = deterministic_seed
    cfg = replace(W1BaselineConfig(), artifact_dir=str(tmp_path / "w1out"))
    res = run_default_w1_baseline_evidence(cfg, n_synthetic_tasks=50)
    assert res.gate_ii_posture == "DEFERRED"
    assert (tmp_path / "w1out" / "w1_baseline_report.json").is_file()
    assert (tmp_path / "w1out" / "w1_baseline_example.json").is_file()


@pytest.mark.determinism("d1")
def test_w1_runner_valid_challenger_populates_challenger_aggregate(deterministic_seed: int) -> None:
    pytest.importorskip("xgboost")
    _ = deterministic_seed
    from pysrc.meta.phase2_artifact_contract import (
        canonical_content_hash,
        derive_phase2_task_pool_identity_hash,
    )
    from pysrc.meta.w1_baseline_adapter import build_baseline_comparison
    from pysrc.meta.w1_baseline_incumbent import XGBoostIncumbentBaseline
    from pysrc.meta.w1_baseline_types import W1ComparisonFingerprints, W1EvidenceSource
    from pysrc.meta.w1_challenger_surface import derive_w1_fold_plan
    from pysrc.meta.w1_evidence_fingerprints import derive_w1_splits_fingerprint
    from pysrc.meta.w1_fold_safe_meta_allocator_adapter import W1FoldSafeSupportOnlyMetaAllocator
    from pysrc.meta.w1_real_task_pool import w1_meta_tasks_as_manifest_inputs
    from pysrc.meta.w1_reptile_challenger_bridge import build_reptile_w1_challenger_surface

    cfg = W1BaselineConfig(n_walk_forward_folds=2, fold_length_bars=4, warmup_bars=2)
    pool = build_w1_synthetic_task_pool(35, cfg)
    ordered = sorted(pool, key=lambda t: (t.t0, t.task_id))
    tqt = {t.task_id: 0.015 + 0.001 * i for i, t in enumerate(ordered)}
    tst = {t.task_id: 0.008 + 0.0005 * i for i, t in enumerate(ordered)}
    cost_d = {
        "spread_bps": float(cfg.spread_bps),
        "slippage_bps": float(cfg.slippage_bps),
        "borrow_rate_ann": float(cfg.borrow_rate_ann),
    }
    cost_fp = canonical_content_hash({"cost_assumptions": cost_d})
    splits_fp = derive_w1_splits_fingerprint(cfg=cfg)
    data_fp = "sha256:" + "3" * 64
    rows = w1_meta_tasks_as_manifest_inputs(tuple(ordered))
    tph = derive_phase2_task_pool_identity_hash(rows)
    fold_plan = derive_w1_fold_plan(ordered, cfg)
    surface = build_reptile_w1_challenger_surface(
        task_pool=tuple(ordered),
        fold_plan=fold_plan,
        trained_state=W1FoldSafeSupportOnlyMetaAllocator(state_fingerprint=tph),
        task_pool_hash=tph,
        data_fingerprint=data_fp,
        splits_fingerprint=splits_fp,
        cost_assumptions_fingerprint=cost_fp,
        signal_set_version=str(ordered[0].signal_set_version),
        created_at_utc="2026-04-24T16:00:00Z",
    )
    ev = W1EvidenceSource(
        market_data_source="governed_historical_dataview",
        label_source="governed_pit_regime_label_surface",
        target_source="query_realized_net_utility",
        challenger_source=surface.source,
        real_market_data_evidence=True,
        w1_gate_closure_eligible=False,
    )
    fp = W1ComparisonFingerprints(
        data_fingerprint=data_fp,
        splits_fingerprint=splits_fp,
        cost_assumptions_fingerprint=cost_fp,
    )
    res = run_w1_baseline_evidence(
        cfg,
        tuple(ordered),
        evidence_lane="governed_real",
        incumbent=XGBoostIncumbentBaseline(),
        task_query_targets=tqt,
        task_support_targets=tst,
        evidence_source=ev,
        comparison_fingerprints=fp,
        challenger_surface=surface,
        task_pool_hash=tph,
    )
    assert res.challenger_aggregate is not None
    assert res.challenger_fold_results is not None
    assert res.evidence_source is not None
    assert res.evidence_source.w1_gate_closure_eligible is False
    bc = build_baseline_comparison(
        res,
        challenger_run_id="run.sha256:" + "9" * 64,
        task_pool_hash=tph,
    )
    assert bc["net_result"] == "INSUFFICIENT_REAL_DATA"


@pytest.mark.determinism("d1")
def test_w1_fixture_smoke_mode_suppresses_gate_closure_despite_sanity_ok(
    deterministic_seed: int,
) -> None:
    """ADR-006: fixture_smoke lane never sets w1_gate_closure_eligible true (patches simulate OK preconditions)."""
    pytest.importorskip("xgboost")
    _ = deterministic_seed
    from pysrc.meta.phase2_artifact_contract import (
        canonical_content_hash,
        derive_phase2_task_pool_identity_hash,
    )
    from pysrc.meta.w1_baseline_incumbent import XGBoostIncumbentBaseline
    from pysrc.meta.w1_baseline_types import W1ComparisonFingerprints, W1EvidenceSource
    from pysrc.meta.w1_challenger_surface import derive_w1_fold_plan
    from pysrc.meta.w1_evidence_fingerprints import derive_w1_splits_fingerprint
    from pysrc.meta.w1_fold_safe_meta_allocator_adapter import W1FoldSafeSupportOnlyMetaAllocator
    from pysrc.meta.w1_real_task_pool import w1_meta_tasks_as_manifest_inputs
    from pysrc.meta.w1_reptile_challenger_bridge import build_reptile_w1_challenger_surface

    cfg = W1BaselineConfig(
        n_walk_forward_folds=2,
        fold_length_bars=4,
        warmup_bars=2,
        evidence_sanity_mode="fixture_smoke",
        evidence_sanity=W1EvidenceSanityConfig.fixture_smoke(),
    )
    pool = build_w1_synthetic_task_pool(35, cfg)
    ordered = sorted(pool, key=lambda t: (t.t0, t.task_id))
    tqt = {t.task_id: 0.015 + 0.001 * i for i, t in enumerate(ordered)}
    tst = {t.task_id: 0.008 + 0.0005 * i for i, t in enumerate(ordered)}
    cost_d = {
        "spread_bps": float(cfg.spread_bps),
        "slippage_bps": float(cfg.slippage_bps),
        "borrow_rate_ann": float(cfg.borrow_rate_ann),
    }
    cost_fp = canonical_content_hash({"cost_assumptions": cost_d})
    splits_fp = derive_w1_splits_fingerprint(cfg=cfg)
    data_fp = "sha256:" + "3" * 64
    rows = w1_meta_tasks_as_manifest_inputs(tuple(ordered))
    tph = derive_phase2_task_pool_identity_hash(rows)
    fold_plan = derive_w1_fold_plan(ordered, cfg)
    surface = build_reptile_w1_challenger_surface(
        task_pool=tuple(ordered),
        fold_plan=fold_plan,
        trained_state=W1FoldSafeSupportOnlyMetaAllocator(state_fingerprint=tph),
        task_pool_hash=tph,
        data_fingerprint=data_fp,
        splits_fingerprint=splits_fp,
        cost_assumptions_fingerprint=cost_fp,
        signal_set_version=str(ordered[0].signal_set_version),
        created_at_utc="2026-04-24T16:00:00Z",
    )
    ev = W1EvidenceSource(
        market_data_source="governed_historical_dataview",
        label_source="governed_pit_regime_label_surface",
        target_source="query_realized_net_utility",
        challenger_source=surface.source,
        real_market_data_evidence=True,
        w1_gate_closure_eligible=False,
        pit_heterogeneity_governance_acknowledged=True,
    )
    fp = W1ComparisonFingerprints(
        data_fingerprint=data_fp,
        splits_fingerprint=splits_fp,
        cost_assumptions_fingerprint=cost_fp,
    )
    with (
        patch(
            "pysrc.meta.w1_baseline_runner.w1_challenger_surface_closure_eligible",
            return_value=True,
        ),
        patch(
            "pysrc.meta.w1_baseline_runner.w1_evidence_sanity_closure_ok", return_value=(True, ())
        ),
    ):
        res = run_w1_baseline_evidence(
            cfg,
            tuple(ordered),
            evidence_lane="governed_real",
            incumbent=XGBoostIncumbentBaseline(),
            task_query_targets=tqt,
            task_support_targets=tst,
            evidence_source=ev,
            comparison_fingerprints=fp,
            challenger_surface=surface,
            task_pool_hash=tph,
        )
    rep = res.evidence_sanity_report
    assert rep is not None
    assert rep["w1_closure_suppressed_by_evidence_sanity_mode"] is True
    assert rep["w1_gate_closure_eligible"] is False
    assert res.evidence_source is not None
    assert res.evidence_source.w1_gate_closure_eligible is False

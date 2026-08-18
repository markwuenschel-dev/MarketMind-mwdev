"""W1 governed real lane: XGBoost incumbent vs meta-allocator challenger integration."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
from tests.python.unit.meta.test_task_generator import _Encoder

from pysrc.meta.curriculum import CurriculumSampler, CurriculumSamplerConfig
from pysrc.meta.phase2_artifact_contract import (
    canonical_content_hash,
    derive_phase2_task_pool_identity_hash,
    validate_phase2_artifact_triple,
)
from pysrc.meta.regime_config import BOCPDConfig
from pysrc.meta.regime_labeler import RegimeLabeler
from pysrc.meta.reptile_trainer_benchmark import run_real_w1_baseline_evidence
from pysrc.meta.reptile_trainer_config import ReptileTrainerConfig
from pysrc.meta.w1_baseline_adapter import build_baseline_comparison
from pysrc.meta.w1_baseline_config import W1BaselineConfig
from pysrc.meta.w1_baseline_errors import W1BaselineEvidenceError
from pysrc.meta.w1_baseline_incumbent import XGBoostIncumbentBaseline
from pysrc.meta.w1_baseline_runner import run_w1_baseline_evidence
from pysrc.meta.w1_baseline_types import W1ComparisonFingerprints, W1EvidenceSource
from pysrc.meta.w1_challenger_surface import (
    W1_CHALLENGER_SURFACE_SCHEMA_VERSION,
    W1ChallengerSurface,
    derive_w1_fold_plan,
    w1_challenger_surface_closure_eligible,
)
from pysrc.meta.w1_evidence_fingerprints import derive_w1_splits_fingerprint
from pysrc.meta.w1_fold_safe_meta_allocator_adapter import W1FoldSafeSupportOnlyMetaAllocator
from pysrc.meta.w1_governed_csv_dataview import W1GovernedCsvDataView
from pysrc.meta.w1_governed_learned_checkpoint_emit import emit_w1_learned_checkpoint_v2
from pysrc.meta.w1_governed_meta_head_trainer import (
    TRAINED_BY_W1_GOVERNED_META_HEAD_V1,
    W1GovernedMetaHeadTrainConfig,
    W1GovernedMetaHeadTrainSuccess,
    fit_w1_governed_meta_head,
)
from pysrc.meta.w1_real_task_pool import (
    W1RealPoolConfig,
    W1RealTaskPoolOutcome,
    build_w1_real_task_pool,
    w1_meta_tasks_as_manifest_inputs,
    w1_support_mean_log_return_net_from_pool,
)
from pysrc.meta.w1_reptile_challenger_bridge import build_reptile_w1_challenger_surface
from pysrc.meta.w1_reptile_trained_meta_allocator_adapter import ReptileTrainedMetaAllocatorAdapter
from pysrc.meta_learning.regime_vocabulary import REGIME_CLASS_ORDER

_FIXTURE_CSV = (
    Path(__file__).resolve().parents[3] / "fixtures" / "w1" / "spy_daily_close_fixture.csv"
)


def _require_xgboost() -> None:
    """Load xgboost only for tests that need the real incumbent stack (avoids eager DLL import for the whole module)."""
    pytest.importorskip("xgboost")


class _Universe:
    def __call__(self, d: date) -> tuple[str, ...]:
        return ("SPY",)


def _pool_and_fingerprints():
    bcfg = BOCPDConfig(cold_start_burn_in=40, vol_window=10, trend_window=15)
    # Decision-grade W1 fold admissibility needs enough unique tasks for non-overlapping splits.
    mins = dict.fromkeys(REGIME_CLASS_ORDER, 24)
    dv = W1GovernedCsvDataView.from_csv(_FIXTURE_CSV, symbol="SPY")
    pool_cfg = W1RealPoolConfig(
        data_view=dv,
        encoder=_Encoder(),
        labeler=RegimeLabeler(bcfg),
        bocpd_config=bcfg,
        universe_resolver=_Universe(),
        signal_set_version=1,
        construction_seed=303,
        bucket_minimums=mins,
        start_ts="2010-01-04T00:00:00Z",
        end_ts="2019-12-31T00:00:00Z",
        episode_stride_days=1,
    )
    pool_out = build_w1_real_task_pool(pool_cfg)
    w1_cfg = W1BaselineConfig(random_seed=4242)
    cost_d = {
        "spread_bps": float(w1_cfg.spread_bps),
        "slippage_bps": float(w1_cfg.slippage_bps),
        "borrow_rate_ann": float(w1_cfg.borrow_rate_ann),
    }
    cost_fp = canonical_content_hash({"cost_assumptions": cost_d})
    splits_fp = derive_w1_splits_fingerprint(cfg=w1_cfg)
    data_fp = pool_out.data_fingerprint
    assert isinstance(data_fp, str)
    assert data_fp.startswith("sha256:")
    return pool_out, pool_cfg, w1_cfg, cost_fp, splits_fp, str(data_fp)


def _learned_challenger_surface_for_spy_pool(
    tmp_path: Path,
    *,
    pool_out: W1RealTaskPoolOutcome,
    pool_cfg: W1RealPoolConfig,
    w1_cfg: W1BaselineConfig,
    cost_fp: str,
    splits_fp: str,
    data_fp: str,
    deterministic_seed: int,
) -> tuple[W1ChallengerSurface, ReptileTrainedMetaAllocatorAdapter]:
    """Build trainer-emitted governed checkpoint + reptile challenger surface (SPY fixture pool)."""
    tasks_sorted = tuple(sorted(pool_out.tasks, key=lambda t: (t.t0, t.task_id)))
    mins = dict.fromkeys(REGIME_CLASS_ORDER, 1)
    rtc = ReptileTrainerConfig()
    samp = CurriculumSampler(
        tasks_sorted,
        config=CurriculumSamplerConfig(
            batch_size=len(tasks_sorted),
            crisis_floor_fraction=float(rtc.crisis_floor_pct),
            bucket_minimums=mins,
            seed=int(pool_cfg.construction_seed),
        ),
    )
    targets = w1_support_mean_log_return_net_from_pool(pool_out)
    fit_out = fit_w1_governed_meta_head(
        sampler=samp,
        reptile_config=rtc,
        support_targets_by_task_id=targets,
        head_cfg=W1GovernedMetaHeadTrainConfig(outer_lr=0.08, n_gradient_steps=150),
        seed=int(deterministic_seed),
    )
    assert isinstance(fit_out, W1GovernedMetaHeadTrainSuccess)
    ck_path = tmp_path / "trainer_emitted_w1_checkpoint.json"
    emit_w1_learned_checkpoint_v2(
        ck_path,
        weights=[float(x) for x in fit_out.weights.tolist()],
        model_state_hash=fit_out.model_state_hash,
        trainer_config_hash=fit_out.trainer_config_hash,
        training_task_pool_hash=pool_out.task_pool_hash,
        training_data_fingerprint=data_fp,
        training_splits_fingerprint=splits_fp,
        signal_set_version=str(pool_cfg.signal_set_version),
        feature_encoder_contract_version="w1_task_regime_one_hot_v1",
        code_version=fit_out.code_version,
        created_at_utc="2026-04-24T18:00:00Z",
        trained_by_runner=TRAINED_BY_W1_GOVERNED_META_HEAD_V1,
        training_run_id=fit_out.training_run_id,
    )
    fold_plan = derive_w1_fold_plan(sorted(pool_out.tasks, key=lambda t: (t.t0, t.task_id)), w1_cfg)
    trained = ReptileTrainedMetaAllocatorAdapter(
        ck_path,
        expected_signal_set_version=str(pool_cfg.signal_set_version),
        expected_training_task_pool_hash=pool_out.task_pool_hash,
        expected_training_data_fingerprint=data_fp,
        expected_training_splits_fingerprint=splits_fp,
    )
    assert trained.governed_checkpoint_lineage_verified is True
    ch_surface = build_reptile_w1_challenger_surface(
        task_pool=pool_out.tasks,
        fold_plan=fold_plan,
        trained_state=trained,
        task_pool_hash=pool_out.task_pool_hash,
        data_fingerprint=data_fp,
        splits_fingerprint=splits_fp,
        cost_assumptions_fingerprint=cost_fp,
        signal_set_version=str(pool_cfg.signal_set_version),
        created_at_utc="2026-04-24T17:00:00Z",
        source="reptile_learned_checkpoint_w1.v1",
        model_family="reptile_meta_allocator_checkpoint",
    )
    assert ch_surface.model_state_hash == trained.model_state_hash
    assert ch_surface.governed_checkpoint_lineage_verified is True
    assert w1_challenger_surface_closure_eligible(ch_surface) is True
    return ch_surface, trained


@pytest.mark.determinism("d1")
@pytest.mark.integration
def test_w1_runner_absent_challenger_still_insufficient(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _require_xgboost()
    _ = deterministic_seed
    pool_out, pool_cfg, _, _, _, _ = _pool_and_fingerprints()
    ds = {
        "pit_compliant": True,
        "knowledge_time_column": "knowledge_time",
        "content_hash": "a",
        "content_hash_expected": "a",
    }
    out_dir = tmp_path / "w1_abs"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = run_real_w1_baseline_evidence(
        output_dir=out_dir,
        pool_cfg=pool_cfg,
        dataset_manifest=ds,
        seed=1,
        timestamp_utc="2026-04-24T14:00:00Z",
    )
    assert res.challenger_aggregate is None
    assert res.evidence_source is not None
    assert res.evidence_source.challenger_source == "absent"
    mv = json.loads((out_dir / "meta_validity_report.json").read_text(encoding="utf-8"))
    assert mv["baseline_comparison"]["net_result"] == "INSUFFICIENT_REAL_DATA"


@pytest.mark.determinism("d1")
@pytest.mark.integration
def test_w1_protocol_plumbing_challenger_insufficient_even_with_governance(
    tmp_path: Path, deterministic_seed: int
) -> None:
    """Fold-safe protocol plumbing must not satisfy W1 gate closure or win/loss taxonomy."""
    _require_xgboost()
    _ = deterministic_seed
    pool_out, pool_cfg, w1_cfg, cost_fp, splits_fp, data_fp = _pool_and_fingerprints()
    fold_plan = derive_w1_fold_plan(sorted(pool_out.tasks, key=lambda t: (t.t0, t.task_id)), w1_cfg)
    ch_surface = build_reptile_w1_challenger_surface(
        task_pool=pool_out.tasks,
        fold_plan=fold_plan,
        trained_state=W1FoldSafeSupportOnlyMetaAllocator(state_fingerprint=pool_out.task_pool_hash),
        task_pool_hash=pool_out.task_pool_hash,
        data_fingerprint=data_fp,
        splits_fingerprint=splits_fp,
        cost_assumptions_fingerprint=cost_fp,
        signal_set_version=str(pool_cfg.signal_set_version),
        created_at_utc="2026-04-24T15:00:00Z",
    )
    ds = {
        "pit_compliant": True,
        "knowledge_time_column": "knowledge_time",
        "content_hash": "a",
        "content_hash_expected": "a",
    }
    out_dir = tmp_path / "w1_ch"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = run_real_w1_baseline_evidence(
        output_dir=out_dir,
        pool_cfg=pool_cfg,
        dataset_manifest=ds,
        seed=2,
        timestamp_utc="2026-04-24T15:00:00Z",
        w1_cfg=w1_cfg,
        challenger_surface=ch_surface,
        pit_heterogeneity_governance_acknowledged=True,
    )
    assert res.challenger_aggregate is not None
    assert res.challenger_fold_results is not None
    assert res.evidence_source is not None
    assert res.evidence_source.challenger_source != "absent"
    assert res.evidence_source.challenger_source != "hash_derived_proxy"
    assert res.evidence_source.real_market_data_evidence is True
    assert res.evidence_source.w1_gate_closure_eligible is False
    assert ch_surface.model_state_hash is None
    mv = json.loads((out_dir / "meta_validity_report.json").read_text(encoding="utf-8"))
    assert mv["baseline_comparison"]["net_result"] == "INSUFFICIENT_REAL_DATA"
    assert "model_state_hash" not in mv["baseline_comparison"].get("challenger_surface", {})
    assert (out_dir / "w1_challenger_surface.json").is_file()
    rows = w1_meta_tasks_as_manifest_inputs(pool_out.tasks)
    assert derive_phase2_task_pool_identity_hash(rows) == pool_out.task_pool_hash
    tm = json.loads((out_dir / "task_manifest.json").read_text(encoding="utf-8"))
    ex = json.loads((out_dir / "execution_assumptions.json").read_text(encoding="utf-8"))
    validate_phase2_artifact_triple(task_doc=tm, meta_doc=mv, exec_doc=ex)


@pytest.mark.determinism("d1")
@pytest.mark.integration
def test_w1_learned_checkpoint_challenger_decision_grade_non_overlapping_task_splits(
    tmp_path: Path,
    deterministic_seed: int,
) -> None:
    """Trainer-emitted governed checkpoint (linear meta-head + support economic targets).

    **PASS**: lineage validators, curriculum gates, fingerprint parity, and governed real pool
    economics from task-specific support/query windows.

    **PASS**: decision-grade closure admissibility from task-level fold planning:
    at least 20 unique train/eval tasks per fold, zero train/eval overlap, and
    non-degenerate task-level target variation.
    """
    _require_xgboost()
    _ = deterministic_seed
    pool_out, pool_cfg, w1_cfg, cost_fp, splits_fp, data_fp = _pool_and_fingerprints()
    ch_surface, trained = _learned_challenger_surface_for_spy_pool(
        tmp_path,
        pool_out=pool_out,
        pool_cfg=pool_cfg,
        w1_cfg=w1_cfg,
        cost_fp=cost_fp,
        splits_fp=splits_fp,
        data_fp=data_fp,
        deterministic_seed=deterministic_seed,
    )
    ds = {
        "pit_compliant": True,
        "knowledge_time_column": "knowledge_time",
        "content_hash": "a",
        "content_hash_expected": "a",
    }
    out_dir = tmp_path / "w1_learned"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = run_real_w1_baseline_evidence(
        output_dir=out_dir,
        pool_cfg=pool_cfg,
        dataset_manifest=ds,
        seed=4,
        timestamp_utc="2026-04-24T17:00:00Z",
        w1_cfg=w1_cfg,
        challenger_surface=ch_surface,
        pit_heterogeneity_governance_acknowledged=True,
    )
    assert res.evidence_source is not None
    assert res.evidence_source.w1_gate_closure_eligible is True
    assert res.evidence_sanity_report is not None
    assert res.evidence_sanity_report.get("w1_closure_suppressed_by_evidence_sanity_mode") is False
    assert res.evidence_sanity_report.get("w1_evidence_sanity_closure_ok") is True
    reasons = set(res.evidence_sanity_report.get("failure_reasons", ()))
    assert "TRAIN_EVAL_TASK_ID_OVERLAP" not in reasons
    assert "INSUFFICIENT_UNIQUE_EVAL_TASKS" not in reasons
    assert "INSUFFICIENT_UNIQUE_TRAIN_TASKS" not in reasons
    assert res.fold_results
    for fr in res.fold_results:
        assert int(fr.n_unique_tasks_scored) >= int(w1_cfg.min_unique_eval_tasks_per_fold)
    mv = json.loads((out_dir / "meta_validity_report.json").read_text(encoding="utf-8"))
    cs = mv["baseline_comparison"]["challenger_surface"]
    assert cs["model_state_hash"] == trained.model_state_hash
    assert cs["governed_checkpoint_lineage_verified"] is True
    assert mv["baseline_comparison"]["net_result"] != "INSUFFICIENT_REAL_DATA"
    tm = json.loads((out_dir / "task_manifest.json").read_text(encoding="utf-8"))
    ex = json.loads((out_dir / "execution_assumptions.json").read_text(encoding="utf-8"))
    validate_phase2_artifact_triple(task_doc=tm, meta_doc=mv, exec_doc=ex)


@pytest.mark.determinism("d1")
@pytest.mark.integration
def test_w1_evidence_sanity_constant_targets_block_closure_and_taxonomy(
    tmp_path: Path,
    deterministic_seed: int,
) -> None:
    """Degenerate pool-level query/support targets must fail pre-closure sanity (ADR-005)."""
    _require_xgboost()
    _ = deterministic_seed
    pool_out, pool_cfg, w1_cfg, cost_fp, splits_fp, data_fp = _pool_and_fingerprints()
    ch_surface, _trained = _learned_challenger_surface_for_spy_pool(
        tmp_path,
        pool_out=pool_out,
        pool_cfg=pool_cfg,
        w1_cfg=w1_cfg,
        cost_fp=cost_fp,
        splits_fp=splits_fp,
        data_fp=data_fp,
        deterministic_seed=deterministic_seed,
    )
    tasks_sorted = sorted(pool_out.tasks, key=lambda t: (t.t0, t.task_id))
    degenerate_qt = {str(t.task_id): 0.0 for t in tasks_sorted}
    degenerate_st = {str(t.task_id): 0.0 for t in tasks_sorted}
    ev = W1EvidenceSource(
        market_data_source="governed_historical_dataview",
        label_source="governed_pit_regime_label_surface",
        target_source="query_realized_net_utility",
        challenger_source=ch_surface.source,
        real_market_data_evidence=True,
        w1_gate_closure_eligible=False,
        pit_heterogeneity_governance_acknowledged=True,
    )
    fp = W1ComparisonFingerprints(
        data_fingerprint=data_fp,
        splits_fingerprint=splits_fp,
        cost_assumptions_fingerprint=cost_fp,
    )
    hetero = bool(pool_out.diagnostics.get("heterogeneous_task_pits_supported", False))
    pit_pol = str(pool_out.diagnostics.get("pit_boundary_policy", ""))
    res = run_w1_baseline_evidence(
        w1_cfg,
        pool_out.tasks,
        evidence_lane="governed_real",
        incumbent=XGBoostIncumbentBaseline(),
        task_query_targets=degenerate_qt,
        task_support_targets=degenerate_st,
        evidence_source=ev,
        comparison_fingerprints=fp,
        pit_boundary_policy=pit_pol,
        heterogeneous_task_pits_supported=hetero,
        challenger_surface=ch_surface,
        task_pool_hash=pool_out.task_pool_hash,
    )
    assert res.evidence_source is not None
    assert res.evidence_source.w1_gate_closure_eligible is False
    assert res.evidence_sanity_report is not None
    assert res.evidence_sanity_report.get("w1_evidence_sanity_closure_ok") is False
    assert res.evidence_sanity_report.get("w1_gate_closure_eligible") is False
    cmp = build_baseline_comparison(res, "challenger-degenerate-taxonomy", pool_out.task_pool_hash)
    assert cmp["net_result"] == "INSUFFICIENT_REAL_DATA"


@pytest.mark.determinism("d1")
@pytest.mark.integration
def test_w1_real_challenger_without_governance_stays_insufficient_evidence(
    tmp_path: Path,
    deterministic_seed: int,
) -> None:
    """Challenger + aggregate without closure eligibility must not emit win/loss taxonomy."""
    _require_xgboost()
    _ = deterministic_seed
    pool_out, pool_cfg, w1_cfg, cost_fp, splits_fp, data_fp = _pool_and_fingerprints()
    fold_plan = derive_w1_fold_plan(sorted(pool_out.tasks, key=lambda t: (t.t0, t.task_id)), w1_cfg)
    ch_surface = build_reptile_w1_challenger_surface(
        task_pool=pool_out.tasks,
        fold_plan=fold_plan,
        trained_state=W1FoldSafeSupportOnlyMetaAllocator(state_fingerprint=pool_out.task_pool_hash),
        task_pool_hash=pool_out.task_pool_hash,
        data_fingerprint=data_fp,
        splits_fingerprint=splits_fp,
        cost_assumptions_fingerprint=cost_fp,
        signal_set_version=str(pool_cfg.signal_set_version),
        created_at_utc="2026-04-24T16:00:00Z",
    )
    ds = {
        "pit_compliant": True,
        "knowledge_time_column": "knowledge_time",
        "content_hash": "a",
        "content_hash_expected": "a",
    }
    out_dir = tmp_path / "w1_nogov"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = run_real_w1_baseline_evidence(
        output_dir=out_dir,
        pool_cfg=pool_cfg,
        dataset_manifest=ds,
        seed=3,
        timestamp_utc="2026-04-24T16:00:00Z",
        w1_cfg=w1_cfg,
        challenger_surface=ch_surface,
        pit_heterogeneity_governance_acknowledged=False,
    )
    assert res.challenger_aggregate is not None
    assert res.evidence_source is not None
    assert res.evidence_source.w1_gate_closure_eligible is False
    assert (out_dir / "w1_challenger_surface.json").is_file()
    mv = json.loads((out_dir / "meta_validity_report.json").read_text(encoding="utf-8"))
    assert mv["baseline_comparison"]["net_result"] == "INSUFFICIENT_REAL_DATA"


@pytest.mark.determinism("d1")
@pytest.mark.integration
def test_w1_challenger_fingerprint_mismatch_fails(tmp_path: Path, deterministic_seed: int) -> None:
    _require_xgboost()
    _ = deterministic_seed
    pool_out, pool_cfg, w1_cfg, cost_fp, splits_fp, data_fp = _pool_and_fingerprints()
    fold_plan = derive_w1_fold_plan(sorted(pool_out.tasks, key=lambda t: (t.t0, t.task_id)), w1_cfg)
    ch_surface = build_reptile_w1_challenger_surface(
        task_pool=pool_out.tasks,
        fold_plan=fold_plan,
        trained_state=W1FoldSafeSupportOnlyMetaAllocator(state_fingerprint=pool_out.task_pool_hash),
        task_pool_hash=pool_out.task_pool_hash,
        data_fingerprint=data_fp,
        splits_fingerprint=splits_fp,
        cost_assumptions_fingerprint=cost_fp,
        signal_set_version=str(pool_cfg.signal_set_version),
        created_at_utc="2026-04-24T15:00:00Z",
    )
    bad_fp = W1ComparisonFingerprints(
        data_fingerprint="sha256:" + "0" * 64,
        splits_fingerprint=splits_fp,
        cost_assumptions_fingerprint=cost_fp,
    )
    ev = W1EvidenceSource(
        market_data_source="governed_historical_dataview",
        label_source="governed_pit_regime_label_surface",
        target_source="query_realized_net_utility",
        challenger_source=ch_surface.source,
        real_market_data_evidence=True,
        w1_gate_closure_eligible=False,
    )
    with pytest.raises(W1BaselineEvidenceError, match="data_fingerprint"):
        run_w1_baseline_evidence(
            w1_cfg,
            pool_out.tasks,
            evidence_lane="governed_real",
            incumbent=XGBoostIncumbentBaseline(),
            task_query_targets=dict(pool_out.task_query_targets_net),
            task_support_targets=w1_support_mean_log_return_net_from_pool(pool_out),
            evidence_source=ev,
            comparison_fingerprints=bad_fp,
            challenger_surface=ch_surface,
            task_pool_hash=pool_out.task_pool_hash,
        )


@pytest.mark.determinism("d1")
@pytest.mark.integration
def test_w1_challenger_leakage_flag_fails_validate(tmp_path: Path, deterministic_seed: int) -> None:
    _require_xgboost()
    _ = deterministic_seed
    pool_out, _, w1_cfg, cost_fp, splits_fp, data_fp = _pool_and_fingerprints()
    fold_plan = derive_w1_fold_plan(sorted(pool_out.tasks, key=lambda t: (t.t0, t.task_id)), w1_cfg)
    good = build_reptile_w1_challenger_surface(
        task_pool=pool_out.tasks,
        fold_plan=fold_plan,
        trained_state=W1FoldSafeSupportOnlyMetaAllocator(state_fingerprint=pool_out.task_pool_hash),
        task_pool_hash=pool_out.task_pool_hash,
        data_fingerprint=data_fp,
        splits_fingerprint=splits_fp,
        cost_assumptions_fingerprint=cost_fp,
        signal_set_version="1",
        created_at_utc="2026-04-24T15:00:00Z",
    )
    bad = W1ChallengerSurface(
        schema_version=W1_CHALLENGER_SURFACE_SCHEMA_VERSION,
        source=good.source,
        model_family=good.model_family,
        predictions=good.predictions,
        task_pool_hash=good.task_pool_hash,
        data_fingerprint=good.data_fingerprint,
        splits_fingerprint=good.splits_fingerprint,
        cost_assumptions_fingerprint=good.cost_assumptions_fingerprint,
        signal_set_version=good.signal_set_version,
        created_at_utc=good.created_at_utc,
        leakage_policy=good.leakage_policy,
        uses_query_labels=True,
        uses_xgboost_outputs=False,
        uses_post_query_metrics=False,
    )
    ev = W1EvidenceSource(
        market_data_source="governed_historical_dataview",
        label_source="governed_pit_regime_label_surface",
        target_source="query_realized_net_utility",
        challenger_source=bad.source,
        real_market_data_evidence=True,
        w1_gate_closure_eligible=False,
    )
    fp = W1ComparisonFingerprints(
        data_fingerprint=data_fp,
        splits_fingerprint=splits_fp,
        cost_assumptions_fingerprint=cost_fp,
    )
    with pytest.raises(W1BaselineEvidenceError, match="leakage"):
        run_w1_baseline_evidence(
            w1_cfg,
            pool_out.tasks,
            evidence_lane="governed_real",
            incumbent=XGBoostIncumbentBaseline(),
            task_query_targets=dict(pool_out.task_query_targets_net),
            task_support_targets=w1_support_mean_log_return_net_from_pool(pool_out),
            evidence_source=ev,
            comparison_fingerprints=fp,
            challenger_surface=bad,
            task_pool_hash=pool_out.task_pool_hash,
        )

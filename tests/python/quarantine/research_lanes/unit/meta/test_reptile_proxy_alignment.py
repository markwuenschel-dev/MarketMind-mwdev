from __future__ import annotations

# ruff: noqa: S101 — pytest uses assert for expectations
import ast
import json
import math
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pysrc.meta.reptile_proxy_alignment import (
    _finalize_aggregate,
    _validate_regime_split_coverage,
    compute_proxy_loss,
    emit_proxy_alignment_report_json,
    load_proxy_alignment_report_json,
    partition_training_and_heldout,
    recompute_content_hash_from_document,
    run_alignment_arm,
    run_proxy_alignment,
)
from pysrc.meta.reptile_proxy_alignment_config import ProxyAlignmentConfig
from pysrc.meta.reptile_proxy_alignment_errors import (
    ArtifactImmutabilityError,
    InsufficientTaskPoolError,
    ProxyAlignmentValidationError,
    ProxyArmDivergenceError,
)
from pysrc.meta.reptile_trainer_benchmark import (
    build_synthetic_benchmark_pool,
    run_default_mlc5_proxy_alignment_evidence,
)
from pysrc.meta.reptile_trainer_config import ReptileTrainerConfig
from pysrc.meta.task import MAX_SIGNALS


def _no_print_or_bare_except_in_mlc5_modules() -> None:
    root = Path(__file__).resolve().parents[4]
    paths = [
        root / "pysrc/meta/reptile_proxy_alignment.py",
        root / "pysrc/meta/reptile_proxy_alignment_config.py",
        root / "pysrc/meta/reptile_proxy_alignment_errors.py",
    ]
    for p in paths:
        tree = ast.parse(p.read_text(encoding="utf-8"))
        src = p.read_text(encoding="utf-8")
        assert "print(" not in src, p.name
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                raise AssertionError(f"bare except in {p.name}")


@pytest.mark.determinism("d1")
@given(st.integers(max_value=0))
def test_n_outer_epochs_non_positive_fails_hypothesis(n: int) -> None:
    with pytest.raises(ProxyAlignmentValidationError):
        ProxyAlignmentConfig(
            n_outer_epochs=int(n),
            challenger_proxy="soft_rank_ic",
            control_proxy="mse_regress_rank",
            trainer_seed=1,
            curriculum_sampler_seed=1,
            heldout_partition_seed=1,
            divergence_window=1,
            outer_step_size=0.01,
            soft_rank_temperature=0.5,
            base_trainer_config=ReptileTrainerConfig(task_failure_abort_threshold=20),
        )


@pytest.mark.determinism("d1")
def test_unsupported_challenger_proxy_string_fails() -> None:
    with pytest.raises(ProxyAlignmentValidationError):
        ProxyAlignmentConfig(
            n_outer_epochs=2,
            challenger_proxy=cast(Any, "not_a_proxy"),
            control_proxy="mse_regress_rank",
            trainer_seed=1,
            curriculum_sampler_seed=1,
            heldout_partition_seed=1,
            divergence_window=1,
            outer_step_size=0.01,
            soft_rank_temperature=0.5,
            base_trainer_config=ReptileTrainerConfig(task_failure_abort_threshold=20),
        )


@pytest.mark.determinism("d1")
def test_equal_configs_hash_identically() -> None:
    a = ProxyAlignmentConfig.default_mlc5_bounded(trainer_seed=42, curriculum_sampler_seed=7)
    b = ProxyAlignmentConfig.default_mlc5_bounded(trainer_seed=42, curriculum_sampler_seed=7)
    assert a.config_hash() == b.config_hash()


@pytest.mark.determinism("d1")
def test_repeated_runs_identical_content_hash(deterministic_seed) -> None:
    _ = deterministic_seed
    cfg = ProxyAlignmentConfig.default_mlc5_bounded(
        trainer_seed=424242,
        curriculum_sampler_seed=999,
        n_outer_epochs=3,
    )
    h1 = run_proxy_alignment(cfg).content_hash["value"]
    h2 = run_proxy_alignment(cfg).content_hash["value"]
    assert h1 == h2


@pytest.mark.determinism("d1")
def test_epoch_zero_deltas_none(deterministic_seed) -> None:
    _ = deterministic_seed
    cfg = ProxyAlignmentConfig.default_mlc5_bounded(
        trainer_seed=424242,
        curriculum_sampler_seed=999,
        n_outer_epochs=2,
    )
    rep = run_proxy_alignment(cfg)
    for arm in (rep.challenger_result, rep.control_result):
        z = arm.epoch_records[0]
        assert z.epoch == 0
        assert z.proxy_loss_delta is None
        assert z.held_out_ic_delta is None


@pytest.mark.determinism("d1")
def test_partition_disjoint_task_ids() -> None:
    pool = build_synthetic_benchmark_pool()
    tr, ho = partition_training_and_heldout(pool, heldout_partition_seed=314159)
    assert not ({t.task_id for t in tr} & {t.task_id for t in ho})


@pytest.mark.determinism("d1")
def test_heldout_ic_only_uses_heldout_pool(deterministic_seed) -> None:
    _ = deterministic_seed
    cfg = ProxyAlignmentConfig.default_mlc5_bounded(
        trainer_seed=424242,
        curriculum_sampler_seed=999,
        n_outer_epochs=1,
    )
    pool = build_synthetic_benchmark_pool()
    hseed = int(cfg.heldout_partition_seed)
    tr, ho = partition_training_and_heldout(pool, heldout_partition_seed=hseed)
    captured: list[tuple[str, ...]] = []

    import pysrc.meta.reptile_proxy_alignment as rpa

    _orig_mean = rpa._mean_heldout_spearman

    def _spy(theta: Any, heldout_tasks: Any, **kwargs: Any) -> float | None:
        captured.append(tuple(t.task_id for t in heldout_tasks))
        return _orig_mean(theta, heldout_tasks, **kwargs)

    with patch.object(rpa, "_mean_heldout_spearman", side_effect=_spy):
        run_alignment_arm(
            cfg,
            "challenger",
            (
                np.random.default_rng(1).standard_normal(MAX_SIGNALS).astype(np.float32)
                * np.float32(0.02)
            ),
            tr,
            ho,
            rng=np.random.default_rng(1),
        )
    assert captured
    assert set(captured[0]) == {t.task_id for t in ho}


@pytest.mark.determinism("d1")
def test_pearson_none_when_single_epoch(deterministic_seed) -> None:
    _ = deterministic_seed
    cfg = ProxyAlignmentConfig.default_mlc5_bounded(
        trainer_seed=424242,
        curriculum_sampler_seed=999,
        n_outer_epochs=1,
    )
    rep = run_proxy_alignment(cfg)
    assert rep.challenger_result.pearson_r is None
    assert rep.control_result.pearson_r is None


@pytest.mark.determinism("d1")
def test_divergence_detected_synthetic_bad_trajectory() -> None:
    z = np.zeros(MAX_SIGNALS, dtype=np.float32)
    raw = [(0, 1.0, 0.5)]
    for e in range(1, 6):
        raw.append((e, 1.0 - 0.1 * e, 0.5 - 0.01 * e))
    agg = _finalize_aggregate("challenger", raw, divergence_window=5, final_theta=z)
    assert agg.divergence_detected is True


@pytest.mark.determinism("d1")
def test_divergence_not_detected_synthetic_good_trajectory() -> None:
    z = np.zeros(MAX_SIGNALS, dtype=np.float32)
    raw = [(0, 1.0, 0.5)]
    for e in range(1, 6):
        raw.append((e, 1.0 - 0.1 * e, 0.5 + 0.05 * e))
    agg = _finalize_aggregate("challenger", raw, divergence_window=5, final_theta=z)
    assert agg.divergence_detected is False


@pytest.mark.determinism("d1")
def test_emit_second_write_raises(tmp_path: Path, deterministic_seed) -> None:
    _ = deterministic_seed
    cfg = ProxyAlignmentConfig.default_mlc5_bounded(n_outer_epochs=1)
    rep = run_proxy_alignment(cfg)
    p = tmp_path / "r.json"
    emit_proxy_alignment_report_json(p, rep)
    with pytest.raises(ArtifactImmutabilityError):
        emit_proxy_alignment_report_json(p, rep)


@pytest.mark.determinism("d1")
def test_recomputed_hash_matches_embedded(tmp_path: Path, deterministic_seed) -> None:
    _ = deterministic_seed
    cfg = ProxyAlignmentConfig.default_mlc5_bounded(n_outer_epochs=1)
    rep = run_proxy_alignment(cfg)
    p = tmp_path / "r.json"
    emit_proxy_alignment_report_json(p, rep)
    doc = load_proxy_alignment_report_json(p)
    assert recompute_content_hash_from_document(doc) == doc["content_hash"]["value"]


@pytest.mark.determinism("d1")
def test_content_hash_structured(tmp_path: Path, deterministic_seed) -> None:
    _ = deterministic_seed
    cfg = ProxyAlignmentConfig.default_mlc5_bounded(n_outer_epochs=1)
    rep = run_proxy_alignment(cfg)
    p = tmp_path / "r.json"
    emit_proxy_alignment_report_json(p, rep)
    doc = load_proxy_alignment_report_json(p)
    ch = doc["content_hash"]
    assert set(ch.keys()) == {"algorithm", "canonicalization", "value"}
    assert all(str(ch[k]).strip() for k in ch)


@pytest.mark.determinism("d1")
def test_ast_scan_new_modules() -> None:
    _no_print_or_bare_except_in_mlc5_modules()


@pytest.mark.determinism("d1")
def test_info_logs_start_end(caplog: pytest.LogCaptureFixture, deterministic_seed) -> None:
    _ = deterministic_seed
    cfg = ProxyAlignmentConfig.default_mlc5_bounded(n_outer_epochs=1)
    with caplog.at_level("INFO", logger="pysrc.meta.reptile_proxy_alignment"):
        run_proxy_alignment(cfg)
    infos = [r for r in caplog.records if r.levelname == "INFO"]
    assert len(infos) >= 2


@pytest.mark.determinism("d1")
def test_governance_flags_in_emitted_artifact(tmp_path: Path, deterministic_seed) -> None:
    _ = deterministic_seed
    cfg = ProxyAlignmentConfig.default_mlc5_bounded(n_outer_epochs=1)
    rep = run_proxy_alignment(cfg)
    p = tmp_path / "r.json"
    emit_proxy_alignment_report_json(p, rep)
    doc = load_proxy_alignment_report_json(p)
    assert doc["gate_ii_status"] == "DEFERRED"
    assert doc["promotion_evidence"] is False


@pytest.mark.determinism("d1")
def test_challenger_and_control_final_theta_differ(deterministic_seed) -> None:
    _ = deterministic_seed
    rep = run_default_mlc5_proxy_alignment_evidence()
    a = np.asarray(rep.challenger_result.final_theta_meta, dtype=np.float64)
    b = np.asarray(rep.control_result.final_theta_meta, dtype=np.float64)
    assert not np.allclose(a, b)


@pytest.mark.determinism("d1")
def test_insufficient_task_pool_error() -> None:
    from pysrc.meta_learning.regime_vocabulary import REGIME_CLASS_ORDER

    train_b = dict.fromkeys(REGIME_CLASS_ORDER, 0)
    held_b = dict.fromkeys(REGIME_CLASS_ORDER, 0)
    train_b["bull"] = 10
    held_b["bull"] = 0
    with pytest.raises(InsufficientTaskPoolError):
        _validate_regime_split_coverage(train_b, held_b)


@pytest.mark.determinism("d1")
def test_compute_proxy_loss_bad_kind() -> None:
    theta = np.zeros(MAX_SIGNALS, dtype=np.float32)
    t = build_synthetic_benchmark_pool()[0]
    with pytest.raises(ProxyAlignmentValidationError):
        compute_proxy_loss("not_a_kind", theta, t)


@pytest.mark.determinism("d1")
def test_compute_proxy_loss_dispatch_kinds(deterministic_seed) -> None:
    _ = deterministic_seed
    t = build_synthetic_benchmark_pool()[0]
    cfg = ReptileTrainerConfig(task_failure_abort_threshold=20, K=2)
    from pysrc.meta.reptile_proxy_alignment import _inner_adapt_mlc5

    th = np.random.default_rng(3).standard_normal(MAX_SIGNALS).astype(np.float32) * np.float32(0.02)
    for i, kind in enumerate(("mse_regress_rank", "soft_rank_ic", "pairwise_ranking")):
        rng_i = np.random.default_rng(7 + i)
        ttp, _pl, _ic = _inner_adapt_mlc5(
            t,
            th,
            config=cfg,
            rng=rng_i,
            inner_proxy=kind,
            soft_rank_temperature=0.4,
        )
        v = compute_proxy_loss(kind, ttp, t, soft_rank_temperature=0.4)
        assert math.isfinite(v)


@pytest.mark.determinism("d1")
def test_challenger_pairwise_proxy_runs(deterministic_seed) -> None:
    _ = deterministic_seed
    cfg = ProxyAlignmentConfig.default_mlc5_bounded(
        trainer_seed=424242,
        curriculum_sampler_seed=999,
        n_outer_epochs=2,
        challenger_proxy="pairwise_ranking",
    )
    rep = run_proxy_alignment(cfg)
    assert rep.challenger_result.epoch_records


@pytest.mark.determinism("d1")
def test_config_wrong_control_proxy() -> None:
    base = ReptileTrainerConfig(task_failure_abort_threshold=20)
    with pytest.raises(ProxyAlignmentValidationError):
        ProxyAlignmentConfig(
            n_outer_epochs=2,
            challenger_proxy="soft_rank_ic",
            control_proxy=cast(Any, "wrong"),
            trainer_seed=1,
            curriculum_sampler_seed=1,
            heldout_partition_seed=1,
            divergence_window=1,
            outer_step_size=0.01,
            soft_rank_temperature=0.5,
            base_trainer_config=base,
        )


@pytest.mark.determinism("d1")
def test_config_validation_outer_step_and_tau() -> None:
    base = ReptileTrainerConfig(task_failure_abort_threshold=20)
    with pytest.raises(ProxyAlignmentValidationError):
        ProxyAlignmentConfig(
            n_outer_epochs=2,
            challenger_proxy="soft_rank_ic",
            control_proxy="mse_regress_rank",
            trainer_seed=1,
            curriculum_sampler_seed=1,
            heldout_partition_seed=1,
            divergence_window=1,
            outer_step_size=-1.0,
            soft_rank_temperature=0.5,
            base_trainer_config=base,
        )
    with pytest.raises(ProxyAlignmentValidationError):
        ProxyAlignmentConfig(
            n_outer_epochs=2,
            challenger_proxy="soft_rank_ic",
            control_proxy="mse_regress_rank",
            trainer_seed=1,
            curriculum_sampler_seed=1,
            heldout_partition_seed=1,
            divergence_window=1,
            outer_step_size=0.01,
            soft_rank_temperature=0.0,
            base_trainer_config=base,
        )
    with pytest.raises(ProxyAlignmentValidationError):
        ProxyAlignmentConfig(
            n_outer_epochs=2,
            challenger_proxy="soft_rank_ic",
            control_proxy="mse_regress_rank",
            trainer_seed=1,
            curriculum_sampler_seed=1,
            heldout_partition_seed=1,
            divergence_window=0,
            outer_step_size=0.01,
            soft_rank_temperature=0.5,
            base_trainer_config=base,
        )


@pytest.mark.determinism("d1")
def test_run_proxy_alignment_overlap_guard(deterministic_seed) -> None:
    _ = deterministic_seed
    import pysrc.meta.reptile_proxy_alignment as rpa

    t = build_synthetic_benchmark_pool()[0]
    with (
        patch.object(
            rpa,
            "partition_training_and_heldout",
            return_value=((t,), (t,)),
        ),
        pytest.raises(ProxyAlignmentValidationError),
    ):
        rpa.run_proxy_alignment(ProxyAlignmentConfig.default_mlc5_bounded(n_outer_epochs=1))


@pytest.mark.determinism("d1")
def test_mean_heldout_nonfinite_returns_none(deterministic_seed) -> None:
    _ = deterministic_seed
    import pysrc.meta.reptile_proxy_alignment as rpa

    def _boom(*_a: Any, **_k: Any) -> Any:
        raise ValueError("inner_nonfinite")

    with patch.object(rpa, "_inner_adapt_mlc5", side_effect=_boom):
        out = rpa._mean_heldout_spearman(
            np.zeros(MAX_SIGNALS, dtype=np.float32),
            build_synthetic_benchmark_pool()[:1],
            inner_cfg=ReptileTrainerConfig(task_failure_abort_threshold=20),
            inner_proxy="mse_regress_rank",
            soft_rank_temperature=0.5,
            subseeds=(1,),
        )
    assert out is None


@pytest.mark.determinism("d1")
def test_run_alignment_arm_propagates_unexpected_valueerror(deterministic_seed) -> None:
    _ = deterministic_seed
    import pysrc.meta.reptile_proxy_alignment as rpa

    def _bad(*_a: Any, **_k: Any) -> Any:
        raise ValueError("unexpected")

    cfg = ProxyAlignmentConfig.default_mlc5_bounded(
        trainer_seed=424242,
        curriculum_sampler_seed=999,
        n_outer_epochs=1,
    )
    pool = build_synthetic_benchmark_pool()
    hseed = int(cfg.heldout_partition_seed)
    tr, ho = partition_training_and_heldout(pool, heldout_partition_seed=hseed)
    with (
        patch.object(rpa, "_inner_adapt_mlc5", side_effect=_bad),
        pytest.raises(ValueError, match="unexpected"),
    ):
        rpa.run_alignment_arm(
            cfg,
            "challenger",
            np.zeros(MAX_SIGNALS, dtype=np.float32),
            tr,
            ho,
            rng=np.random.default_rng(1),
        )


@pytest.mark.determinism("d1")
def test_neg_soft_spearman_loss_finite_extremes() -> None:
    import pysrc.meta.reptile_proxy_alignment as rpa

    pred1 = np.array([80.0, -80.0], dtype=np.float64)
    pred2 = np.array([-80.0, 80.0], dtype=np.float64)
    y = np.array([1.0, 0.0], dtype=np.float32)
    v1 = rpa._neg_soft_spearman_ic_loss(pred1, y, tau=0.01)
    v2 = rpa._neg_soft_spearman_ic_loss(pred2, y, tau=0.01)
    assert math.isfinite(v1)
    assert math.isfinite(v2)


@pytest.mark.determinism("d1")
def test_neg_soft_spearman_ic_grad_matches_finite_difference(deterministic_seed) -> None:
    _ = deterministic_seed
    import pysrc.meta.reptile_proxy_alignment as rpa

    rng = np.random.default_rng(42)
    n, d_feat = 5, 4
    x_mat = rng.standard_normal((n, d_feat)).astype(np.float32)
    y_vec = rng.standard_normal(n).astype(np.float32)
    w_vec = rng.standard_normal(d_feat).astype(np.float64)
    tau = 0.35
    ag = rpa._neg_soft_spearman_ic_loss_grad(x_mat, y_vec, w_vec, tau)
    eps = 1e-5
    num = np.zeros_like(w_vec)
    for k in range(d_feat):
        wp = w_vec.copy()
        wm = w_vec.copy()
        wp[k] += eps
        wm[k] -= eps
        lp = rpa._neg_soft_spearman_ic_loss((x_mat.astype(np.float64) @ wp).reshape(-1), y_vec, tau)
        lm = rpa._neg_soft_spearman_ic_loss((x_mat.astype(np.float64) @ wm).reshape(-1), y_vec, tau)
        num[k] = (lp - lm) / (2.0 * eps)
    assert np.allclose(ag, num, rtol=5e-3, atol=5e-3)


@pytest.mark.determinism("d1")
def test_recompute_hash_type_guard() -> None:
    with pytest.raises(TypeError):
        recompute_content_hash_from_document(cast(Any, []))


@pytest.mark.determinism("d1")
def test_proxy_arm_divergence_error_str() -> None:
    e = ProxyArmDivergenceError("x", details={"a": 1})
    assert "x" in str(e)
    assert e.details["a"] == 1


@pytest.mark.determinism("d1")
def test_run_alignment_arm_inner_nonfinite_epoch_then_recover(deterministic_seed) -> None:
    _ = deterministic_seed
    import pysrc.meta.reptile_proxy_alignment as rpa

    orig = rpa._inner_adapt_mlc5
    state = {"n": 0}

    def _wrap(*a: Any, **k: Any) -> Any:
        state["n"] += 1
        if state["n"] == 1:
            raise ValueError("inner_nonfinite")
        return orig(*a, **k)

    cfg = ProxyAlignmentConfig.default_mlc5_bounded(
        trainer_seed=424242,
        curriculum_sampler_seed=999,
        n_outer_epochs=3,
    )
    pool = build_synthetic_benchmark_pool()
    hseed = int(cfg.heldout_partition_seed)
    tr, ho = partition_training_and_heldout(pool, heldout_partition_seed=hseed)
    th = np.random.default_rng(2).standard_normal(MAX_SIGNALS).astype(np.float32) * np.float32(0.02)
    with patch.object(rpa, "_inner_adapt_mlc5", side_effect=_wrap):
        agg = rpa.run_alignment_arm(cfg, "control", th, tr, ho, rng=np.random.default_rng(1))
    assert any(rec.proxy_loss is None for rec in agg.epoch_records)
    assert any(rec.proxy_loss is not None for rec in agg.epoch_records)


@pytest.mark.determinism("d1")
def test_example_artifact_schema() -> None:
    root = Path(__file__).resolve().parents[4]
    ex = root / "artifacts" / "phase_ii" / "mlc5" / "reptile_proxy_alignment_example.json"
    assert ex.exists(), "committed example artifact required"
    doc = json.loads(ex.read_text(encoding="utf-8"))
    assert doc.get("is_example") is True
    assert doc.get("promotion_evidence") is False
    assert doc["gate_ii_status"] == "DEFERRED"
    ch = doc["content_hash"]
    assert ch["algorithm"]
    assert ch["canonicalization"]
    assert ch["value"]

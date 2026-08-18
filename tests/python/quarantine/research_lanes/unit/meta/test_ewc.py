from __future__ import annotations

# ruff: noqa: S101
import ast
import json
from pathlib import Path

import numpy as np
import pytest

from pysrc.meta.ewc import (
    _assert_disjoint,
    _partition_pool,
    _validate_partition_coverage,
    apply_ewc_correction,
    compute_diagonal_fisher,
    emit_ewc_forgetting_report_json,
    load_ewc_forgetting_report_json,
    recompute_content_hash_from_document,
    run_ewc_sweep,
)
from pysrc.meta.ewc_config import EWCSweepConfig
from pysrc.meta.ewc_errors import (
    ArtifactImmutabilityError,
    EWCValidationError,
    InsufficientTaskPoolError,
)
from pysrc.meta.reptile_trainer_benchmark import build_synthetic_benchmark_pool


def _scan_new_modules() -> None:
    root = Path(__file__).resolve().parents[4]
    files = [
        root / "pysrc/meta/ewc.py",
        root / "pysrc/meta/ewc_config.py",
        root / "pysrc/meta/ewc_errors.py",
    ]
    for p in files:
        text = p.read_text(encoding="utf-8")
        assert "print(" not in text, p.name
        tree = ast.parse(text)
        for n in ast.walk(tree):
            if isinstance(n, ast.ExceptHandler) and n.type is None:
                raise AssertionError(f"bare except in {p}")


@pytest.mark.determinism("d1")
def test_lambda_without_zero_fails() -> None:
    with pytest.raises(EWCValidationError):
        EWCSweepConfig(
            lambda_ewc_values=(1.0, 10.0),
            n_pretrain_steps=1,
            n_update_steps=1,
            anchor_set_size_per_bucket=1,
            trainer_seed=1,
            historical_partition_seed=1,
            outer_step_size=0.01,
            base_trainer_config=EWCSweepConfig.default_mlc6_bounded().base_trainer_config,
        )


@pytest.mark.determinism("d1")
def test_duplicate_lambda_fails() -> None:
    with pytest.raises(EWCValidationError):
        EWCSweepConfig(
            lambda_ewc_values=(0.0, 1.0, 1.0),
            n_pretrain_steps=1,
            n_update_steps=1,
            anchor_set_size_per_bucket=1,
            trainer_seed=1,
            historical_partition_seed=1,
            outer_step_size=0.01,
            base_trainer_config=EWCSweepConfig.default_mlc6_bounded().base_trainer_config,
        )


@pytest.mark.determinism("d1")
def test_unsorted_lambda_fails() -> None:
    with pytest.raises(EWCValidationError):
        EWCSweepConfig(
            lambda_ewc_values=(0.0, 10.0, 1.0),
            n_pretrain_steps=1,
            n_update_steps=1,
            anchor_set_size_per_bucket=1,
            trainer_seed=1,
            historical_partition_seed=1,
            outer_step_size=0.01,
            base_trainer_config=EWCSweepConfig.default_mlc6_bounded().base_trainer_config,
        )


@pytest.mark.determinism("d1")
def test_equal_configs_hash_identically() -> None:
    a = EWCSweepConfig.default_mlc6_bounded(trainer_seed=42, historical_partition_seed=7)
    b = EWCSweepConfig.default_mlc6_bounded(trainer_seed=42, historical_partition_seed=7)
    assert a.config_hash() == b.config_hash()


@pytest.mark.determinism("d1")
def test_runs_identical_content_hash(deterministic_seed) -> None:
    _ = deterministic_seed
    cfg = EWCSweepConfig.default_mlc6_bounded()
    h1 = run_ewc_sweep(cfg).content_hash["value"]
    h2 = run_ewc_sweep(cfg).content_hash["value"]
    assert h1 == h2


@pytest.mark.determinism("d1")
def test_partitions_are_disjoint() -> None:
    pool = build_synthetic_benchmark_pool()
    historical, new_batches, heldout, fresh = _partition_pool(pool, seed=314159)
    _assert_disjoint(historical, new_batches, heldout, fresh)


@pytest.mark.determinism("d1")
def test_insufficient_pool_raises() -> None:
    tiny = build_synthetic_benchmark_pool()[:3]
    with pytest.raises(InsufficientTaskPoolError):
        _validate_partition_coverage(tiny, (), (), ())


@pytest.mark.determinism("d1")
def test_apply_ewc_lambda_zero_no_change() -> None:
    th = np.array([1.0, 2.0], dtype=np.float64)
    out = apply_ewc_correction(th, np.array([0.0, 0.0]), np.array([1.0, 1.0]), 0.0)
    assert np.array_equal(out, th)
    assert out is not th


@pytest.mark.determinism("d1")
def test_apply_ewc_reduces_fisher_weighted_distance() -> None:
    th = np.array([2.0, -1.0], dtype=np.float64)
    anchor = np.array([0.0, 0.0], dtype=np.float64)
    f = np.array([3.0, 1.0], dtype=np.float64)
    before = np.sum(f * ((th - anchor) ** 2))
    out = apply_ewc_correction(th, anchor, f, 0.1)
    after = np.sum(f * ((out - anchor) ** 2))
    assert after < before


@pytest.mark.determinism("d1")
def test_apply_ewc_does_not_mutate_input() -> None:
    th = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    keep = th.copy()
    _ = apply_ewc_correction(th, np.zeros(3), np.ones(3), 0.2)
    assert np.array_equal(th, keep)


@pytest.mark.determinism("d1")
def test_compute_fisher_shape_nonnegative() -> None:
    pool = build_synthetic_benchmark_pool()[:5]
    cfg = EWCSweepConfig.default_mlc6_bounded()
    f = compute_diagonal_fisher(
        np.zeros(64, dtype=np.float64), pool, cfg.base_trainer_config, np.random.default_rng(1)
    )
    assert f.shape == (64,)
    assert np.all(f >= 0.0)


@pytest.mark.determinism("d1")
def test_from_scratch_schema_fields(deterministic_seed) -> None:
    _ = deterministic_seed
    rep = run_ewc_sweep(EWCSweepConfig.default_mlc6_bounded())
    fs = rep.from_scratch_result
    assert fs.arm_kind == "from_scratch"
    assert fs.lambda_ewc is None
    for row in fs.update_records:
        assert row.theta_l2_drift_from_anchor is None


@pytest.mark.determinism("d1")
def test_stronger_lambda_lower_drift_and_final_theta_differs(deterministic_seed) -> None:
    _ = deterministic_seed
    rep = run_ewc_sweep(EWCSweepConfig.default_mlc6_bounded())
    lam0 = next(x for x in rep.warm_start_arm_results if x.lambda_ewc == 0.0)
    lam100 = next(x for x in rep.warm_start_arm_results if x.lambda_ewc == 100.0)
    assert (lam100.final_theta_l2_drift or 0.0) < (lam0.final_theta_l2_drift or 0.0)
    a = np.asarray(lam0.final_theta_meta, dtype=np.float64)
    b = np.asarray(lam100.final_theta_meta, dtype=np.float64)
    assert not np.allclose(a, b)


@pytest.mark.determinism("d1")
def test_step0_delta_none_all_arms(deterministic_seed) -> None:
    _ = deterministic_seed
    rep = run_ewc_sweep(EWCSweepConfig.default_mlc6_bounded())
    assert rep.from_scratch_result.update_records[0].heldout_ic_delta is None
    for arm in rep.warm_start_arm_results:
        assert arm.update_records[0].heldout_ic_delta is None


@pytest.mark.determinism("d1")
def test_emit_immutability_and_hash(tmp_path: Path, deterministic_seed) -> None:
    _ = deterministic_seed
    rep = run_ewc_sweep(EWCSweepConfig.default_mlc6_bounded())
    p = tmp_path / "ewc.json"
    emit_ewc_forgetting_report_json(p, rep)
    with pytest.raises(ArtifactImmutabilityError):
        emit_ewc_forgetting_report_json(p, rep)
    doc = load_ewc_forgetting_report_json(p)
    assert recompute_content_hash_from_document(doc) == doc["content_hash"]["value"]
    assert set(doc["content_hash"].keys()) == {"algorithm", "canonicalization", "value"}
    assert "from_scratch_result" in doc
    assert "warm_start_arm_results" in doc
    assert doc["gate_ii_status"] == "DEFERRED"
    assert doc["promotion_evidence"] is False


@pytest.mark.determinism("d1")
def test_no_print_no_bare_except() -> None:
    _scan_new_modules()


@pytest.mark.determinism("d1")
def test_example_artifact_present() -> None:
    root = Path(__file__).resolve().parents[4]
    ex = root / "artifacts" / "phase_ii" / "mlc6" / "ewc_forgetting_example.json"
    if ex.exists():
        doc = json.loads(ex.read_text(encoding="utf-8"))
        assert doc.get("is_example") is True

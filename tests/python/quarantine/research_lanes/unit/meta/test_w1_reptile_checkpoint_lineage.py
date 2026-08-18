"""Checkpoint lineage validation for :class:`ReptileTrainedMetaAllocatorAdapter`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tests.python.unit.meta.test_w1_reptile_challenger_bridge import _task

from pysrc.meta.phase2_artifact_contract import canonical_content_hash
from pysrc.meta.w1_baseline_config import W1BaselineConfig
from pysrc.meta.w1_challenger_surface import (
    derive_w1_fold_plan,
    w1_challenger_surface_closure_eligible,
)
from pysrc.meta.w1_reptile_challenger_bridge import (
    W1ChallengerUnavailableError,
    build_reptile_w1_challenger_surface,
)
from pysrc.meta.w1_reptile_trained_meta_allocator_adapter import (
    GOVERNED_W1_CHECKPOINT_TRAINED_BY_RUNNERS,
    W1_LEARNED_CHECKPOINT_SCHEMA_V2,
    ReptileTrainedMetaAllocatorAdapter,
    w1_learned_checkpoint_body_for_content_hash,
)


def _unit_expectations() -> dict[str, str]:
    return {
        "expected_signal_set_version": "1",
        "expected_training_task_pool_hash": "sha256:" + "c" * 64,
        "expected_training_data_fingerprint": "sha256:" + "a" * 64,
        "expected_training_splits_fingerprint": "sha256:" + "d" * 64,
    }


def _fixture_ckpt_path() -> Path:
    return (
        Path(__file__).resolve().parents[3] / "fixtures" / "w1" / "minimal_learned_checkpoint.json"
    )


def _base_payload(*, runner: str) -> dict:
    return {
        "schema_version": W1_LEARNED_CHECKPOINT_SCHEMA_V2,
        "model_state_hash": "sha256:" + "1" * 64,
        "trainer_config_hash": "sha256:" + "2" * 64,
        "training_task_pool_hash": "sha256:" + "c" * 64,
        "training_data_fingerprint": "sha256:" + "a" * 64,
        "training_splits_fingerprint": "sha256:" + "d" * 64,
        "signal_set_version": "1",
        "feature_encoder_contract_version": "w1_task_regime_one_hot_v1",
        "code_version": "test",
        "created_at_utc": "2026-04-24T00:00:00Z",
        "trained_by_runner": runner,
        "training_run_id": "run.sha256:" + "e" * 64,
        "weights": [0.01, 0.02, -0.01, 0.0, 0.0, 0.03, 0.01],
    }


def _finalize_checkpoint(payload_without_cch: dict) -> dict:
    cch = canonical_content_hash(w1_learned_checkpoint_body_for_content_hash(payload_without_cch))
    return {**payload_without_cch, "checkpoint_content_hash": cch}


def _write_ckpt(path: Path, doc: dict) -> None:
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


@pytest.mark.determinism("d1")
def test_minimal_fixture_loads_but_not_governed_lineage(deterministic_seed: int) -> None:
    """Hand-authored fixture is accepted for harness scores only — not governed lineage."""
    _ = deterministic_seed
    a = ReptileTrainedMetaAllocatorAdapter(_fixture_ckpt_path(), **_unit_expectations())
    assert a.governed_checkpoint_lineage_verified is False


@pytest.mark.determinism("d1")
def test_minimal_fixture_surface_not_closure_eligible(deterministic_seed: int) -> None:
    _ = deterministic_seed
    cfg = W1BaselineConfig(n_walk_forward_folds=2, fold_length_bars=3, warmup_bars=2)
    pool = tuple(_task(i) for i in range(5))
    fold_plan = derive_w1_fold_plan(pool, cfg)
    trained = ReptileTrainedMetaAllocatorAdapter(_fixture_ckpt_path(), **_unit_expectations())
    surface = build_reptile_w1_challenger_surface(
        task_pool=pool,
        fold_plan=fold_plan,
        trained_state=trained,
        task_pool_hash="sha256:" + "c" * 64,
        data_fingerprint="sha256:" + "a" * 64,
        splits_fingerprint="sha256:" + "d" * 64,
        cost_assumptions_fingerprint="sha256:" + "e" * 64,
        signal_set_version="1",
        created_at_utc="2026-01-02T00:00:00Z",
        source="reptile_learned_checkpoint_w1.v1",
        model_family="reptile_meta_allocator_checkpoint",
    )
    assert surface.governed_checkpoint_lineage_verified is False
    assert not w1_challenger_surface_closure_eligible(surface)


@pytest.mark.determinism("d1")
def test_missing_training_task_pool_hash_raises(deterministic_seed: int, tmp_path: Path) -> None:
    _ = deterministic_seed
    p = _base_payload(runner=next(iter(GOVERNED_W1_CHECKPOINT_TRAINED_BY_RUNNERS)))
    del p["training_task_pool_hash"]
    _write_ckpt(tmp_path / "x.json", _finalize_checkpoint(p))
    with pytest.raises(W1ChallengerUnavailableError, match="missing required lineage"):
        ReptileTrainedMetaAllocatorAdapter(tmp_path / "x.json", **_unit_expectations())


@pytest.mark.determinism("d1")
def test_missing_training_data_fingerprint_raises(deterministic_seed: int, tmp_path: Path) -> None:
    _ = deterministic_seed
    p = _base_payload(runner=next(iter(GOVERNED_W1_CHECKPOINT_TRAINED_BY_RUNNERS)))
    del p["training_data_fingerprint"]
    _write_ckpt(tmp_path / "x.json", _finalize_checkpoint(p))
    with pytest.raises(W1ChallengerUnavailableError, match="missing required lineage"):
        ReptileTrainedMetaAllocatorAdapter(tmp_path / "x.json", **_unit_expectations())


@pytest.mark.determinism("d1")
def test_signal_set_version_mismatch_raises(deterministic_seed: int, tmp_path: Path) -> None:
    _ = deterministic_seed
    p = _base_payload(runner=next(iter(GOVERNED_W1_CHECKPOINT_TRAINED_BY_RUNNERS)))
    p["signal_set_version"] = "99"
    _write_ckpt(tmp_path / "x.json", _finalize_checkpoint(p))
    with pytest.raises(W1ChallengerUnavailableError, match="signal_set_version"):
        ReptileTrainedMetaAllocatorAdapter(tmp_path / "x.json", **_unit_expectations())


@pytest.mark.determinism("d1")
def test_checkpoint_content_hash_mismatch_raises(deterministic_seed: int, tmp_path: Path) -> None:
    _ = deterministic_seed
    p = _base_payload(runner=next(iter(GOVERNED_W1_CHECKPOINT_TRAINED_BY_RUNNERS)))
    doc = _finalize_checkpoint(p)
    doc["checkpoint_content_hash"] = "sha256:" + "0" * 64
    _write_ckpt(tmp_path / "x.json", doc)
    with pytest.raises(W1ChallengerUnavailableError, match="checkpoint_content_hash"):
        ReptileTrainedMetaAllocatorAdapter(tmp_path / "x.json", **_unit_expectations())


@pytest.mark.determinism("d1")
def test_unapproved_trained_by_runner_not_governed_lineage(
    deterministic_seed: int, tmp_path: Path
) -> None:
    _ = deterministic_seed
    p = _base_payload(runner="unknown.runner.v0")
    _write_ckpt(tmp_path / "x.json", _finalize_checkpoint(p))
    a = ReptileTrainedMetaAllocatorAdapter(tmp_path / "x.json", **_unit_expectations())
    assert a.governed_checkpoint_lineage_verified is False


@pytest.mark.determinism("d1")
def test_governed_checkpoint_produces_closure_eligible_surface(
    deterministic_seed: int, tmp_path: Path
) -> None:
    _ = deterministic_seed
    runner = next(iter(GOVERNED_W1_CHECKPOINT_TRAINED_BY_RUNNERS))
    p = _base_payload(runner=runner)
    _write_ckpt(tmp_path / "gov.json", _finalize_checkpoint(p))
    a = ReptileTrainedMetaAllocatorAdapter(tmp_path / "gov.json", **_unit_expectations())
    assert a.governed_checkpoint_lineage_verified is True
    cfg = W1BaselineConfig(n_walk_forward_folds=2, fold_length_bars=3, warmup_bars=2)
    pool = tuple(_task(i) for i in range(5))
    fold_plan = derive_w1_fold_plan(pool, cfg)
    surface = build_reptile_w1_challenger_surface(
        task_pool=pool,
        fold_plan=fold_plan,
        trained_state=a,
        task_pool_hash="sha256:" + "c" * 64,
        data_fingerprint="sha256:" + "a" * 64,
        splits_fingerprint="sha256:" + "d" * 64,
        cost_assumptions_fingerprint="sha256:" + "e" * 64,
        signal_set_version="1",
        created_at_utc="2026-01-02T00:00:00Z",
        source="reptile_learned_checkpoint_w1.v1",
        model_family="reptile_meta_allocator_checkpoint",
    )
    assert surface.governed_checkpoint_lineage_verified is True
    assert w1_challenger_surface_closure_eligible(surface)


@pytest.mark.determinism("d0")
def test_governed_runner_allow_list_includes_meta_head_trainer(deterministic_seed: int) -> None:
    _ = deterministic_seed
    assert "mm.reptile_trainer.w1_governed_v1" in GOVERNED_W1_CHECKPOINT_TRAINED_BY_RUNNERS
    assert "mm.w1_governed_meta_head_trainer.v1" in GOVERNED_W1_CHECKPOINT_TRAINED_BY_RUNNERS

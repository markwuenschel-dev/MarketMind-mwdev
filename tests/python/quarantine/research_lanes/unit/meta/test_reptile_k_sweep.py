from __future__ import annotations

# ruff: noqa: S101 — pytest uses assert for expectations
import json
from pathlib import Path

import pytest

from pysrc.meta.reptile_k_sweep_config import (
    GOVERNED_K_VALUES,
    KSweepConfig,
    KSweepReport,
)
from pysrc.meta.reptile_k_sweep_errors import ArtifactImmutabilityError, KSweepValidationError
from pysrc.meta.reptile_k_sweep_runner import (
    emit_reptile_k_sweep_report_json,
    recompute_content_hash_from_document,
    run_k_sweep,
    saturation_from_adjacent_means,
)
from pysrc.meta.reptile_trainer_config import ReptileTrainerConfig


def _no_print_calls_in_modules() -> None:
    root = Path(__file__).resolve().parents[4]
    paths = [
        root / "pysrc/meta/reptile_k_sweep_config.py",
        root / "pysrc/meta/reptile_k_sweep_runner.py",
        root / "pysrc/meta/reptile_k_sweep_errors.py",
        root / "pysrc/meta/reptile_trainer_benchmark.py",
    ]
    for p in paths:
        text = p.read_text(encoding="utf-8")
        assert "print(" not in text, p.name


@pytest.mark.determinism("d1")
def test_invalid_k_values_fail_at_construction() -> None:
    with pytest.raises(KSweepValidationError):
        KSweepConfig(
            k_values=(0, 3, 5),
            trainer_seed=1,
            curriculum_sampler_seed=1,
            base_trainer_config=ReptileTrainerConfig(task_failure_abort_threshold=20),
            saturation_epsilon=1e-4,
        )


@pytest.mark.determinism("d1")
def test_k_values_must_be_sorted_unique() -> None:
    with pytest.raises(KSweepValidationError):
        KSweepConfig(
            k_values=(5, 0, 1),
            trainer_seed=1,
            curriculum_sampler_seed=1,
            base_trainer_config=ReptileTrainerConfig(task_failure_abort_threshold=20),
            saturation_epsilon=1e-4,
        )
    with pytest.raises(KSweepValidationError):
        KSweepConfig(
            k_values=(0, 0, 1),
            trainer_seed=1,
            curriculum_sampler_seed=1,
            base_trainer_config=ReptileTrainerConfig(task_failure_abort_threshold=20),
            saturation_epsilon=1e-4,
        )


@pytest.mark.determinism("d1")
def test_equal_configs_hash_identically() -> None:
    a = KSweepConfig.default_mlc4_bounded(trainer_seed=42, curriculum_sampler_seed=7)
    b = KSweepConfig.default_mlc4_bounded(trainer_seed=42, curriculum_sampler_seed=7)
    assert a.k_sweep_config_hash() == b.k_sweep_config_hash()


@pytest.mark.determinism("d1")
def test_repeated_runs_identical_report_hashes(deterministic_seed) -> None:
    _ = deterministic_seed
    cfg = KSweepConfig.default_mlc4_bounded(trainer_seed=424242, curriculum_sampler_seed=999)
    h1 = run_k_sweep(cfg).content_hash["value"]
    h2 = run_k_sweep(cfg).content_hash["value"]
    assert h1 == h2


@pytest.mark.determinism("d1")
def test_k_zero_exact_zero_gain_per_task(deterministic_seed) -> None:
    _ = deterministic_seed
    cfg = KSweepConfig(
        k_values=(0,),
        trainer_seed=424242,
        curriculum_sampler_seed=999,
        base_trainer_config=ReptileTrainerConfig(task_failure_abort_threshold=20),
        saturation_epsilon=1e-4,
    )
    rep = run_k_sweep(cfg)
    positions = [row.batch_position for row in rep.per_task_records]
    assert positions == list(range(len(positions)))
    for row in rep.per_task_records:
        assert row.k == 0
        assert row.delta_ic == 0.0
        assert row.ic_meta == row.ic_adapted


@pytest.mark.determinism("d1")
def test_regime_completeness_null_safe(deterministic_seed) -> None:
    _ = deterministic_seed
    cfg = KSweepConfig(
        k_values=(0, 1),
        trainer_seed=424242,
        curriculum_sampler_seed=999,
        base_trainer_config=ReptileTrainerConfig(task_failure_abort_threshold=20),
        saturation_epsilon=1e-4,
    )
    rep = run_k_sweep(cfg)
    keys = {(r.k, r.regime_class) for r in rep.regime_results}
    for k in cfg.k_values:
        for b in ("bull", "bear", "sideways", "high_vol", "crisis"):
            assert (k, b) in keys
            row = next(x for x in rep.regime_results if x.k == k and x.regime_class == b)
            assert row.task_count >= 0
            if row.task_count == 0:
                assert row.mean_delta_ic is None


@pytest.mark.determinism("d1")
def test_completed_arms_positive_wall_clock(deterministic_seed) -> None:
    _ = deterministic_seed
    rep = run_k_sweep(KSweepConfig.default_mlc4_bounded())
    for a in rep.arm_results:
        if a.arm_status == "COMPLETED":
            assert a.wall_clock_s > 0.0


@pytest.mark.determinism("d1")
def test_second_emit_fails_immutability(tmp_path: Path, deterministic_seed) -> None:
    _ = deterministic_seed
    p = tmp_path / "r.json"
    rep = run_k_sweep(
        KSweepConfig(
            k_values=(0, 1),
            trainer_seed=424242,
            curriculum_sampler_seed=999,
            base_trainer_config=ReptileTrainerConfig(task_failure_abort_threshold=20),
            saturation_epsilon=1e-4,
        )
    )
    emit_reptile_k_sweep_report_json(p, rep)
    with pytest.raises(ArtifactImmutabilityError):
        emit_reptile_k_sweep_report_json(p, rep)


@pytest.mark.determinism("d1")
def test_nan_ic_marks_failed_arm(monkeypatch: pytest.MonkeyPatch, deterministic_seed) -> None:
    _ = deterministic_seed
    import pysrc.meta.reptile_k_sweep_runner as rnr

    def _fake_inner(*_a, **_k):
        import numpy as np

        z = np.zeros(64, dtype=np.float32)
        return z, [], 0.0, float("nan")

    monkeypatch.setattr(rnr, "_inner_adapt", _fake_inner)
    cfg = KSweepConfig(
        k_values=(0,),
        trainer_seed=424242,
        curriculum_sampler_seed=999,
        base_trainer_config=ReptileTrainerConfig(task_failure_abort_threshold=20),
        saturation_epsilon=1e-4,
    )
    rep = run_k_sweep(cfg)
    assert rep.arm_results[0].arm_status == "FAILED"
    assert rep.arm_results[0].failure_code == "INNER_DIVERGENCE"


@pytest.mark.determinism("d1")
def test_inner_nonfinite_marks_failed_arm(
    monkeypatch: pytest.MonkeyPatch,
    deterministic_seed,
) -> None:
    _ = deterministic_seed
    import pysrc.meta.reptile_k_sweep_runner as rnr

    def _boom(*_a, **_k):
        raise ValueError("inner_nonfinite")

    monkeypatch.setattr(rnr, "_inner_adapt", _boom)
    cfg = KSweepConfig(
        k_values=(0, 1),
        trainer_seed=424242,
        curriculum_sampler_seed=999,
        base_trainer_config=ReptileTrainerConfig(task_failure_abort_threshold=20),
        saturation_epsilon=1e-4,
    )
    rep = run_k_sweep(cfg)
    assert rep.arm_results[0].arm_status == "FAILED"
    assert rep.arm_results[0].failure_code == "INNER_DIVERGENCE"
    assert not rep.per_task_records


@pytest.mark.determinism("d1")
def test_logging_start_end(deterministic_seed, monkeypatch: pytest.MonkeyPatch) -> None:
    _ = deterministic_seed
    import pysrc.meta.reptile_k_sweep_runner as rnr

    events: list[str] = []

    def _cap(msg: str, **kw: object) -> None:
        events.append(msg)

    monkeypatch.setattr(rnr.LOG, "info", _cap)
    run_k_sweep(
        KSweepConfig(
            k_values=(0,),
            trainer_seed=424242,
            curriculum_sampler_seed=999,
            base_trainer_config=ReptileTrainerConfig(task_failure_abort_threshold=20),
            saturation_epsilon=1e-4,
        )
    )
    assert "mlc4_k_sweep_start" in events
    assert "mlc4_k_sweep_end" in events


@pytest.mark.determinism("d1")
def test_saturation_flat_curve() -> None:
    assert saturation_from_adjacent_means(1.0, 1.0, epsilon=0.01) is True
    assert saturation_from_adjacent_means(1.0, 1.5, epsilon=0.01) is False
    assert saturation_from_adjacent_means(None, 1.0, epsilon=0.01) is False


@pytest.mark.determinism("d1")
def test_hash_recomputes_from_round_trip_json(tmp_path: Path, deterministic_seed) -> None:
    _ = deterministic_seed
    p = tmp_path / "out.json"
    rep = run_k_sweep(
        KSweepConfig(
            k_values=(0, 1),
            trainer_seed=424242,
            curriculum_sampler_seed=999,
            base_trainer_config=ReptileTrainerConfig(task_failure_abort_threshold=20),
            saturation_epsilon=1e-4,
        )
    )
    emit_reptile_k_sweep_report_json(p, rep)
    loaded = json.loads(p.read_text(encoding="utf-8"))
    assert recompute_content_hash_from_document(loaded) == loaded["content_hash"]["value"]


@pytest.mark.determinism("d1")
def test_default_mlc4_bounded_rejects_stale_parallel_kwarg() -> None:
    with pytest.raises(TypeError):
        KSweepConfig.default_mlc4_bounded(parallel_worker_cap=3)


@pytest.mark.determinism("d1")
def test_ksweep_config_rejects_stale_parallel_kwarg() -> None:
    with pytest.raises(TypeError):
        KSweepConfig(
            k_values=(0,),
            trainer_seed=1,
            curriculum_sampler_seed=1,
            base_trainer_config=ReptileTrainerConfig(task_failure_abort_threshold=20),
            saturation_epsilon=1e-4,
            parallel_worker_cap=1,
        )


@pytest.mark.determinism("d1")
def test_governed_k_constants() -> None:
    assert GOVERNED_K_VALUES == (0, 1, 2, 5, 10, 20)


@pytest.mark.determinism("d1")
def test_report_json_document_matches_canonical_hash(deterministic_seed) -> None:
    _ = deterministic_seed
    rep = run_k_sweep(
        KSweepConfig(
            k_values=(0,),
            trainer_seed=424242,
            curriculum_sampler_seed=999,
            base_trainer_config=ReptileTrainerConfig(task_failure_abort_threshold=20),
            saturation_epsilon=1e-4,
        )
    )
    doc = rep.to_json_document()
    assert doc["content_hash"]["value"] == recompute_content_hash_from_document(doc)
    assert isinstance(rep, KSweepReport)


@pytest.mark.determinism("d1")
def test_no_print_in_new_modules() -> None:
    _no_print_calls_in_modules()


@pytest.mark.determinism("d1")
def test_saturation_epsilon_invalid() -> None:
    with pytest.raises(KSweepValidationError):
        KSweepConfig(
            k_values=(0,),
            trainer_seed=1,
            curriculum_sampler_seed=1,
            base_trainer_config=ReptileTrainerConfig(task_failure_abort_threshold=20),
            saturation_epsilon=float("nan"),
        )


@pytest.mark.determinism("d1")
def test_per_task_record_keys_unique_with_batch_position(
    monkeypatch: pytest.MonkeyPatch,
    deterministic_seed,
) -> None:
    _ = deterministic_seed
    import pysrc.meta.reptile_k_sweep_runner as rnr
    from pysrc.meta.reptile_trainer_benchmark import build_synthetic_benchmark_pool

    t = build_synthetic_benchmark_pool()[0]
    fake_batch = (t, t)

    def _fake_sample(*, curriculum_sampler_seed: int) -> tuple[list, tuple]:
        _ = curriculum_sampler_seed
        return [], fake_batch

    monkeypatch.setattr(rnr, "_sample_benchmark_tasks", _fake_sample)
    cfg = KSweepConfig(
        k_values=(0,),
        trainer_seed=424242,
        curriculum_sampler_seed=999,
        base_trainer_config=ReptileTrainerConfig(task_failure_abort_threshold=20),
        saturation_epsilon=1e-4,
    )
    rep = run_k_sweep(cfg)
    keys = {(r.k, r.task_id, r.batch_position) for r in rep.per_task_records}
    assert len(keys) == len(rep.per_task_records)
    same_id = [r for r in rep.per_task_records if r.task_id == t.task_id]
    assert len(same_id) == 2
    assert {r.batch_position for r in same_id} == {0, 1}


@pytest.mark.determinism("d1")
def test_load_round_trip(tmp_path: Path, deterministic_seed) -> None:
    _ = deterministic_seed
    from pysrc.meta.reptile_k_sweep_runner import load_k_sweep_report_json

    p = tmp_path / "x.json"
    rep = run_k_sweep(
        KSweepConfig(
            k_values=(0,),
            trainer_seed=424242,
            curriculum_sampler_seed=999,
            base_trainer_config=ReptileTrainerConfig(task_failure_abort_threshold=20),
            saturation_epsilon=1e-4,
        )
    )
    emit_reptile_k_sweep_report_json(p, rep)
    doc = load_k_sweep_report_json(p)
    assert doc["schema_version"] == rep.schema_version
    assert "batch_position" in doc["per_task_records"][0]


@pytest.mark.determinism("d1")
def test_inner_loop_divergence_propagates_non_inner_valueerror(
    monkeypatch: pytest.MonkeyPatch, deterministic_seed
) -> None:
    _ = deterministic_seed
    import pysrc.meta.reptile_k_sweep_runner as rnr

    def _bad(*_a, **_k):
        raise ValueError("other")

    monkeypatch.setattr(rnr, "_inner_adapt", _bad)
    cfg = KSweepConfig(
        k_values=(0,),
        trainer_seed=424242,
        curriculum_sampler_seed=999,
        base_trainer_config=ReptileTrainerConfig(task_failure_abort_threshold=20),
        saturation_epsilon=1e-4,
    )
    with pytest.raises(ValueError, match="other"):
        run_k_sweep(cfg)

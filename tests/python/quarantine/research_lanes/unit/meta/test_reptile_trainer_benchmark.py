from __future__ import annotations

# ruff: noqa: S101 — pytest uses assert for expectations
import pytest

from pysrc.meta.reptile_trainer_benchmark import (
    run_bounded_benchmark,
    run_default_mlc4_k_sweep_evidence,
    run_default_mlc5_proxy_alignment_evidence,
    run_default_mlc6_ewc_forgetting_evidence,
)


@pytest.mark.determinism("d1")
def test_run_bounded_benchmark_smoke(deterministic_seed) -> None:
    _ = deterministic_seed
    out = run_bounded_benchmark(seed=424242)
    assert out["schema_version"] == "mlc3.benchmark.v1"
    assert out["seed"] == 424242


@pytest.mark.determinism("d1")
def test_run_default_mlc4_k_sweep_evidence(deterministic_seed) -> None:
    _ = deterministic_seed
    rep = run_default_mlc4_k_sweep_evidence()
    assert rep.gate_ii_status == "DEFERRED"
    assert rep.promotion_evidence is False
    assert rep.determinism_tier == "D1"
    assert len(rep.arm_results) == 6


@pytest.mark.determinism("d1")
def test_run_default_mlc5_proxy_alignment_evidence(deterministic_seed) -> None:
    _ = deterministic_seed
    rep = run_default_mlc5_proxy_alignment_evidence()
    assert rep.gate_ii_status == "DEFERRED"
    assert rep.promotion_evidence is False
    assert rep.determinism_tier == "D1"
    assert rep.n_outer_epochs == 20


@pytest.mark.determinism("d1")
def test_run_default_mlc6_ewc_forgetting_evidence(deterministic_seed) -> None:
    _ = deterministic_seed
    rep = run_default_mlc6_ewc_forgetting_evidence()
    assert rep.gate_ii_status == "DEFERRED"
    assert rep.promotion_evidence is False
    assert rep.determinism_tier == "D1"

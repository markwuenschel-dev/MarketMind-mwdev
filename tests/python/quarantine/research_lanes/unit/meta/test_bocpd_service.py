"""Unit tests for BOCPD core + BOCPDRegimeService."""

from __future__ import annotations

from datetime import UTC, datetime

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings

from pysrc.meta.bocpd_service import (
    BOCPDRegimeService,
    NIGPrior,
    ServiceSnapshot,
    SufficientStats,
    bocpd_update,
)
from pysrc.meta.regime_config import BOCPDConfig


def _make_prior(cfg: BOCPDConfig) -> NIGPrior:
    return NIGPrior(
        mu0=0.0,
        kappa0=float(cfg.prior_kappa0),
        alpha0=float(cfg.prior_alpha0),
        beta0=1.0,
    )


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_single_update_log_space_no_underflow(deterministic_seed: int) -> None:
    cfg = BOCPDConfig(max_run_length=128)
    prior = _make_prior(cfg)
    log_p = np.array([0.0], dtype=np.float64)
    stats = SufficientStats(
        mu=np.array([prior.mu0]),
        kappa=np.array([prior.kappa0]),
        alpha=np.array([prior.alpha0]),
        beta=np.array([prior.beta0]),
    )
    rng = np.random.default_rng(deterministic_seed)
    for _ in range(1500):
        x = float(rng.normal(0.0, 1.0))
        log_p, stats, _ = bocpd_update(x, log_p, stats, cfg, prior)
        assert np.all(np.isfinite(log_p))
        assert np.isfinite(float(np.logaddexp.reduce(log_p)))


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_change_probability_sums_correctly(deterministic_seed: int) -> None:
    cfg = BOCPDConfig(max_run_length=64)
    prior = _make_prior(cfg)
    log_p = np.array([0.0], dtype=np.float64)
    stats = SufficientStats(
        mu=np.array([prior.mu0]),
        kappa=np.array([prior.kappa0]),
        alpha=np.array([prior.alpha0]),
        beta=np.array([prior.beta0]),
    )
    rng = np.random.default_rng(deterministic_seed)
    for _ in range(500):
        x = float(rng.normal(0.0, 0.5))
        log_p, stats, _ = bocpd_update(x, log_p, stats, cfg, prior)
        assert abs(float(np.exp(np.logaddexp.reduce(log_p))) - 1.0) < 1e-5


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_truncation_preserves_total_mass() -> None:
    cfg = BOCPDConfig(max_run_length=16, hazard_rate=0.05)
    prior = _make_prior(cfg)
    log_p = np.array([0.0], dtype=np.float64)
    stats = SufficientStats(
        mu=np.array([prior.mu0]),
        kappa=np.array([prior.kappa0]),
        alpha=np.array([prior.alpha0]),
        beta=np.array([prior.beta0]),
    )
    for i in range(200):
        x = 0.1 * np.sin(i / 3.0)
        log_p_before = log_p.copy()
        m_before = float(np.exp(np.logaddexp.reduce(log_p_before)))
        log_p, stats, _ = bocpd_update(float(x), log_p, stats, cfg, prior)
        m_after = float(np.exp(np.logaddexp.reduce(log_p)))
        assert abs(m_before - m_after) < 1e-4
        assert log_p.size <= cfg.max_run_length


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_snapshot_restore_deterministic(deterministic_seed: int) -> None:
    cfg = BOCPDConfig(max_run_length=64)
    prior = _make_prior(cfg)
    log_p = np.array([0.0], dtype=np.float64)
    stats = SufficientStats(
        mu=np.array([prior.mu0]),
        kappa=np.array([prior.kappa0]),
        alpha=np.array([prior.alpha0]),
        beta=np.array([prior.beta0]),
    )
    rng = np.random.default_rng(deterministic_seed)
    for _ in range(30):
        x = float(rng.normal(0.0, 1.0))
        log_p, stats, _ = bocpd_update(x, log_p, stats, cfg, prior)
    snap = ServiceSnapshot(
        log_posterior=log_p,
        sufficient_stats=stats,
        observation_count=30,
        config_hash=cfg.content_hash(),
        prior=prior,
    )
    log_p2 = snap.log_posterior.copy()
    stats2 = SufficientStats(
        mu=snap.sufficient_stats.mu.copy(),
        kappa=snap.sufficient_stats.kappa.copy(),
        alpha=snap.sufficient_stats.alpha.copy(),
        beta=snap.sufficient_stats.beta.copy(),
    )
    for _ in range(10):
        x = float(rng.normal(0.0, 1.0))
        log_p, stats, _ = bocpd_update(x, log_p, stats, cfg, prior)
        log_p2, stats2, _ = bocpd_update(x, log_p2, stats2, cfg, prior)
        assert np.allclose(log_p, log_p2)
        assert np.allclose(stats.mu, stats2.mu)


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_replay_bit_identical(deterministic_seed: int) -> None:
    cfg = BOCPDConfig()
    hist = np.linspace(-1.0, 1.0, 50, dtype=np.float64)
    ts = datetime(2020, 1, 1, tzinfo=UTC)

    def run() -> list[str]:
        svc = BOCPDRegimeService(cfg)
        svc.initialize(hist)
        out: list[str] = []
        for i in range(40):
            lr = float(hist[i % hist.size])
            rec = svc.update(
                ts,
                lr,
                log_return=0.001,
                pit_boundary_idx=i,
                log_rv_history=hist[: i + 1],
                returns_history=np.full(i + 1, 0.0001, dtype=np.float64),
            )
            out.append(rec.state_snapshot_id)
        return out

    a = run()
    b = run()
    assert a == b


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_cold_start_flag_first_n_days() -> None:
    cfg = BOCPDConfig(cold_start_burn_in=100)
    hist = np.linspace(0.0, 1.0, 120, dtype=np.float64)
    svc = BOCPDRegimeService(cfg)
    svc.initialize(hist)
    ts = datetime(2020, 1, 1, tzinfo=UTC)
    for i in range(100):
        rec = svc.update(
            ts,
            float(hist[i]),
            log_return=0.0,
            pit_boundary_idx=i,
            log_rv_history=hist[: i + 1],
            returns_history=np.zeros(i + 1, dtype=np.float64),
        )
        assert rec.boundary_flag == "cold_start"


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_update_emits_bocpd_gated_diagnostic_field() -> None:
    cfg = BOCPDConfig(cold_start_burn_in=10)
    hist = np.linspace(0.05, 0.15, 25, dtype=np.float64)
    svc = BOCPDRegimeService(cfg)
    svc.initialize(hist)
    ts = datetime(2020, 1, 1, tzinfo=UTC)
    rec = svc.update(
        ts,
        0.2,
        log_return=0.0,
        pit_boundary_idx=50,
        log_rv_history=np.linspace(0.05, 0.2, 51, dtype=np.float64),
        returns_history=np.zeros(51, dtype=np.float64),
    )
    assert rec.diag_regime_class_bocpd_gated in (
        "bull",
        "bear",
        "sideways",
        "high_vol",
        "crisis",
    )


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_update_emits_regime_label_from_canonical_label_not_regime_class() -> None:
    cfg = BOCPDConfig(cold_start_burn_in=10)
    hist = np.linspace(0.05, 0.15, 25, dtype=np.float64)
    svc = BOCPDRegimeService(cfg)
    svc.initialize(hist)
    ts = datetime(2020, 1, 1, tzinfo=UTC)
    rec = svc.update(
        ts,
        0.2,
        log_return=0.0,
        pit_boundary_idx=50,
        log_rv_history=np.linspace(0.05, 0.2, 51, dtype=np.float64),
        returns_history=np.zeros(51, dtype=np.float64),
    )
    assert rec.regime_label == rec.regime_id
    assert rec.regime_label != rec.regime_class


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_effective_at_is_availability_time() -> None:
    cfg = BOCPDConfig()
    hist = np.linspace(0.0, 1.0, 50, dtype=np.float64)
    svc = BOCPDRegimeService(cfg)
    svc.initialize(hist)
    ts = datetime(2024, 6, 15, 16, 0, tzinfo=UTC)
    rec = svc.update(
        ts,
        0.5,
        log_return=0.0,
        pit_boundary_idx=120,
        log_rv_history=np.linspace(0, 1, 121, dtype=np.float64),
        returns_history=np.zeros(121, dtype=np.float64),
    )
    assert rec.effective_at >= rec.decision_ts


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_state_snapshot_id_excludes_diagnostics() -> None:
    cfg = BOCPDConfig()
    hist = np.linspace(0.0, 1.0, 80, dtype=np.float64)
    svc = BOCPDRegimeService(cfg)
    svc.initialize(hist)
    ts = datetime(2020, 1, 1, tzinfo=UTC)
    i = 50
    rec1 = svc.update(
        ts,
        float(hist[i]),
        log_return=0.01,
        pit_boundary_idx=i,
        log_rv_history=hist[: i + 1],
        returns_history=np.full(i + 1, 0.001, dtype=np.float64),
    )
    svc2 = BOCPDRegimeService(cfg)
    svc2.initialize(hist)
    rec2 = svc2.update(
        ts,
        float(hist[i]),
        log_return=-0.04,
        pit_boundary_idx=i,
        log_rv_history=hist[: i + 1],
        returns_history=np.full(i + 1, -0.002, dtype=np.float64),
    )
    assert rec1.state_snapshot_id == rec2.state_snapshot_id
    assert rec1.trend_score_raw != rec2.trend_score_raw


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_change_point_detection_on_synthetic_shift() -> None:
    """After a sustained level shift in log-RV, posterior entropy should fall (mass concentrates)."""
    cfg = BOCPDConfig(hazard_rate=0.12, cp_threshold=0.05, max_run_length=128)
    hist = np.full(25, 0.02, dtype=np.float64)
    svc = BOCPDRegimeService(cfg)
    svc.initialize(hist)
    ts = datetime(2020, 1, 1, tzinfo=UTC)
    ent_pre: list[float] = []
    ent_post: list[float] = []
    for i in range(60):
        x = 1.85 if i >= 28 else 0.02
        lr = np.full(i + 1, x, dtype=np.float64)
        rec = svc.update(
            ts,
            float(x),
            log_return=float(x),
            pit_boundary_idx=i,
            log_rv_history=lr,
            returns_history=np.full(i + 1, float(x), dtype=np.float64),
        )
        if 10 <= i < 28:
            ent_pre.append(rec.posterior_entropy)
        if i >= 40:
            ent_post.append(rec.posterior_entropy)
    assert float(np.mean(ent_post)) < float(np.mean(ent_pre)) - 0.05


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_student_t_vs_gaussian_config_switch() -> None:
    hist = np.linspace(0.0, 1.0, 30, dtype=np.float64)
    ts = datetime(2020, 1, 1, tzinfo=UTC)

    def run(cfg: BOCPDConfig) -> tuple[float, str]:
        svc = BOCPDRegimeService(cfg)
        svc.initialize(hist)
        rec = svc.update(
            ts,
            0.5,
            log_return=0.0,
            pit_boundary_idx=50,
            log_rv_history=np.linspace(0, 1, 51, dtype=np.float64),
            returns_history=np.zeros(51, dtype=np.float64),
        )
        return rec.change_probability, rec.state_snapshot_id

    c_t, s_t = run(BOCPDConfig(observation_model="student_t"))
    c_g, s_g = run(BOCPDConfig(observation_model="gaussian"))
    assert np.isfinite(c_t)
    assert np.isfinite(c_g)
    assert (abs(c_t - c_g) > 1e-6) or (s_t != s_g)


@pytest.mark.property
@pytest.mark.determinism("d2")
@settings(
    max_examples=15,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    hazard_rate=st.floats(min_value=0.005, max_value=0.05, allow_nan=False, allow_infinity=False)
)
def test_hazard_rate_sensitivity(deterministic_seed: int, hazard_rate: float) -> None:
    cfg_lo = BOCPDConfig(hazard_rate=hazard_rate, max_run_length=64, cp_threshold=0.2)
    cfg_hi = BOCPDConfig(
        hazard_rate=min(0.3, hazard_rate * 8.0), max_run_length=64, cp_threshold=0.2
    )
    np.random.default_rng(deterministic_seed)
    xs = np.full(200, 0.02, dtype=np.float64)

    def mean_cp(c: BOCPDConfig) -> float:
        svc = BOCPDRegimeService(c)
        svc.initialize(xs[:30])
        ts = datetime(2020, 1, 1, tzinfo=UTC)
        cps: list[float] = []
        for i in range(80):
            lr = float(xs[i + 30])
            rec = svc.update(
                ts,
                lr,
                log_return=0.0,
                pit_boundary_idx=i + 30,
                log_rv_history=xs[: i + 31],
                returns_history=np.zeros(i + 31, dtype=np.float64),
            )
            cps.append(rec.change_probability)
        return float(np.mean(cps))

    # Smooth constant log-RV: higher hazard allocates more mass to short run lengths →
    # typically higher average change-point probability mass at the current step.
    assert mean_cp(cfg_hi) >= mean_cp(cfg_lo) - 1e-3


@pytest.mark.property
@pytest.mark.determinism("d2")
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(max_rl=st.integers(min_value=8, max_value=64))
def test_truncation_cap_respected(deterministic_seed: int, max_rl: int) -> None:
    cfg = BOCPDConfig(max_run_length=max_rl)
    prior = _make_prior(cfg)
    log_p = np.array([0.0], dtype=np.float64)
    stats = SufficientStats(
        mu=np.array([prior.mu0]),
        kappa=np.array([prior.kappa0]),
        alpha=np.array([prior.alpha0]),
        beta=np.array([prior.beta0]),
    )
    rng = np.random.default_rng(deterministic_seed)
    for _ in range(400):
        x = float(rng.normal(0.0, 1.0))
        log_p, stats, _ = bocpd_update(x, log_p, stats, cfg, prior)
        assert log_p.size <= max_rl


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_snapshot_before_init_raises() -> None:
    svc = BOCPDRegimeService(BOCPDConfig())
    with pytest.raises(RuntimeError, match="initialize"):
        svc.snapshot()


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_update_before_init_raises() -> None:
    svc = BOCPDRegimeService(BOCPDConfig())
    ts = datetime(2020, 1, 1, tzinfo=UTC)
    with pytest.raises(RuntimeError, match="initialize"):
        svc.update(
            ts,
            0.1,
            log_return=0.0,
            pit_boundary_idx=5,
            log_rv_history=np.linspace(0, 1, 6, dtype=np.float64),
            returns_history=np.zeros(6, dtype=np.float64),
        )


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_from_snapshot_rejects_config_hash_mismatch() -> None:
    cfg = BOCPDConfig()
    hist = np.linspace(0.0, 1.0, 20, dtype=np.float64)
    svc = BOCPDRegimeService(cfg)
    svc.initialize(hist)
    snap = svc.snapshot()
    bad = BOCPDConfig(hazard_rate=0.5)
    with pytest.raises(ValueError, match="content_hash"):
        BOCPDRegimeService.from_snapshot(snap, bad)


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_from_snapshot_matches_continued_run(deterministic_seed: int) -> None:
    cfg = BOCPDConfig(max_run_length=128)
    hist = np.linspace(0.0, 1.0, 60, dtype=np.float64)
    svc = BOCPDRegimeService(cfg)
    svc.initialize(hist)
    ts = datetime(2020, 1, 1, tzinfo=UTC)
    for i in range(15):
        svc.update(
            ts,
            float(hist[i]),
            log_return=0.0,
            pit_boundary_idx=i,
            log_rv_history=hist[: i + 1],
            returns_history=np.zeros(i + 1, dtype=np.float64),
        )
    snap = svc.snapshot()
    svc2 = BOCPDRegimeService.from_snapshot(snap, cfg)
    for i in range(15, 25):
        r1 = svc.update(
            ts,
            float(hist[i % hist.size]),
            log_return=0.0,
            pit_boundary_idx=i,
            log_rv_history=np.linspace(0, 1, i + 1, dtype=np.float64),
            returns_history=np.zeros(i + 1, dtype=np.float64),
        )
        r2 = svc2.update(
            ts,
            float(hist[i % hist.size]),
            log_return=0.0,
            pit_boundary_idx=i,
            log_rv_history=np.linspace(0, 1, i + 1, dtype=np.float64),
            returns_history=np.zeros(i + 1, dtype=np.float64),
        )
        assert r1.state_snapshot_id == r2.state_snapshot_id

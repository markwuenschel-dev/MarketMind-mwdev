"""Adams & MacKay BOCPD (log-space) + orchestrator regime service (AQ-04)."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from pysrc.meta.regime_config import BOCPDConfig
from pysrc.meta.regime_labeler import RegimeLabeler
from pysrc.meta_learning.regime_vocabulary import RegimeClassLabel
from pysrc.ops.hashing import canonicalize_json_bytes


@dataclass(frozen=True)
class NIGPrior:
    """Normal-inverse-gamma prior hyperparameters for log-RV observations."""

    mu0: float
    kappa0: float
    alpha0: float
    beta0: float


@dataclass(frozen=True)
class SufficientStats:
    """Per-run-length NIG posterior parameters (same length as log_posterior)."""

    mu: NDArray[np.float64]
    kappa: NDArray[np.float64]
    alpha: NDArray[np.float64]
    beta: NDArray[np.float64]


def nig_update_obs(
    mu: float, kappa: float, alpha: float, beta: float, x: float
) -> tuple[float, float, float, float]:
    """Sequential NIG update after one Gaussian-like observation (Murphy 2007)."""
    kappa_new = kappa + 1.0
    mu_new = mu + (x - mu) / kappa_new
    alpha_new = alpha + 0.5
    beta_new = beta + 0.5 * (x - mu) ** 2 * kappa / kappa_new
    return mu_new, kappa_new, alpha_new, beta_new


def _nig_to_sums(
    mu: float, kappa: float, alpha: float, beta: float, prior: NIGPrior
) -> tuple[float, float, float]:
    """Recover (n, sum_x, sum_x2) relative to prior for merge operations."""
    n = 2.0 * (alpha - prior.alpha0)
    if n <= 0.0:
        return 0.0, 0.0, 0.0
    sum_x = kappa * mu - prior.kappa0 * prior.mu0
    # Inversion of β_n = β_0 + 0.5 * (sum_x2 + κ_0 μ_0^2 - κ_n μ_n^2) for batch NIG.
    sum_x2 = 2.0 * (beta - prior.beta0) - prior.kappa0 * prior.mu0**2 + kappa * mu**2
    return n, sum_x, sum_x2


def _sums_to_nig(
    n: float, sum_x: float, sum_x2: float, prior: NIGPrior
) -> tuple[float, float, float, float]:
    """Batch NIG posterior from prior + aggregate sums (n observations)."""
    kappa_n = prior.kappa0 + n
    mu_n = (prior.kappa0 * prior.mu0 + sum_x) / kappa_n
    alpha_n = prior.alpha0 + n / 2.0
    beta_n = prior.beta0 + 0.5 * (sum_x2 + prior.kappa0 * prior.mu0**2 - kappa_n * mu_n**2)
    return mu_n, kappa_n, alpha_n, beta_n


def merge_nig_tail(
    mus: NDArray[np.float64],
    kappas: NDArray[np.float64],
    alphas: NDArray[np.float64],
    betas: NDArray[np.float64],
    prior: NIGPrior,
) -> tuple[float, float, float, float]:
    """
    Merge multiple per-run-length NIG states into one using additive (n, sum_x, sum_x2).

    Used when folding truncated run-length bins into the final retained bin; preserves
    total posterior mass in the run-length dimension (paired with log-mass folding).
    """
    n_tot = 0.0
    sx_tot = 0.0
    sx2_tot = 0.0
    for i in range(len(mus)):
        n_i, sx_i, sx2_i = _nig_to_sums(
            float(mus[i]), float(kappas[i]), float(alphas[i]), float(betas[i]), prior
        )
        n_tot += n_i
        sx_tot += sx_i
        sx2_tot += sx2_i
    return _sums_to_nig(n_tot, sx_tot, sx2_tot, prior)


def _logpdf_student_t(x: float, df: float, loc: float, scale: float) -> float:
    """Log PDF of Student-t (scalar); df > 0, scale > 0."""
    if df <= 0.0 or scale <= 0.0:
        return -np.inf
    z = (x - loc) / scale
    return float(
        math.lgamma((df + 1.0) / 2.0)
        - math.lgamma(df / 2.0)
        - 0.5 * math.log(df * math.pi)
        - math.log(scale)
        - (df + 1.0) / 2.0 * math.log1p(z * z / df)
    )


def _logpdf_normal(x: float, loc: float, scale: float) -> float:
    if scale <= 0.0:
        return -np.inf
    z = (x - loc) / scale
    return float(-0.5 * math.log(2.0 * math.pi) - math.log(scale) - 0.5 * z * z)


def predictive_logpdf(
    x: float,
    mu: float,
    kappa: float,
    alpha: float,
    beta: float,
    observation_model: Literal["student_t", "gaussian"],
) -> float:
    """Posterior predictive log-density at x under NIG (Student-t default; Gaussian ablation)."""
    if alpha <= 0.0 or kappa <= 0.0 or beta <= 0.0:
        return -np.inf
    if observation_model == "student_t":
        df = 2.0 * alpha
        scale = math.sqrt(beta * (kappa + 1.0) / (alpha * kappa))
        if scale <= 0.0 or not math.isfinite(scale):
            return -np.inf
        return _logpdf_student_t(x, df, mu, scale)
    var = (beta / alpha) * (kappa + 1.0) / kappa
    if var <= 0.0 or not math.isfinite(var):
        return -np.inf
    scale = math.sqrt(var)
    return _logpdf_normal(x, mu, scale)


def _extend_stats_with_observation(
    prior: NIGPrior, old: SufficientStats, x: float
) -> SufficientStats:
    """Prepend prior-updated segment; shift-update each legacy hypothesis (Gundersen-style recursion)."""
    n = old.mu.size
    mu = np.empty(n + 1, dtype=np.float64)
    kappa = np.empty(n + 1, dtype=np.float64)
    alpha = np.empty(n + 1, dtype=np.float64)
    beta = np.empty(n + 1, dtype=np.float64)
    mu[0], kappa[0], alpha[0], beta[0] = nig_update_obs(
        prior.mu0, prior.kappa0, prior.alpha0, prior.beta0, x
    )
    for i in range(n):
        mu[i + 1], kappa[i + 1], alpha[i + 1], beta[i + 1] = nig_update_obs(
            float(old.mu[i]), float(old.kappa[i]), float(old.alpha[i]), float(old.beta[i]), x
        )
    return SufficientStats(mu=mu, kappa=kappa, alpha=alpha, beta=beta)


def _truncate_run_length_mass(
    log_post: NDArray[np.float64],
    stats: SufficientStats,
    prior: NIGPrior,
    max_run_length: int,
) -> tuple[NDArray[np.float64], SufficientStats]:
    """
    Truncate to max_run_length; fold excess log-mass into the final bin.

    NIG parameters for the final bin merge all truncated tail hypotheses via (n, sum_x, sum_x2).
    """
    if log_post.size <= max_run_length:
        return log_post, stats
    tail = log_post[max_run_length - 1 :]
    folded = float(np.logaddexp.reduce(tail))
    new_log = np.concatenate([log_post[: max_run_length - 1], np.array([folded], dtype=np.float64)])
    tail_mu = stats.mu[max_run_length - 1 :]
    tail_ka = stats.kappa[max_run_length - 1 :]
    tail_al = stats.alpha[max_run_length - 1 :]
    tail_be = stats.beta[max_run_length - 1 :]
    merged_mu, merged_ka, merged_al, merged_be = merge_nig_tail(
        tail_mu, tail_ka, tail_al, tail_be, prior
    )
    new_mu = np.append(stats.mu[: max_run_length - 1], merged_mu)
    new_ka = np.append(stats.kappa[: max_run_length - 1], merged_ka)
    new_al = np.append(stats.alpha[: max_run_length - 1], merged_al)
    new_be = np.append(stats.beta[: max_run_length - 1], merged_be)
    return new_log, SufficientStats(mu=new_mu, kappa=new_ka, alpha=new_al, beta=new_be)


def bocpd_update(
    x: float,
    log_posterior: NDArray[np.float64],
    sufficient_stats: SufficientStats,
    config: BOCPDConfig,
    prior: NIGPrior,
) -> tuple[NDArray[np.float64], SufficientStats, float]:
    """
    Single-step Adams & MacKay update (log-space; Gundersen filtering structure).

    Returns: (new_log_posterior, new_sufficient_stats, change_probability)
    Index 0 is the changepoint hypothesis mass after observing x.
    """
    h = config.hazard_rate
    log_h = math.log(h)
    log_1mh = math.log(1.0 - h)
    obs = config.observation_model

    log_msg = log_posterior
    log_pred = np.empty_like(log_msg, dtype=np.float64)
    for i in range(log_msg.size):
        log_pred[i] = predictive_logpdf(
            x,
            float(sufficient_stats.mu[i]),
            float(sufficient_stats.kappa[i]),
            float(sufficient_stats.alpha[i]),
            float(sufficient_stats.beta[i]),
            obs,
        )

    # Changepoint branch: prior predictive for a brand-new segment (not a mixture of per-run-length predictives).
    log_prior_pred = predictive_logpdf(
        x,
        float(prior.mu0),
        float(prior.kappa0),
        float(prior.alpha0),
        float(prior.beta0),
        obs,
    )

    log_cp = log_prior_pred + log_h + float(np.logaddexp.reduce(log_msg))
    log_growth = log_pred + log_msg + log_1mh
    new_log_joint = np.concatenate((np.array([log_cp], dtype=np.float64), log_growth))
    new_log_joint -= float(np.logaddexp.reduce(new_log_joint))

    new_stats = _extend_stats_with_observation(prior, sufficient_stats, x)
    new_log, new_stats2 = _truncate_run_length_mass(
        new_log_joint, new_stats, prior, config.max_run_length
    )

    change_probability = math.exp(float(new_log[0] - np.logaddexp.reduce(new_log)))
    return new_log, new_stats2, change_probability


def _posterior_entropy(log_p: NDArray[np.float64]) -> float:
    p = np.exp(log_p - np.max(log_p))
    z = float(np.sum(p))
    if z <= 0.0:
        return 0.0
    p = p / z
    mask = p > 0.0
    return float(-np.sum(p[mask] * np.log(p[mask])))


def _transition_mass(log_p: NDArray[np.float64], transition_max_rl: int) -> float:
    """Mass on run-length indices 0..transition_max_rl inclusive (under full posterior)."""
    hi = min(transition_max_rl, log_p.size - 1)
    if hi < 0:
        return 0.0
    return float(
        math.exp(float(np.logaddexp.reduce(log_p[: hi + 1])) - float(np.logaddexp.reduce(log_p)))
    )


def _run_length_expectation_index(log_p: NDArray[np.float64]) -> float:
    """Expected run-length index under the posterior (0 = CP hypothesis)."""
    p = np.exp(log_p - np.max(log_p))
    z = float(np.sum(p))
    if z <= 0.0:
        return 0.0
    p = p / z
    idx = np.arange(log_p.size, dtype=np.float64)
    return float(np.sum(idx * p))


def _bocpd_state_from_probs(
    change_probability: float,
    transition_mass: float,
    config: BOCPDConfig,
) -> Literal["stable", "transition", "cp"]:
    if change_probability >= config.cp_threshold:
        return "cp"
    if transition_mass >= config.transition_threshold:
        return "transition"
    return "stable"


@dataclass(frozen=True)
class ServiceSnapshot:
    """Compact replay state (truncated log-posterior + NIG grid + counters + prior)."""

    log_posterior: NDArray[np.float64]
    sufficient_stats: SufficientStats
    observation_count: int
    config_hash: str
    prior: NIGPrior


def state_snapshot_id(snapshot: ServiceSnapshot) -> str:
    """Content-addressed id from snapshot (excludes labels, diagnostics, timestamps)."""
    payload: dict[str, Any] = {
        "log_posterior": snapshot.log_posterior.astype(np.float64).tolist(),
        "sufficient_stats": {
            "mu": snapshot.sufficient_stats.mu.astype(np.float64).tolist(),
            "kappa": snapshot.sufficient_stats.kappa.astype(np.float64).tolist(),
            "alpha": snapshot.sufficient_stats.alpha.astype(np.float64).tolist(),
            "beta": snapshot.sufficient_stats.beta.astype(np.float64).tolist(),
        },
        "observation_count": snapshot.observation_count,
        "config_hash": snapshot.config_hash,
    }
    body = canonicalize_json_bytes(payload)
    digest = hashlib.sha256(body).hexdigest()
    return f"sha256:{digest}"


@dataclass(frozen=True)
class RegimeLabelRecord:
    """Canonical replay fields + additive regime_class + diagnostic sidecar.

    `regime_label` carries the canonical service label for replay compatibility.
    Under the governed BOCPD service this is the same compositional label as
    `regime_id`; `regime_class` remains a non-canonical additive projection.

    ``effective_at`` is the **availability time** — the earliest time a
    governed consumer could observe this label.  It is set from
    ``decision_ts``, not from a retrospective change-point timestamp
    (AQ-04 resolution).  ``cold_start`` is an explicit boolean flag
    (``True`` during the ``cold_start_burn_in`` window); it duplicates
    the structural information in ``boundary_flag`` but is surfaced as a
    separate field so governed downstream gates can reject burn-in
    labels without string compares.
    """

    entity_id: str
    decision_ts: datetime
    regime_id: str
    regime_label: str
    effective_at: datetime
    state_snapshot_id: str
    input_snapshot_id: str
    config_version: str
    change_probability: float
    boundary_flag: Literal["cold_start", "change_point", "transition", "stable"]
    regime_class: RegimeClassLabel
    diag_regime_class_bocpd_gated: RegimeClassLabel
    diag_regime_class_extended: RegimeClassLabel
    run_length_mode: int
    run_length_expectation: float
    transition_probability: float
    posterior_entropy: float
    trend_score_raw: float
    vol_score_raw: float
    cold_start: bool = False


class BOCPDRegimeService:
    """
    Orchestrator-managed regime service per AQ-04.

    Monitored signal: log(annualized 21-day realized vol).
    This service accepts ONLY log-RV values for the BOCPD core. The caller supplies
    log_return for compositional regime_id labeling.
    """

    def __init__(self, config: BOCPDConfig) -> None:
        self._config = config
        self._prior: NIGPrior | None = None
        self._log_post: NDArray[np.float64] | None = None
        self._stats: SufficientStats | None = None
        self._obs_count = 0
        self._labeler = RegimeLabeler(config)
        self._input_snapshot_id = ""

    def initialize(self, historical_log_rv: NDArray[np.float64]) -> None:
        """Estimate data-dependent NIG priors from a burn-in slice; BOCPD starts empty (online stream)."""
        x = np.asarray(historical_log_rv, dtype=np.float64).ravel()
        if x.size == 0:
            raise ValueError("historical_log_rv must be non-empty")

        burn = min(x.size, max(1, self._config.cold_start_burn_in))
        burn_slice = x[:burn]

        mu0 = (
            float(np.mean(burn_slice))
            if self._config.prior_mu0 is None
            else float(self._config.prior_mu0)
        )
        if self._config.prior_beta0 is None:
            v = float(np.var(burn_slice, ddof=0))
            beta0 = max(v * (self._config.prior_alpha0 - 1.0), 1e-8)
        else:
            beta0 = float(self._config.prior_beta0)

        self._prior = NIGPrior(
            mu0=mu0,
            kappa0=float(self._config.prior_kappa0),
            alpha0=float(self._config.prior_alpha0),
            beta0=beta0,
        )
        prior = self._prior

        self._log_post = np.array([0.0], dtype=np.float64)
        self._stats = SufficientStats(
            mu=np.array([prior.mu0], dtype=np.float64),
            kappa=np.array([prior.kappa0], dtype=np.float64),
            alpha=np.array([prior.alpha0], dtype=np.float64),
            beta=np.array([prior.beta0], dtype=np.float64),
        )
        self._obs_count = 0

        payload = {"prior_mu0": mu0, "prior_beta0": beta0, "n_hist": int(x.size), "burn": int(burn)}
        self._input_snapshot_id = (
            f"sha256:{hashlib.sha256(canonicalize_json_bytes(payload)).hexdigest()}"
        )

    def update(
        self,
        decision_ts: datetime,
        log_rv: float,
        *,
        log_return: float,
        entity_id: str = "ES",
        pit_boundary_idx: int,
        log_rv_history: NDArray[np.float64],
        returns_history: NDArray[np.float64],
    ) -> RegimeLabelRecord:
        """Process one log-RV observation; emit label + diagnostics."""
        if self._prior is None or self._log_post is None or self._stats is None:
            raise RuntimeError("initialize() must be called before update()")

        prior = self._prior
        self._log_post, self._stats, cp = bocpd_update(
            float(log_rv), self._log_post, self._stats, self._config, prior
        )
        self._obs_count += 1

        log_p = self._log_post
        trans_mass = _transition_mass(log_p, self._config.transition_max_rl)
        r_mode = int(np.argmax(log_p))
        r_exp = _run_length_expectation_index(log_p)
        h_ent = _posterior_entropy(log_p)

        b_state = _bocpd_state_from_probs(cp, trans_mass, self._config)

        trend = self._labeler.compute_trend_regime(returns_history, pit_boundary_idx)
        vol = self._labeler.compute_vol_regime(log_rv_history, pit_boundary_idx)
        regime_id = self._labeler.compute_regime_id(trend, vol, b_state)
        sev = self._labeler.compute_severity_flag_vol_score_raw(log_rv_history, pit_boundary_idx)
        # b_state still passed: Level 1 regime_id only; MLN-02-AMD-01 drops BOCPD from L2 crisis.
        r_class = self._labeler.project_regime_class(trend, vol, b_state, severity_flag=sev)
        r_bocpd = RegimeLabeler.project_regime_class_bocpd_gated_reference(trend, vol, b_state)
        r_class_ext = self._labeler.project_regime_class_extended(trend, vol, b_state)

        is_cold_start = pit_boundary_idx < self._config.cold_start_burn_in
        if is_cold_start:
            boundary: Literal["cold_start", "change_point", "transition", "stable"] = "cold_start"
        elif cp >= self._config.cp_threshold:
            boundary = "change_point"
        elif trans_mass >= self._config.transition_threshold:
            boundary = "transition"
        else:
            boundary = "stable"

        assert self._prior is not None
        snap = ServiceSnapshot(
            log_posterior=self._log_post.copy(),
            sufficient_stats=SufficientStats(
                mu=self._stats.mu.copy(),
                kappa=self._stats.kappa.copy(),
                alpha=self._stats.alpha.copy(),
                beta=self._stats.beta.copy(),
            ),
            observation_count=self._obs_count,
            config_hash=self._config.content_hash(),
            prior=self._prior,
        )
        sid = state_snapshot_id(snap)

        if decision_ts.tzinfo is None:
            effective_at = decision_ts.replace(tzinfo=UTC)
        else:
            effective_at = decision_ts.astimezone(UTC)

        return RegimeLabelRecord(
            entity_id=entity_id,
            decision_ts=decision_ts,
            regime_id=regime_id,
            regime_label=regime_id,
            effective_at=effective_at,
            state_snapshot_id=sid,
            input_snapshot_id=self._input_snapshot_id,
            config_version=self._config.config_version,
            change_probability=cp,
            boundary_flag=boundary,
            regime_class=r_class,
            diag_regime_class_bocpd_gated=r_bocpd,
            diag_regime_class_extended=r_class_ext,
            run_length_mode=r_mode,
            run_length_expectation=r_exp,
            transition_probability=trans_mass,
            posterior_entropy=h_ent,
            trend_score_raw=float(log_return),
            vol_score_raw=float(log_rv),
            cold_start=bool(is_cold_start),
        )

    def snapshot(self) -> ServiceSnapshot:
        """Serialize current BOCPD state for replay."""
        if self._prior is None or self._log_post is None or self._stats is None:
            raise RuntimeError("initialize() must be called before snapshot()")
        return ServiceSnapshot(
            log_posterior=self._log_post.copy(),
            sufficient_stats=SufficientStats(
                mu=self._stats.mu.copy(),
                kappa=self._stats.kappa.copy(),
                alpha=self._stats.alpha.copy(),
                beta=self._stats.beta.copy(),
            ),
            observation_count=self._obs_count,
            config_hash=self._config.content_hash(),
            prior=self._prior,
        )

    @classmethod
    def from_snapshot(cls, snapshot: ServiceSnapshot, config: BOCPDConfig) -> BOCPDRegimeService:
        """Restore service for deterministic continuation."""
        if snapshot.config_hash != config.content_hash():
            raise ValueError("config content_hash does not match snapshot.config_hash")
        svc = cls(config)
        svc._prior = snapshot.prior
        svc._log_post = snapshot.log_posterior.copy()
        svc._stats = SufficientStats(
            mu=snapshot.sufficient_stats.mu.copy(),
            kappa=snapshot.sufficient_stats.kappa.copy(),
            alpha=snapshot.sufficient_stats.alpha.copy(),
            beta=snapshot.sufficient_stats.beta.copy(),
        )
        svc._obs_count = snapshot.observation_count
        svc._labeler = RegimeLabeler(config)
        svc._input_snapshot_id = ""
        return svc

"""Typed BOCPD + regime labeling configuration (RG-09 §5.2)."""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import asdict, dataclass
from typing import Any, Literal

from pysrc.ops.hashing import canonicalize_json_bytes


@dataclass(frozen=True)
class BOCPDConfig:
    """All §5.2 parameters. Frozen — new config = new instance."""

    hazard_rate: float = 1 / 100
    observation_model: Literal["student_t", "gaussian"] = "student_t"
    prior_mu0: float | None = None
    prior_kappa0: float = 1.0
    prior_alpha0: float = 3.0
    prior_beta0: float | None = None
    max_run_length: int = 512
    vol_window: int = 21
    trend_window: int = 63
    trend_flat_epsilon: float = 0.02
    vol_bucket_method: Literal["tercile", "quintile"] = "tercile"
    cp_threshold: float = 0.5
    transition_threshold: float = 0.3
    transition_max_rl: int = 5
    cold_start_burn_in: int = 100
    # Feeds MLN-02-AMD-01 severity_flag only; Level 2 crisis ignores BOCPD by amendment.
    crisis_vol_score_percentile: float = 90.0
    config_version: str = "rg09_v1.0.2"

    def __post_init__(self) -> None:
        if not (0.0 < self.hazard_rate < 1.0):
            raise ValueError("hazard_rate must be in (0, 1)")
        if self.prior_kappa0 <= 0:
            raise ValueError("prior_kappa0 must be positive")
        if self.prior_alpha0 <= 0:
            raise ValueError("prior_alpha0 must be positive")
        if self.prior_beta0 is not None and self.prior_beta0 <= 0:
            raise ValueError("prior_beta0 must be positive when set")
        if self.max_run_length <= 0:
            raise ValueError("max_run_length must be positive")
        if self.vol_window <= 0 or self.trend_window <= 0:
            raise ValueError("vol_window and trend_window must be positive")
        if self.trend_flat_epsilon < 0:
            raise ValueError("trend_flat_epsilon must be non-negative")
        if not (0.0 < self.cp_threshold <= 1.0):
            raise ValueError("cp_threshold must be in (0, 1]")
        if not (0.0 < self.transition_threshold <= 1.0):
            raise ValueError("transition_threshold must be in (0, 1]")
        if self.transition_max_rl < 0:
            raise ValueError("transition_max_rl must be non-negative")
        if self.cold_start_burn_in < 0:
            raise ValueError("cold_start_burn_in must be non-negative")
        if not (0.0 < self.crisis_vol_score_percentile < 100.0):
            raise ValueError("crisis_vol_score_percentile must be in (0, 100)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def content_hash(self) -> str:
        """HMAC-SHA256 over canonical JSON (sorted keys)."""
        payload = self.to_dict()
        key = b"pysrc.rg09.bocpd_config.v1"
        body = canonicalize_json_bytes(payload)
        digest = hmac.new(key, body, hashlib.sha256).hexdigest()
        return f"hmac-sha256:{digest}"

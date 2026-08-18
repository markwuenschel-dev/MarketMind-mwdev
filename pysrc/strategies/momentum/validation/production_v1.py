from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CPCVConfig:
    n_splits: int = 6
    n_test_paths: int = 2


@dataclass(frozen=True)
class ProductionValidationProfile:
    profile_id: str = "production_v1"
    dsr_p_value_max: float = 0.05
    min_trl_target_confidence: float = 0.95
    pbo_max: float = 0.50
    cpcv: CPCVConfig = CPCVConfig()

    def to_dict(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "dsr_p_value_max": self.dsr_p_value_max,
            "min_trl_target_confidence": self.min_trl_target_confidence,
            "pbo_max": self.pbo_max,
            "cpcv": {
                "n_splits": self.cpcv.n_splits,
                "n_test_paths": self.cpcv.n_test_paths,
            },
        }


PRODUCTION_V1_PROFILE = ProductionValidationProfile()

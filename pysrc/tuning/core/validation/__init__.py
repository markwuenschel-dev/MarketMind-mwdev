"""Pure, deterministic validators for walk-forward, purged CV, CPCV, and leakage checks."""

from pysrc.tuning.core.validation.cost_stress import apply_cost_stress
from pysrc.tuning.core.validation.cpcv import cpcv_splits
from pysrc.tuning.core.validation.determinism import fingerprint_splits
from pysrc.tuning.core.validation.embargo import apply_embargo, trim_to_embargo
from pysrc.tuning.core.validation.leakage import LeakageDetectedError, assert_no_future_in_train
from pysrc.tuning.core.validation.purged_cv import purge_training_index, purged_splits
from pysrc.tuning.core.validation.regime_splits import regime_conditioned_splits
from pysrc.tuning.core.validation.replay_consistency import assert_splits_identical
from pysrc.tuning.core.validation.walkforward import walkforward_splits

__all__ = [
    "walkforward_splits",
    "purge_training_index",
    "purged_splits",
    "apply_embargo",
    "trim_to_embargo",
    "cpcv_splits",
    "LeakageDetectedError",
    "assert_no_future_in_train",
    "apply_cost_stress",
    "regime_conditioned_splits",
    "assert_splits_identical",
    "fingerprint_splits",
]

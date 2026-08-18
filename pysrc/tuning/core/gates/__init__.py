"""Pure statistical and structural gate evaluation logic."""

from pysrc.tuning.core.gates.determinism_gate import passes_determinism_gate
from pysrc.tuning.core.gates.execution_integrity import passes_integrity_gate, valid_cas_hash
from pysrc.tuning.core.gates.meta_validity import passes_meta_gate
from pysrc.tuning.core.gates.pit_gate import passes_pit_gate
from pysrc.tuning.core.gates.promotion_gate import evaluate_promotion_gate
from pysrc.tuning.core.gates.rollback_gate import should_rollback
from pysrc.tuning.core.gates.stat_validity import (
    deflated_sharpe_ratio,
    passes_dsr_gate,
    passes_harvey_tstat,
)

__all__ = [
    "deflated_sharpe_ratio",
    "passes_dsr_gate",
    "passes_harvey_tstat",
    "passes_meta_gate",
    "passes_pit_gate",
    "passes_determinism_gate",
    "valid_cas_hash",
    "passes_integrity_gate",
    "evaluate_promotion_gate",
    "should_rollback",
]

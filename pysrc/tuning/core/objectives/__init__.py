"""Pure objective scoring: metrics, penalties, composite scores, and Pareto logic."""

from pysrc.tuning.core.objectives.capacity import (
    apply_capacity_penalty,
    capacity_decay_factor,
)
from pysrc.tuning.core.objectives.composite_score import composite_score
from pysrc.tuning.core.objectives.metrics import (
    calmar_ratio,
    max_drawdown,
    sharpe_ratio,
    turnover_ratio,
)
from pysrc.tuning.core.objectives.pareto import is_pareto_efficient, pareto_front
from pysrc.tuning.core.objectives.penalties import drawdown_penalty, turnover_penalty
from pysrc.tuning.core.objectives.stability import (
    fold_score_variance,
    is_stable,
    stability_score,
)

__all__ = [
    "sharpe_ratio",
    "max_drawdown",
    "turnover_ratio",
    "calmar_ratio",
    "turnover_penalty",
    "drawdown_penalty",
    "composite_score",
    "is_pareto_efficient",
    "pareto_front",
    "fold_score_variance",
    "is_stable",
    "stability_score",
    "capacity_decay_factor",
    "apply_capacity_penalty",
]

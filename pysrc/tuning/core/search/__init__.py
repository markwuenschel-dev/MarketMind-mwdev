"""Pure search primitives: space definitions, samplers, and search algorithms."""

from pysrc.tuning.core.search.bayes_opt import BayesOptError, expected_improvement
from pysrc.tuning.core.search.constraints import (
    ConstraintViolationError,
    check_bounds,
    check_no_nan,
    validate_params,
)
from pysrc.tuning.core.search.dimensions import bounds, is_continuous, is_discrete
from pysrc.tuning.core.search.evolutionary import tournament_select, uniform_crossover
from pysrc.tuning.core.search.multi_fidelity import successive_halving_brackets
from pysrc.tuning.core.search.sampler import sample_uniform
from pysrc.tuning.core.search.space import SearchSpace
from pysrc.tuning.core.search.warm_start import inject_prior_trials

__all__ = [
    "SearchSpace",
    "is_continuous",
    "is_discrete",
    "bounds",
    "ConstraintViolationError",
    "check_bounds",
    "check_no_nan",
    "validate_params",
    "sample_uniform",
    "BayesOptError",
    "expected_improvement",
    "tournament_select",
    "uniform_crossover",
    "successive_halving_brackets",
    "inject_prior_trials",
]

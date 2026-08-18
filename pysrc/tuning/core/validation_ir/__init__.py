"""Boundary validation layer: converts raw IR to validated IR or raises typed errors."""

from pysrc.tuning.core.validation_ir.compatibility_checks import (
    CompatibilityError,
    check_search_objective_compatibility,
    check_search_validation_compatibility,
)
from pysrc.tuning.core.validation_ir.determinism_checks import (
    VALID_TIERS,
    DeterminismViolationError,
    assert_tier_not_downgraded,
    validate_tier,
)
from pysrc.tuning.core.validation_ir.pit_checks import (
    PITViolationError,
    validate_fold_pit,
    validate_no_leakage,
    validate_task_pit,
)
from pysrc.tuning.core.validation_ir.schema_checks import (
    SchemaCheckError,
    validate_metadata,
    validate_spec_hash,
)
from pysrc.tuning.core.validation_ir.structural import (
    StructuralValidationError,
    validate_search_ir,
    validate_task_ir,
)

__all__ = [
    "StructuralValidationError",
    "validate_search_ir",
    "validate_task_ir",
    "SchemaCheckError",
    "validate_spec_hash",
    "validate_metadata",
    "PITViolationError",
    "validate_fold_pit",
    "validate_no_leakage",
    "validate_task_pit",
    "VALID_TIERS",
    "DeterminismViolationError",
    "validate_tier",
    "assert_tier_not_downgraded",
    "CompatibilityError",
    "check_search_validation_compatibility",
    "check_search_objective_compatibility",
]

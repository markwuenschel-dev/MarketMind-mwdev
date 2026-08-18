from __future__ import annotations

"""
Canonical result utilities and typed error hierarchy for the tuning subsystem.

Exports:
- TuningError         — base exception for all tuning failures.
- EngineNotAvailableError — raised when an optional dependency is missing.
- best_trial()        — convenience accessor for the best TrialRecord.
- merge_metadata()    — shallow dict merge (extra wins on conflict).

No sklearn / skopt / optuna imports in this module.
"""

from typing import Any

from pysrc.tuning.specs import TrialRecord, TuningResult

# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------


class TuningError(Exception):
    """Base exception for all failures originating in the tuning subsystem.

    Callers that need to catch any tuning failure should catch TuningError.
    Callers that need to distinguish specific failure kinds should catch the
    appropriate subclass.
    """


class EngineNotAvailableError(TuningError):
    """Raised when the optional dependency required by an engine is not installed.

    The error message names both the missing package and the engine that
    requires it, so users get an immediately actionable fix.

    Example:
        raise EngineNotAvailableError("bayes", "scikit-optimize (skopt)")
        # → "Engine 'bayes' requires scikit-optimize (skopt), which is not
        #    installed. Install it with: pip install scikit-optimize"
    """

    def __init__(self, engine: str, package: str) -> None:
        # Derive the install target from the package string: strip the
        # parenthetical alias portion so we surface the canonical PyPI name.
        install_target = package.split("(")[0].strip() if "(" in package else package
        super().__init__(
            f"Engine {engine!r} requires {package}, which is not installed. "
            f"Install it with: pip install {install_target}"
        )
        self.engine = engine
        self.package = package


# ---------------------------------------------------------------------------
# Convenience utilities
# ---------------------------------------------------------------------------


def best_trial(result: TuningResult) -> TrialRecord:
    """Return the TrialRecord with the best score in result.

    The "best" trial is the one that matches result.best_score exactly.
    If multiple trials share the best score, the first occurrence is returned
    (preserving evaluation order for determinism).

    Args:
        result: A fully-populated TuningResult.

    Returns:
        The TrialRecord whose score equals result.best_score.

    Raises:
        TuningError: If result.trials is empty.
        TuningError: If no trial matches result.best_score (should not occur
                     for well-formed results, but provides a clear failure mode).
    """
    if not result.trials:
        raise TuningError(
            f"best_trial(): TuningResult from engine {result.engine!r} contains "
            "no trials; cannot determine best trial."
        )

    for trial in result.trials:
        if trial.score == result.best_score:
            return trial

    # Defensive: should not happen for results produced by the canonical engines.
    raise TuningError(
        f"best_trial(): no trial in TuningResult from engine {result.engine!r} "
        f"matches best_score={result.best_score!r}. "
        "This indicates a malformed TuningResult produced by a non-canonical engine."
    )


def merge_metadata(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow merge of base and extra, where extra wins on conflict.

    Neither input dict is mutated.

    Args:
        base:  Initial metadata dict.
        extra: Override metadata dict.  Keys present in both dicts are taken
               from extra.

    Returns:
        A new dict containing all keys from base and extra.
    """
    merged = dict(base)
    merged.update(extra)
    return merged


# Re-export ObjectiveDirection for callers who only import from result.
__all__ = [
    "TuningError",
    "EngineNotAvailableError",
    "TuningResult",
    "best_trial",
    "merge_metadata",
]

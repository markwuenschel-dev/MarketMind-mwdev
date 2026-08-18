"""Warm-start utilities: inject prior experiment results into the search state."""

from __future__ import annotations

from pysrc.tuning.core.ir.search_ir import SearchIR, Trial


def inject_prior_trials(
    ir: SearchIR,
    prior_trials: tuple[Trial, ...],
) -> SearchIR:
    """Return a new SearchIR with prior_trials prepended to the trial history."""
    combined = prior_trials + ir.trials
    best = ir.best_trial_id
    if not best and prior_trials:
        best_t = max(
            prior_trials,
            key=lambda t: max(t.scores.values(), default=0.0),
            default=None,
        )
        if best_t is not None:
            best = best_t.trial_id
    return SearchIR(
        job_id=ir.job_id,
        algorithm=ir.algorithm,
        space_hash=ir.space_hash,
        meta=ir.meta,
        trials=combined,
        best_trial_id=best,
        n_pending=ir.n_pending,
    )


__all__ = ["inject_prior_trials"]

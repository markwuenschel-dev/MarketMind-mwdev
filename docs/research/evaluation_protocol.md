# Evaluation Protocol (Research Stage)

This protocol defines minimum evaluation discipline for prototype-stage evidence.

Canonical references:
- [`docs/src/PhaseIIArtifactContract.md`](../src/PhaseIIArtifactContract.md)
- [`docs/src/PhaseIIResearchExecutionPlaybook.md`](../src/PhaseIIResearchExecutionPlaybook.md)
- [`docs/src/Risk_and_Execution_Realism_Protocol.md`](../src/Risk_and_Execution_Realism_Protocol.md)

## Objective

Produce believable baseline-versus-challenger evidence under realistic assumptions, without over-claiming production readiness.

## Required Rules

1. **Time-aware splits only:** Use walk-forward or equivalent time-respecting evaluation. No random split claims for decision-grade results.
2. **Matched assumptions:** Baseline and challenger must run with the same split, data boundary, and execution assumptions.
3. **Execution realism declared:** Cost/slippage/latency/fill assumptions must be explicit in `execution_assumptions.json`.
4. **Artifact completeness:** Runs used for governance claims include `task_manifest.json`, `meta_validity_report.json`, and `execution_assumptions.json`.
5. **Non-promotable honesty:** Prototype-stage wins are research evidence, not production authorization.

## Default Metric Set

- Net Sharpe (after declared costs)
- Turnover-adjusted utility
- Drawdown profile
- Calibration/coherence diagnostics where applicable
- Regime-aware breakdowns where available

## Run Acceptance Checklist

- [ ] Baseline and challenger compared on identical data/split/assumptions
- [ ] Assumptions file emitted and linked
- [ ] Required artifact triple emitted and structurally valid
- [ ] Leakage checklist completed for the run
- [ ] Result classification documented in `experiment_registry.md`

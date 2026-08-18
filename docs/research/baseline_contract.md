# Baseline Contract (Prototype Stage)

This document defines how baseline-versus-challenger comparisons are made in the research stage.

It intentionally links to canonical contracts instead of duplicating them.

## Canonical References

- Artifact contract: [`docs/src/PhaseIIArtifactContract.md`](../src/PhaseIIArtifactContract.md)
- Threshold state authority: [`docs/src/ThresholdGovernanceRegister.md`](../src/ThresholdGovernanceRegister.md)

## Baseline Surfaces

- W1 baseline runner surface:
  - [`pysrc/meta/w1_baseline_runner.py`](../../pysrc/meta/w1_baseline_runner.py)
  - [`pysrc/meta/w1_baseline_incumbent.py`](../../pysrc/meta/w1_baseline_incumbent.py)
  - [`pysrc/meta/w1_baseline_io.py`](../../pysrc/meta/w1_baseline_io.py)
- W2 allocator benchmark surface:
  - [`pysrc/meta/allocator_benchmark/runner.py`](../../pysrc/meta/allocator_benchmark/runner.py)
  - [`pysrc/meta/allocator_benchmark/report.py`](../../pysrc/meta/allocator_benchmark/report.py)

## Comparator Rules

1. Baseline and challenger must share the same task pool / time split geometry.
2. Baseline and challenger must share the same cost/slippage/latency/fill assumptions.
3. Baseline and challenger must use identical data boundary assumptions.
4. Any exception must be declared as a validity risk in `research_risk_register.md`.

## Assumption Parity Contract

Use the explicit parity declaration carried by execution assumptions:
- [`pysrc/meta/execution_assumptions_config.py`](../../pysrc/meta/execution_assumptions_config.py) (`ExecutionParityDeclaration`)

Every run intended for baseline/challenger claims must point to the emitted `execution_assumptions.json` and record parity status in `experiment_registry.md`.

## Prototype Claim Boundary

- A run may support a research claim ("challenger appears stronger under declared assumptions").
- A run may not be treated as production approval from this contract alone.

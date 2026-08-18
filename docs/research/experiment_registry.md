# Experiment Registry

Lightweight index of research experiments and their evidence artifacts.

Source-of-truth remains committed code and emitted artifacts. This file is the navigation layer.

## Required Fields

| Field | Description |
|---|---|
| Experiment ID | Stable identifier for the experiment record |
| Date | Run date |
| Commit | Git commit used |
| Hypothesis ID | Link to `hypothesis_log.md` |
| Baseline ID | Baseline contract reference used |
| Assumptions Artifact | Path to `execution_assumptions.json` |
| Task Manifest | Path to `task_manifest.json` |
| Validity Report | Path to `meta_validity_report.json` |
| Outcome | PASS / FAIL / INCONCLUSIVE / NOT_EVALUATED |
| Notes | Short interpretation, no long narrative |

## Registry

| Experiment ID | Date | Commit | Hypothesis ID | Baseline ID | Assumptions Artifact | Task Manifest | Validity Report | Outcome | Notes |
|---|---|---|---|---|---|---|---|---|---|
| EXP-SEED-001 | 2026-05-05 | TBD | HYP-001 | W1/W2 baseline surfaces | TBD | TBD | TBD | NOT_EVALUATED | Seed entry |

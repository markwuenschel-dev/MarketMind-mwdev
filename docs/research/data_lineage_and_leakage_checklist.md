# Data Lineage and Leakage Checklist

Per-run checklist for prototype-stage data validity.

Canonical PIT and lineage doctrine:
- [`docs/src/DataGovernanceCharter.md`](../src/DataGovernanceCharter.md)

## Run Header

- Run ID:
- Date:
- Code ref (commit):
- Task manifest path:
- Meta validity report path:
- Execution assumptions path:

## Checklist

| Check | Pass/Fail | Evidence | Notes |
|---|---|---|---|
| Data access path uses governed as-of semantics (`DataView.as_of(T)` or governed equivalent) |  |  |  |
| Training/evaluation split boundaries are time-aware and non-overlapping |  |  |  |
| No train/eval leakage through features, labels, metadata, or post-hoc joins |  |  |  |
| Universe membership was resolved from governed records, not price availability shortcuts |  |  |  |
| Crisis holdout treatment matches declared protocol for this lane |  |  |  |
| Data snapshot / vintage identity is recorded and stable |  |  |  |
| Label timing / effective-at semantics are consistent with decision-time visibility |  |  |  |
| Any approximation or fallback is explicitly declared and justified |  |  |  |
| Leakage/property tests relevant to the run passed |  |  |  |

## Reviewer Signoff (Research Stage)

- Checked by:
- Date:
- Follow-up required:

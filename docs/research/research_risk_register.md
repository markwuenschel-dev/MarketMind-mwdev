# Research Risk Register

Focused risk register for prototype-stage research validity.

This is not an enterprise operational risk policy.

## Severity Scale

- `H`: High impact on evidence credibility
- `M`: Moderate impact
- `L`: Low impact

## Status

- `OPEN`, `MITIGATING`, `ACCEPTED_TEMPORARILY`, `CLOSED`

## Risks

| Risk ID | Risk | Severity | Current Controls | Trigger / Early Signal | Mitigation | Owner | Status |
|---|---|---|---|---|---|---|---|
| RSK-001 | Time leakage from non-governed data access paths | H | PIT doctrine + tests + governed data access contracts | Unexpected uplift with poor out-of-time behavior | Block non-as-of access and require checklist completion | TBD | OPEN |
| RSK-002 | Baseline/challenger mismatch in costs or split assumptions | H | Execution assumptions parity declaration | Challenger outperforms only under unmatched assumptions | Require parity flags and explicit assumption artifacts per run | TBD | OPEN |
| RSK-003 | Multiple-testing drift from too many iterative variants | M | Decision log + experiment registry | Frequent threshold tweaks after result review | Require declared hypothesis and success condition before run | TBD | OPEN |
| RSK-004 | Insufficient task diversity causing unstable conclusions | M | Task-level admissibility checks in current W1 governance | High variance across folds/tasks | Expand task pool before promoting conclusions | TBD | OPEN |

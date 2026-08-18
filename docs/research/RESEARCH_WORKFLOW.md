# Research Workflow (Prototype Stage)

This workflow defines lightweight governance for research commits. (The former
release-lane / governance-ledger protocol is retired — see ADR-004 in [`../DECISIONS.md`](../DECISIONS.md).)

## Use This Workflow When

- Work is exploratory or iterative research.
- No SemVer release is being authored.
- No architectural identity scheme or storage contract is being changed.

## Minimum Required Steps

1. Run targeted tests for changed behavior.
2. Ensure reproducibility anchors are captured (commit, config, key artifacts).
3. If the run is material, add/update one row in `experiment_registry.md`.
4. If methodology or policy interpretation changed, add a row to `decision_log.md`.
5. If a new validity risk appears, update `research_risk_register.md`.

## Escalate to Release Lane When

- Versioned release documentation is being published.
- Contract/schema identity changes are introduced.
- Governance claims depend on new formal manifest/trace outputs.

# Research Governance Hub

This folder is the lightweight governance layer for the research-prototype stage.

Use these files to answer one question quickly and consistently:
"Are we generating believable evidence that the model idea works?"

## Scope

- Focus on research validity, reproducibility, baseline/challenger discipline, and explicit research risks.
- Do not use this folder for production-compliance process, launch approvals, or enterprise incident doctrine.

## Canonical References

- Point-in-time and lineage doctrine: [`docs/src/DataGovernanceCharter.md`](../src/DataGovernanceCharter.md)
- Phase II required artifact surfaces: [`docs/src/PhaseIIArtifactContract.md`](../src/PhaseIIArtifactContract.md)
- Long-lived decision authority: [`docs/src/ResolutionLedger.md`](../src/ResolutionLedger.md)
- Threshold identity/state authority: [`docs/src/ThresholdGovernanceRegister.md`](../src/ThresholdGovernanceRegister.md)

## Files in This Folder

- [`hypothesis_log.md`](./hypothesis_log.md): hypotheses and outcome status
- [`data_lineage_and_leakage_checklist.md`](./data_lineage_and_leakage_checklist.md): per-run validity checks
- [`evaluation_protocol.md`](./evaluation_protocol.md): time-aware evaluation rules
- [`baseline_contract.md`](./baseline_contract.md): incumbent and parity contract
- [`experiment_registry.md`](./experiment_registry.md): experiment index linking runs to artifacts
- [`decision_log.md`](./decision_log.md): ADR-lite methodology decisions
- [`research_risk_register.md`](./research_risk_register.md): validity-focused risk register
- [`RESEARCH_WORKFLOW.md`](./RESEARCH_WORKFLOW.md): lightweight workflow for non-release research commits
- [`repo_storage_policy.md`](./repo_storage_policy.md): storage ownership, retention tiers, and size/path guardrails

## Update Policy

- Update `hypothesis_log.md`, `experiment_registry.md`, and the leakage checklist for each material experiment run.
- Update `decision_log.md` when methodology changes or constraints change.
- Update `research_risk_register.md` when a new validity risk is introduced, elevated, or retired.
- Keep entries append-only. Add new rows; do not rewrite history.

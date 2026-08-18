# Repo Storage Policy (Research Prototype)

This policy prevents repository bloat while preserving reproducibility.

## Canonical Ownership

- Committed research evidence belongs in `artifacts/` and must stay compact (Tier A/B only).
- Full run outputs belong in `runs/` and are local or external by default.
- Governance/protocol docs belong in `docs/src/` and `docs/research/`.
- Fixtures belong in `fixtures/` with one canonical fixture per scenario family plus generation recipes.

## Retention Tiers

- `TierA` (commit): reproducibility-critical metadata and compact evidence.
- `TierB` (optional commit): small debug subsets needed by tests/regressions.
- `TierC` (do not commit by default): full predictions, full opportunity tables, giant joined audits, raw debug dumps.

## Default Size Thresholds

- Warn on new committed files larger than 1 MB.
- Fail on new committed files larger than 5 MB unless allowlisted.
- Fail on new committed CSV/JSON/Parquet larger than 2 MB unless explicitly tagged as Tier A/B.
- Fail on any tracked cache/vendor/worktree path regardless of size.
- Require migration pointer metadata for removed tracked artifacts larger than 5 MB.

## Blocked Paths

The following must never be tracked:

- `.poetry-cache/`, `.venv*/`, `.mypy_cache/`, `.pytest_cache/`
- `.claude/worktrees/`, `.codex*/`, `.cursor/`
- `node_modules/`, `htmlcov/`, `.cache/`, `target/`
- `docs/out/` (build outputs)

## Lane Rules

- Research lane may commit: compact metric summaries, checksums, config snapshots, small index/manifest files, and `docs/research/experiment_registry.md` updates.
- Research lane should not commit: full prediction dumps, full opportunity manifests, full joined audit tables, or large debug dumps. (The release-ledger / traces / manifests / DOCX machinery is retired — see ADR-004 in [`../DECISIONS.md`](../DECISIONS.md).)

## Artifact Promotion Rule

A run output is not committed by default. It can be promoted only if all are true:

1. Needed for reproducibility, CI, regression testing, or published evidence.
2. It is the smallest useful representation.
3. It has an owner and retention tier.
4. It has a stable path and naming convention.
5. It does not duplicate existing artifacts.

## Artifact Migration Safety Rule

Do not delete/migrate a large artifact until a pointer record exists with:

- original path
- new location or retrieval instruction
- checksum/hash
- artifact type
- producing command or run id (if known)
- date migrated
- reason for migration

## Git History Policy

This policy does not rewrite git history by default. Any history rewrite is a separate explicit decision.

## Local Housekeeping

Use `python scripts/cleanup_local_sprawl.py` for dry-run cleanup estimation and
`python scripts/cleanup_local_sprawl.py --apply` to remove local worktree/cache/build sprawl.

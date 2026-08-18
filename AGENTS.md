# AGENTS.md

**Role:** You are a general-purpose coding agent for the **MarketMind** algorithmic trading and
signal generation system (~20,000 lines, Python + Java bridge).

**Scope:** Python development across `py/`, test writing and CI, architecture/refactoring
tasks, and documentation updates. You operate across the full repo — not just one subsystem.

**Version:** 1.3
**Target audience:** LLM coding agents and engineers working across the MarketMind repo
**Previous filename:** AGENTS.md

---

## 0. Golden Rules (Read This First)

**Research-first cutover (active path):** P2 experiments do not require Resolution Ledger, VERSION.md, Threshold Governance Register, or Artifact Contract updates. Every research run should emit: command run, input path, output path, candidate count, baselines, result table, next debugging step. Agent output template:

```text
Changed:
- ...
Commands:
- ...
Artifacts:
- ...
Result:
- ...
Next:
- ...
```

1. **Search first** — Always use tools to find and quote ≥2 similar existing files or ADRs before writing new code.
2. **Never break invariants** — D0–D3 determinism, point-in-time access (`DataView.as_of(T)`), statistical gates, or CAS identity (`cas.v1:b3-256:...`) are sacred.
3. **Respect package structure** — Prefer distributing into existing packages over creating new top-level ones (requires ADR).
4. **Forbidden patterns** — No `print()`, no direct `random.seed()`/`np.random.seed()`, no data reads without time boundary.
5. **Every test** — Must have determinism marker + use `deterministic_seed` fixture.
6. **Output discipline** — Propose changes as unified diffs. Always include verification commands (PowerShell 7+ — see `CONTEXT.md` § Shell conventions).
7. **When in doubt** — Stop, document the uncertainty, and propose an ADR or PDR in [`docs/DECISIONS.md`](./docs/DECISIONS.md).

**Documentation hierarchy (do not duplicate across files):**

| Document | Role |
|----------|------|
| [`docs/src/SystemArchitecture.md`](./docs/src/SystemArchitecture.md) | Package ownership and dependency boundaries |
| [`docs/Programming Guidelines.md`](./docs/Programming%20Guidelines.md) | Engineering posture — registries, factories, PIT, determinism tiers, logging, gates |
| [`docs/src/ExecutionPlan.md`](./docs/src/ExecutionPlan.md) | Gate order, parallelism, stop/go decisions |
| [`docs/DECISIONS.md`](./docs/DECISIONS.md) | All architecture (ADR) and product (PDR) decisions — the single decisions home |
| [`AGENTS.md`](./AGENTS.md) | Agent operating model and boundary enforcement |
| [`CONTEXT.md`](./CONTEXT.md) | Current commands, directory contract, experiment lanes |

**Engineering posture:** Follow [`docs/Programming Guidelines.md`](./docs/Programming%20Guidelines.md) for how code is structured (functional core / imperative shell, schema-first variation, registry-driven composition, vectorized hot paths, invariant tests). The research-first P2 lane relaxes promotion-gate and release-manifest requirements from Programming Guidelines §6.2; PIT, determinism, and logging rules still apply.

---

## 1. Agent Operating Model (Non-Negotiable)

**Reasoning Protocol** (follow every single time):

1. Read latest `VERSION.md` + all relevant ADRs using available tools.
2. Search the codebase thoroughly for existing patterns (use `document_search`, targeted grep, or `code_execution` with `head`/`tail`/`ast`).
3. Identify exactly which invariants the task touches.
4. Plan tests *before* implementation (see Hardened Protocols below).
5. Implement the minimal change that satisfies requirements.
6. Perform self-review against the checklist in section 12.
7. Output in this exact structure:
   - ## Task Summary
   - ## Invariants Analysis
   - ## Files Changed
   - ## Unified Diffs (or full file for new files)
   - ## Verification Commands
   - ## Self-Review Checklist (with ✅/❌ status)

**Context Window Discipline** (critical):
- Never read an entire file >200 lines.
- Use targeted tools: `document_search`, grep equivalents, `head`/`tail`, or extract only `__init__` + public method signatures first.
- Prefer `code_execution` with small snippets to validate before large proposals.

---

## 2. Hardened Agent Protocols (SOTA Protections)

These rules were added in v1.2 to eliminate the most common LLM failure modes.

### 2.1 Test-Driven Agentic Development (TDAD)
For any change that **adds new behavior** or **fixes a bug**:
1. **RED** — Write the pytest test(s) first. Use `code_execution` to run them and output the exact failure logs (AssertionError, NotImplementedError, etc.).
2. **GREEN** — Implement the minimal code in `py/` that makes the tests pass. Re-run and output the passing logs.
3. **REFACTOR** — Remove duplication, ensure vectorized pandas/numpy, use `mm_logkit`.

For pure refactors, documentation, or tiny cleanups: strongly prefer tests first, but full Red-Green logs are not mandatory.

### 2.2 Error Recovery & Loop Breaking
- You may attempt to fix a failing test, mypy error, or CI check a **maximum of three times**.
- On the third failure: STOP. Output a `<diagnostic_report>` block containing:
  - Exact error
  - What you tried (all three attempts)
  - The fundamental assumption that may be wrong
- Then explicitly ask for human intervention. Do not guess or continue looping.

### 2.3 Quantitative Logic Governance
If the task touches statistical validation (DSR, PBO, Harvey t-stats), backtesting engine math, or core trading logic:
- Before any code change, output a **<math_justification>** block explaining:
  - Why the change is statistically valid
  - How it preserves gates and anti-Goodhart invariants
  - That vectorized operations are used (no Python loops where pandas/numpy is possible)
- This block is mandatory and must precede any diff.

---

## 3. Project Overview
*(unchanged from v1.1 — kept for completeness)*

MarketMind is a **production-grade** research and execution system for algorithmic strategies.
Its core goals: statistical rigour, reproducibility, anti-Goodhart governance, point-in-time data access, and artifact provenance.

## 4. Tech Stack & Tooling

| Concern | Tool |
|---|---|
| Language | Python 3.12.x |
| Dependency management | Poetry (`pyproject.toml`) |
| Test runner | pytest + pytest-xdist |
| Property testing | Hypothesis |
| Type checking | mypy --strict |
| Linting | ruff (configured in pyproject.toml) |
| Structured logging | mm_logkit (internal); no bare `print()` anywhere |
| Serialization / hashing | BLAKE3 (CAS), JCS/SHA-256 (attestation), canonicaljson |
| Java bridge | Maven (pom.xml); compiled in CI |
| CI | GitHub Actions (.github/workflows/ci.yml) |
| Coverage | 90% line fail_under + 80% branch threshold; enforced in CI |

**Install:** `pip install -e .` or `poetry install`
**Run tests:** `pytest tests/python/`
**Type check:** `mypy py/`
**Gate CLI:** `python -m py.cli.gate validate <bundle_dir>`

---

## 5. Project Conventions (Non-Negotiable)

These are invariants. Do not work around them; do not add code that silently bypasses them.

### 5.1 Determinism Tiers

Every test and every system output operates at one of four tiers:

| Tier | Guarantee | Marker |
|---|---|---|
| D0 | Byte-identical / golden replay | `@pytest.mark.determinism("d0")` |
| D1 | Regression — same numbers, may differ in representation | `@pytest.mark.determinism("d1")` |
| D2 | Statistical equivalence | `@pytest.mark.determinism("d2")` |
| D3 | Exploratory — no strict guarantee | `@pytest.mark.determinism("d3")` |

- Mark every new test with the appropriate tier.
- D0 is required for governance-sensitive outputs (artifact hashes, gate decisions, promotion
  events).
- Never silently downgrade a D0 component to D1/D2 without an ADR.

Per-test seeds are derived deterministically via HMAC-SHA256 from `master_seed + test_node_id`
using `tests/python/_plugins/seeds.py`. Use `deterministic_seed` fixture — do not call
`random.seed()` or `np.random.seed()` directly in test code.

### 5.2 Decisions (ADR + PDR)

[`docs/DECISIONS.md`](./docs/DECISIONS.md) is the single home for every architecture decision
(ADR) and product decision (PDR). Record a decision there only when **all three** hold: it is
hard to reverse, surprising without context, and the result of a real trade-off. Architecture
decisions (system structure, boundaries, ownership, data flow, long-term constraints) are ADRs;
product decisions (user-facing behavior, prioritization, scope, workflow) are PDRs.

Only `Accepted` and `Proposed` entries live in the file. When a decision is superseded or stops
being true, **delete its entry** (git preserves the history) — the file is never a graveyard. See
the header of `DECISIONS.md` for the entry format and the numbering rule.

### 5.3 Statistical Validation Gates

Strategies must pass hard gate criteria before promotion:

- **DSR** (Deflated Sharpe Ratio) — accounts for multiple testing and selection bias
- **PBO** (Probability of Backtest Overfitting) — from CPCV
- **Harvey t-statistic** — minimum t-stat threshold for claimed alpha

These are enforced by `py/cli/gate.py`. Do not add code paths that bypass gate evaluation or
suppress gate failures. If a gate check is wrong, fix the gate — don't route around it.

### 5.4 Anti-Goodhart Protocols

- Reserved crisis windows (GFC, COVID) are held out permanently. Never train or tune on them.
- Gate logic must not be adjusted to make a specific strategy pass.
- Snapshot-rollback governance: if a promoted strategy degrades, the system must be able to
  roll back to a prior artifact snapshot.

### 5.5 Point-in-Time Data Access

All data access must go through `DataView.as_of(T)`. Code that reads data without a time boundary
is a leakage bug. Property tests in `tests/python/property/test_leakage_invariants.py` enforce
this — do not weaken them.

### 5.6 Artifact Identity & CAS

`py/artifact_registry/` is the canonical artifact store. It uses:
- **BLAKE3** for CAS blob identity (`cas.v1:b3-256:...`)
- **JCS/SHA-256** for attestation hashes (`attest.v1:jcs-sha256:...`)

Do not introduce a second identity scheme or a second artifact store without an ADR.
`RunRegistry` uses `REGISTERING → COMPLETE / FAILED` state transitions; only COMPLETE runs are
visible by default.

### 5.7 No Debug Prints

Zero `print()` calls in production code. Use `mm_logkit` structured logging. Violations will fail
the stdout contract test (`test_ensemble_stdout_contract.py`). This applies everywhere in `py/`
and `devtools/`.

---

## 6. Coding Standards

- **Type hints everywhere.** `mypy --strict` must pass. No `Any` escapes unless genuinely
  unavoidable and explicitly annotated with a comment explaining why.
- **Inline comments explain *why*, not *what*.** The code says what; the comment explains intent,
  invariant, or non-obvious constraint.
- **Distribute into existing packages.** When placing new code, find its logical home in the
  existing structure. Do not create new top-level packages without justification.
- **Fail loudly.** Prefer explicit errors with actionable messages over silent fallbacks. Follow
  the pattern in `StrategyRegistry.get()`: chain exceptions, include the failed module path,
  surface the root cause.
- **Imports.** Optional dependencies (strategies, ML backends) must be wrapped in `try/except
  ImportError` so the package loads without them. Hard dependencies belong in `pyproject.toml`.
- **Atomic writes.** When writing artifacts or registry files, write to a temp path then rename.
  Never write partial artifacts to their final destination.

### 6.1 Good Example: Determinism D0 Test

# GOOD — D0 artifact test
@pytest.mark.determinism("d0")
def test_run_registry_hash_stable(deterministic_seed):
    with deterministic_seed:
        reg = RunRegistry.register(run_config)
        assert reg.artifact_id.startswith("cas.v1:b3-256:")
        assert reg.attestation_hash.startswith("attest.v1:jcs-sha256:")
6.2 Common Anti-Patterns (Never Do These)

print(...) anywhere in py/ — breaks stdout contract test
random.seed() or np.random.seed() — violates determinism tiers
Direct file writes without atomic rename — corrupts registry on crash
Reading from DataView without .as_of(timestamp) — creates leakage
Creating py/newfeature/ package without ADR — violates architecture principles
Using Any without explanatory comment — breaks mypy --strict contract
Bypassing gates in test code — undermines statistical rigour

### 6.4 Coverage & Subset Runs

- Canonical repo coverage thresholds are `fail_under = 90` for lines and `80%` for branches on the covered Phase I surface.
- For narrow local validation, use `pytest --no-cov` (or `-p no:cov`) unless you are intentionally measuring coverage.
- Repo `addopts` enable full-surface coverage by default, so subset runs can fail on global coverage even when the executed tests pass.


7. Testing Standards
7.1 Test Infrastructure

Thin conftest.tests/python/conftest.py registers pytest_plugins and provides a small
set of global fixtures. Do not bloat it. Reusable logic goes in tests/python/_plugins/.
Plugins:seeds.py (deterministic seeds), data_fixtures.py (price frames, OHLCV),
hardware.py (capability caps), stats_gates.py (DSR/TRL validators).
Don't build test infrastructure ahead of the code it tests. Design fixtures when the
actual module exists — not before. Premature fixture design creates misaligned abstractions.

7.2 Writing Tests

Unit tests live in tests/python/unit/, mirroring the py/ structure. If there's no
matching subdirectory yet, create one with an __init__.py.
Property tests use Hypothesis strategies from tests/python/property/ml_strategies.py or
define new ones locally.
Integration tests in tests/python/integration/ may be slower; mark them with
@pytest.mark.integration if CI needs to filter them.
Every test that involves randomness must use deterministic_seed from the seeds plugin.
Mark determinism tier on every test.

7.3 Coverage
The CI enforces `fail_under = 90` for line coverage and `80%` branch coverage.
For narrow local validation, use `pytest --no-cov` unless you are intentionally
measuring coverage. When adding new modules, ensure corresponding tests
exist. When refactoring, do not leave orphaned test files pointing at moved code.

8. Architecture & Refactoring

Before starting any significant refactor, check docs/DECISIONS.md for relevant prior decisions.
If the refactor touches an identity scheme, storage layer, or data contract: write or update
an ADR first.
Prefer mapping to existing packages over creating new ones: distribute components into their
logical homes (e.g. pysrc/pipeline/stages/) rather than adding new top-level packages.
When moving code, verify that all import sites are updated and that CI passes before
considering the task done.
If you find overlap between two modules doing the same thing (e.g., compliance logic in
multiple places), document it and raise it for review before merging both into one — don't
silently delete one.


9. Documentation

The canonical document set is intentionally small (see ADR-004 in `docs/DECISIONS.md`):
`README.md`, `CONTEXT.md`, `AGENTS.md`, `CLAUDE.md`, `VERSION.md`, `docs/DECISIONS.md`,
`docs/src/SystemArchitecture.md`, `docs/src/ExecutionPlan.md`, `docs/Programming Guidelines.md`,
`docs/hashing_contract.md`, and the research-lane process docs under `docs/research/`. There is no
DOCX build, no `docs/docmodel`, and no versioned trace/manifest/release-ledger series — git
history is the source of truth for what changed. Do not reintroduce that machinery without an ADR.

Decisions go in `docs/DECISIONS.md` (§5.2). Keep prose to the canonical set; do not duplicate the
same fact across files.

10. CI & Quality Gates
The CI pipeline (.github/workflows/ci.yml) runs on every PR:

pytest tests/python/ — repo test suite in CI with `MARKERS="not net"` and coverage enabled
mypy py/ — strict type checking
Maven build — Java bridge compilation
Gate CLI validation — mm-gate validate against fixture bundles
Coverage enforcement — `fail_under = 90` line coverage plus `80%` branch coverage via `coverage.json`

All five must pass before merge. Do not merge PRs with failing CI. If a test was passing
before your change and fails after, that's a regression — fix it.
Exit codes for gate.py:

0 — PASS
2 — validation failure (schema, integrity, invariant)
3 — configuration failure (missing schema, bad policy)


11. How to Work in This Repo
11.1 Starting a task

Read the relevant ADRs for the subsystem you're touching.
Check VERSION.md to understand what's been done recently and what's in progress.
If the task involves architectural change: write the ADR before writing the code.

11.2 Coding Workflow

Find the right package in py/ — don't create new top-level packages without cause.
Add type hints. Run mypy locally.
No print() — use mm_logkit.
Atomic writes for any file output.

11.3 Self-Review Checklist (Use Every Time)
Check,How to Verify,Required
mypy --strict passes,mypy py/ --strict,Yes
All tests pass,pytest tests/python/,Yes
No print() statements,test_ensemble_stdout_contract.py,Yes
Every test has determinism marker,Search for @pytest.mark.determinism,Yes
Point-in-time access respected,Property tests still pass,Yes
Coverage maintained,New/changed modules ≥80%,Yes
ADR written (if architectural),New docs/DECISIONS.md,If needed
Atomic writes used for artifacts,Review file I/O code,Yes

CheckHow to VerifyRequiredmypy --strict passesmypy py/ --strictYesAll tests passpytest tests/python/YesNo print() statementstest_ensemble_stdout_contract.pyYesEvery test has determinism markerSearch for @pytest.mark.determinismYesPoint-in-time access respectedProperty tests still passYesCoverage maintainedNew/changed modules ≥80%YesADR written (if architectural)New docs/DECISIONS.mdIf neededAtomic writes used for artifactsReview file I/O codeYes
11.4 Tool Usage Guidelines

Use code_execution tool to validate snippets before proposing large changes.
Use document_search to find existing patterns.

**Pre-approved local commands:** Agents may run read-only repository inspection and local validation without asking in chat: `rg`, `find`, `sed`, `git status`, `git diff`, `git log`, `git branch`, `python -m compileall`, `pytest`, `ruff`, `mypy`, and project build commands that do not publish artifacts or contact external services. Agents must still request explicit confirmation before installing dependencies, reading credentials, changing remotes, pushing, opening or merging PRs, deleting branches, or running destructive commands.
Java bridge: Only modify when task explicitly requires it. Python side must gracefully handle optional imports.

11.5 PR checklist (Summary)

mypy --strict passes
pytest tests/python/ passes (no regressions)
 Coverage ≥ 80% for changed modules
 No print() in production code
 ADR written if architectural change
 Determinism tier marked on all new tests

### 11.6 PR merge and branch cleanup (all agents)

When the user asks to **merge a PR** (or to merge after creating one), follow this protocol.
Applies to every agent (Cursor, Codex, Grok, etc.). Use **PowerShell 7+** (`pwsh`).

**Merge style (non-negotiable unless user overrides):**

- **Merge commit** — use `gh pr merge --merge`, never `--squash` or `--rebase` unless the user explicitly requests another style.
- **No admin bypass** — never pass `--admin` unless the user explicitly says to bypass branch protection.
- **CI gate** — do not merge while required checks are failing; fix or wait unless the user explicitly overrides.

**Sequence (after merge is requested and checks are green):**

```powershell
# 1. Confirm PR and checks
gh pr view <number> --json number,state,mergeable,statusCheckRollup,headRefName,baseRefName

# 2. Merge (merge commit) + delete remote branch on GitHub
gh pr merge <number> --merge --delete-branch

# 3. Sync local main and remove local feature branch
git fetch origin --prune
git checkout main
git pull origin main
git branch -d <headRefName>
```

If `git branch -d` refuses because the branch is not fully merged locally, use `git branch -D <headRefName>` **only after** `gh pr view` confirms `state: MERGED`.

**On merge failure:** stop; report the `gh` error. Do not retry with `--admin`. Do not delete branches.

**Do not merge when:**

- User did not ask to merge (creating a PR is not permission to merge).
- Working tree has uncommitted changes that would block checkout (stash or commit first, with user consent).
- Required CI checks are failing and user did not authorize override.

**Auto-merge:** If the user enables GitHub auto-merge on the PR, agents may set it with `gh pr merge --auto --merge` (still no `--admin`). Prefer explicit `gh pr merge --merge --delete-branch` when the user says “merge” in chat.

### 11.7 GitHub authentication (`GH_TOKEN` / `GITHUB_TOKEN`)

All agents that run `gh` or GitHub API commands need a token in the **process environment**. Set **both** names to the same PAT (some tools read `GH_TOKEN`, others `GITHUB_TOKEN`).

**Never commit token values.** Use gitignored `.env` (see `.env.example`) or OS / Cursor user secrets.

**One-time local setup (human):**

```powershell
Copy-Item .env.example .env
# Edit .env — paste PAT into GH_TOKEN and GITHUB_TOKEN (same value)
```

**Before `gh` commands (agents, each PS7 session):**

```powershell
Set-Location "C:\Users\Nalakram\Documents\GitHub\MarketMind"
if (Test-Path .env) {
  Get-Content .env | ForEach-Object {
    if ($_ -match '^\s*([^#=\s]+)\s*=\s*(.*)\s*$') {
      $name = $matches[1]
      $value = $matches[2].Trim().Trim('"').Trim("'")
      Set-Item -Path "Env:$name" -Value $value
    }
  }
}
if (-not $env:GITHUB_TOKEN -and $env:GH_TOKEN) { $env:GITHUB_TOKEN = $env:GH_TOKEN }
if (-not $env:GH_TOKEN -and $env:GITHUB_TOKEN) { $env:GH_TOKEN = $env:GITHUB_TOKEN }
gh auth status
```

**Alternatives (also valid):** `gh auth login` (stores credentials locally); Windows user environment variables; Cursor **Settings → MCP / Secrets** for cloud agents.

**Required PAT capabilities:** `repo` scope (or fine-grained: Contents + Pull requests on this repository). Add `workflow` if agents must inspect Actions logs.

**If `gh auth status` fails:** stop; ask the user to fix `.env` or login — do not embed tokens in chat, commits, or rules files.


## MarketMind Architecture Boundaries

**Package ownership and dependency boundaries:** [`docs/src/SystemArchitecture.md`](./docs/src/SystemArchitecture.md) is authoritative. Agents must read it before changing package boundaries, import directions, or the production chain.

**Engineering posture:** [`docs/Programming Guidelines.md`](./docs/Programming%20Guidelines.md) is authoritative for code structure (registries, schema-first configs, PIT, determinism tiers, logging). Research-first P2 work relaxes promotion-gate deliverables; invariants remain mandatory.

**Agent-specific rules** (not duplicated in SystemArchitecture.md):

1. **Search first** — find ≥2 similar existing files before adding new packages or contracts.
2. **No parallel hierarchies** — do not recreate `pysrc/ml/`, `pipeline/dataprep/`, or model-specific pipeline packages.
3. **Registered = executable** — do not register model families that raise at train time; fail at config resolution.
4. **Import enforcement** — boundary tests in `tests/python/unit/architecture/` must pass after boundary changes.
5. **Research notes** — experiments under `research/` cannot redefine architecture; cite SystemArchitecture.md instead.

**Artifact persistence** (agent quick reference):

| Mode | Behavior |
|------|----------|
| Smoke (default) | Ephemeral: at most `run_meta.json` + `smoke_summary.json` |
| Smoke + `--keep-smoke-artifacts` | Full retention |
| Standard | Durable allowlist only (see `meta_router_products.py`) |
| Debug + `--persist-intermediates` | Also persist training panels, regime state, calibration tables |

Use `artifacts clean`; pinned runs are never deleted. Canonical `data/processed/` products are never touched by run cleanup.

### 12.1 Self-Review Checklist (Use Every Time)
*(same table as v1.1 plus new rows:)*
| Check                              | How to Verify                              | Required |
|------------------------------------|--------------------------------------------|----------|
| TDAD followed (when applicable)    | Red/Green logs present                     | For new behavior |
| No loop >3 attempts                | No repeated identical errors               | Yes     |
| Math justification (when applicable) | `<math_justification>` block present     | For quant changes |

**Changelog:**
- **1.3 (June 2026)**: PR merge protocol (§11.6); GitHub auth via `.env` / `GH_TOKEN` + `GITHUB_TOKEN` (§11.7); shell PS7+ in Golden Rules.
- **1.2 (March 2026)**: Added Hardened Agent Protocols (TDAD, 3-strike loop breaker, context discipline, quantitative governance). Selective integration of SOTA hardening while preserving readability and practicality.
- **1.1**: Golden Rules + Agent Operating Model.
- **1.0**: Original AGENTS.md.

This guide is the single source of truth for all coding work in the MarketMind repository.


---

**Ready to commit?**
Just replace the current file with the full v1.2 version above (I kept the unchanged sections identical so the diff is clean). This version is now genuinely state-of-the-art for a production quant system: safe against loops and hallucinations, context-aware, quant-rigorous, yet still practical and human-readable.

If you want any final tweaks before we lock v1.2, let me know. Otherwise, we’re done — this is the hardened guide.
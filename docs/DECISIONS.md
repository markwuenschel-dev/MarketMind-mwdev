# MarketMind Decisions (ADR + PDR)

This is the **single home** for every architecture decision (ADR) and product decision
(PDR). One file; read it top-to-bottom and you have the full set of decisions that currently
constrain the project.

> **Numbering restarted 2026-06-22.** Pre-demolition ADR-001..009 (a `py/`-era graph/storage/W1
> lineage) live only in git history and `archive/` and are **unrelated** to the numbers here.
> The code identifier `adr007_parity` (hashing) is also unrelated to any entry below.

## How to use this file

- **When to record a decision.** Add an entry only when *all three* hold: it is **hard to
  reverse**, it is **surprising without context** (a future reader would wonder "why on earth?"),
  and it was **a real trade-off** (genuine alternatives existed). If a choice is easily reversed,
  obvious, or had no alternative, do not record it.
- **ADR vs PDR.** System structure, boundaries, ownership, data flow, long-term constraints →
  **ADR**. User-facing behavior, prioritization, scope, workflow choices → **PDR**. When a change
  is both, write the ADR (structure) and reference it from a PDR (product) only if the product
  call is itself non-obvious.
- **Format.** Required on every entry: a `## ADR-NNN · Title` heading and a
  `Status: Accepted|Proposed · DATE` line, followed by a short **Decision + why**. Optional
  sections (Context, Options, Consequences for ADRs; Scope, Horizon/Gate for PDRs) are included
  only when they earn their place — a trivial decision is three lines.
- **Lifecycle.** Only `Accepted` and `Proposed` entries live here. When a decision is superseded
  or stops being true, **delete its entry** (git preserves the history); the replacement, if any,
  notes "supersedes the prior X decision." This file is never a graveyard.
- Numbering is two independent dense sequences (`ADR-NNN`, `PDR-NNN`), newest on top within each
  section. Scan for the highest number and increment.

---

## Architecture Decisions (ADR)

## ADR-007 · AWS operational topology and immutable execution evidence
Status: Accepted · 2026-06-22

**Decision:** AWS is the initial production cloud. Kotlin/Spring services run as immutable containers on ECS Fargate; PostgreSQL on RDS owns transactional operational state; S3 owns the encrypted, versioned, retention-locked evidence archive; and AWS Secrets Manager owns service and broker credentials. Infrastructure changes are reviewed, versioned infrastructure-as-code. EKS is out of the initial release.

**Why:** The system needs one operational boundary for Cognito, services, transactional state, evidence, secrets, monitoring, and rollback. Kubernetes would add platform complexity before the paper-to-live execution lifecycle proves it is needed.

**Consequences:** Clients never receive broker, vendor, or AWS credentials. Deployments promote immutable images through dev, paper, and live with explicit approvals and rollback. The local paper-execution host uses the same contracts, so moving it to AWS changes topology—not behavior.

## ADR-006 · Execution and inference are isolated adapters
Status: Accepted · 2026-06-22

**Decision:** IBKR is the sole launch broker, accessed only through a broker-neutral Kotlin adapter. The adapter runs beside IB Gateway on a dedicated local Windows paper-trading host first, then on an isolated Windows EC2 execution host at scaled production; clients never connect to IBKR. Python remains the research and training runtime. A separate, versioned C++/Julia inference layer evaluates approved model artifacts through an internal interface; Kotlin does not load model files or embed quantitative inference logic.

**Why:** Broker protocols, Gateway authentication/recovery, and inference runtimes are volatile integration concerns. They must not leak into strategies, public APIs, or clients.

**Consequences:** Only the IBKR adapter translates canonical orders to broker requests. The inference interface carries versioned inputs, outputs, and artifact identity. IBKR market data is the execution source of truth; research may retain separately governed historical/vendor inputs.

## ADR-005 · Cross-platform clients are untrusted; Kotlin/Spring is the product boundary
Status: Accepted · 2026-06-22

**Decision:** JavaFX/FXML is retired from the forward product path. The existing Next.js product codebase is the single responsive client codebase for web, desktop, and mobile; Tauri packages its static client build for desktop and mobile. Kotlin/Spring is the sole product backend: it owns the OpenAPI-first REST/JSON command and read API, WebSocket event stream, authorization, risk, approvals, order orchestration, audit, and broker/inference integration. Next.js route handlers are not a product backend.

**Why:** Desktop, mobile, and web require the same trusted execution boundary. Tauri cannot depend on server-rendered Next.js behavior, and trading authority must not live in clients.

**Consequences:** Next.js uses Amazon Cognito authorization-code + PKCE and a generated TypeScript API client. Kotlin validates JWTs and enforces roles. Offline clients may show clearly stale cached data but cannot submit, approve, override, cancel, or queue orders.

## ADR-004 · Single canonical doc set; retire the documentation machinery
Status: Accepted · 2026-06-22

**Decision:** The canonical documents that describe MarketMind are exactly: `README.md`,
`CONTEXT.md`, `AGENTS.md`, `CLAUDE.md`, `VERSION.md`, `docs/DECISIONS.md`,
`docs/src/SystemArchitecture.md`, `docs/src/ExecutionPlan.md`, `docs/Programming Guidelines.md`,
`docs/hashing_contract.md`, and the active research-lane process docs under `docs/research/`.
The DOCX/governance documentation machinery is **retired**: the `devtools/docs/` builder, the
`docs/docmodel/` model, `docs/manifests/`, `docs/traces/`, `docs/diagrams/`, the
`docs/archive/governance/` release ledger, and the coupled `tests/.../docs/` suites are deleted.

**Why:** ADR-002's demolition found that the documentation meant to preserve rationale recorded
*ceremony* instead (299 files / 2.8 MB, a 20-version Resolution Ledger, 31 traces, change
manifests). `VERSION.md` already declared the release ledger dead and `CONTEXT.md` already
stopped blessing the DOCX suite; the build machinery was the unfinished half of that demolition.
A documentation system that outruns one-screen explainability is the exact failure ADR-002
named. Decisions now live here; everything else is either operational SSOT (CONTEXT,
SystemArchitecture, ExecutionPlan, Programming Guidelines) or git history.

**Consequences:**
- No versioned trace/manifest series and no generated DOCX. The living trace map is the README.
- `AGENTS.md` §8 (release protocol) and §9 (DOCX suite) are removed. The CI storage guardrail's
  block on adding `docs/traces|manifests` paths is **kept** — it now guards against silently
  reintroducing the machinery.
- A structural guard test asserts `DECISIONS.md` stays well-formed (IDs unique; every entry
  `Accepted`/`Proposed`), replacing the brittle phrase-pinning doc tests that were deleted.

## ADR-003 · `meta_learning` is generic substrate; `meta` is the trading policy layer
Status: Accepted · 2026-06-21

**Decision:** Two layers, one direction. `meta_learning/` is the **domain-agnostic ML substrate**
(MetaTask/inference contracts, context encoders, regime vocabularies, generic task registries,
embedding/coherence diagnostics, generic inference boundaries — it knows nothing about trading).
`meta/` is the **trading-specific policy layer built on the substrate** (gating & policy
selection, mixture-of-experts, abstention, exposure scaling, trading decision constraints,
Reptile task-adaptation for trading policies, regime/task definitions with trading semantics).
**Enforced dependency rule:** `meta/` may import `meta_learning/`; `meta_learning/` must **never**
import `meta/` (guarded in `tests/python/unit/architecture/import_boundary.py`).

**Why:** The pair (~17k loc) *looked* like a forbidden parallel hierarchy only because the
boundary was never recorded. It is a deliberate two-layer split, and `meta_learning/` already
imports `meta/` zero times — the direction is already correct.

**Consequences:** The cluster stays a bucket-2 keep (ADR-002), but as two clean layers. A
one-time re-shelving makes the boundary exact — sequence it only when the cluster is revived for
the policy product (PDR-002), not before: dedupe `task_generator.py` to the substrate; fold
`meta/{task,task_registry}.py` into `meta_learning`; split `regime_config/regime_labeler` so only
trading-semantic regimes stay in `meta`; move generic `bocpd_service.py`/`curriculum.py` down. The
import-boundary test is the long-term guard.

## ADR-002 · Demolish-in-place, not greenfield — three-bucket keep-list
Status: Accepted · 2026-06-21 · (formerly ADR-005)

**Decision:** Demolish in place. Keep the verified data foundation plus anything with a named,
recorded reason; delete the rest. The keep/cut test is **"can you name why it's here?"** — not
"is it on the active path?", because some breadth is intentional. Three buckets:
1. **Active** — traceable from `panel_flow.py → train_model_matrix → models → result`, plus the
   canonical panel and `products.py`. Reason = load-bearing now.
2. **Justified keep** — off-path but kept deliberately, *with a recorded use + rough horizon that
   connects to the active product*. "Might need it someday" does not qualify.
3. **Delete** — off-path with no nameable use: accidental accretion, empty/duplicate packages,
   retired plumbing.

Every surviving file carries a one-line reason. Process rule going forward: **nothing — code or
docs — outruns one-screen explainability.**

**Why:** The project lost comprehension — two mirror sprawls (27 top-level packages, 7
orchestration entry points, retired W2/W3/W4 lanes, 123 run dirs; and 299 governance-doc files).
Root cause is *process*, not architecture: each experiment was a fresh agent-written script with
its own I/O root and nothing was ever deleted. Greenfield resets the code but not the process, so
it reproduces the sprawl while discarding hard-won knowledge. Decisive fact: the data foundation
is sound (`data/processed/full_indicator_feature_panel/panel.parquet`, rebuilt 2026-06-21,
19,227 instruments / 12.1M rows / 31 indicators / 1-day forward-return supervision).

**Options:** (1) greenfield from zero; (2) finish the in-place cutover as-is; (3) demolish-in-place
— chosen.

**Recorded keep-list:**

| Keep | Bucket | Reason (use) |
|---|---|---|
| `pipeline/` (panel, products, canonical_data, dataprep), `models/`, `artifact_registry/`, `cli/`, `contracts/`, `data/`, `ops/` logging | 1 active | the live panel-matrix product |
| `backtesting/` | 2 | predictions → performance; execution layer for the next product |
| `portfolio/` | 2 | predictions → positions seam — re-wire onto the panel flow |
| `strategies/` | 2 | research portfolio — re-wire onto the panel flow |
| `meta/` + `meta_learning/` | 2 | regime-aware policy/allocation layer on the panel matrix; internal boundary = ADR-003; product = PDR-002 |
| Kotlin bridge | 2 | planned execution layer |

**Deleted (bucket 3):** `autotune/` (duplicate of `tuning/`), `config/` + `preprocessing/` (empty),
`utils/` (3 loc), the retired `pipeline/meta_router/` plumbing and `meta/allocator_benchmark/`
W2/W3/W4 lanes; collapse the 7 orchestration entry points to 1.

**Consequences:**
- Demolition is surgical (prune stubs, retire dead plumbing, consolidate orchestrators), not
  "raze 80%." `strategies/` and `portfolio/` import *through* the retired `meta_router` lane, so
  they were re-homed onto the panel flow **before** that lane was deleted.
- **Shared horizon:** every bucket-2 keep is gated on the same milestone — the panel product
  validating. That makes **validating the panel product the single critical path.** If it fails
  validation, the bucket-2 keeps fall to bucket 3.
- Retired-lane findings remain historical/unverified until re-derived on the active flow.
- Execution detail (the staged deletions, ~−14,885 lines, byte-identical extract-then-delete
  moves) is in git history.

## ADR-001 · Active product-flow and artifact boundaries
Status: Accepted · 2026-06-20

**Decision:** Make the active product flow exclusive and move all run ownership to
`pysrc.artifact_registry`. `pysrc.contracts` owns the neutral `StandardizedPredictionArtifact`
and `StandardizedTradeIntentArtifact` schemas (lineage semantics, no storage concepts).
`pysrc.artifact_registry` owns roles, run state, CAS and attestation identities, storage,
allocation, materialization, and `ResolvedArtifact[T]`. Strategies consume only a COMPLETE
prediction run via `--source-run-id`; candidate portfolios consume only a COMPLETE trade-intent
run the same way. **No public execution command accepts `--output-dir`.** Backtesting retains
ownership of `PortfolioTargetPlan`.

**Why:** The active CLI exposed retired P2/router/W3/W4 compatibility commands that allocated
caller-selected output dirs and trained upstream products implicitly — an ambiguous product flow
and a second artifact-ownership surface in `pysrc.artifacts`. The production chain needs typed,
point-in-time handoffs and a single owner of run state and identity.

**Options:** (1) retain compatibility commands and translate paths; (2) keep `pysrc.artifacts` as
a shared facade; (3) exclusive active flow with single ownership — chosen.

**Consequences:** Retired CLI registration and router compatibility imports are removed; registry
cleanup is reference-safe and defaults to a deterministic dry run. *(Update 2026-06-21, ADR-002
demolition: the hollow `meta-router` command group — it emitted empty `targets` — was removed
along with the `pipeline/meta_router/` package.)*

---

## Product Decisions (PDR)

## PDR-004 · Owner-controlled autonomous trading and exceptional overrides
Status: Accepted · 2026-06-22

**Decision:** The first live product serves owner/proprietary accounts only. `owner`, `viewer`, and `service` are the initial roles; only the owner can activate a strategy, change risk policy, make a paper-to-live decision, or override a rejected order. Every order passes the risk decision service, but the owner may explicitly send a rejected order after fresh authentication and a reason. An override applies to one order only and records rejected limits, proposed order, identity, reason, and timestamp immutably.

**Why:** The owner retains final authority without making normal risk checks invisible. This is a proprietary autonomous-trading product, not a third-party account or custody product.

**Scope:** Initial equity/ETF operation is cash-only and long-only, using limit or marketable-limit orders. Margin, shorts, and later instruments need separate policy enablement. Risk limits are versioned owner-approved policy changes, never silent UI edits. Privileged actions and every override require passkey or MFA.

**Horizon / Gate:** The kill switch immediately blocks new orders. Working-order treatment is an explicit, versioned risk-policy choice. Critical execution, broker, risk, stream, host, or reconciliation failures notify the owner by push plus SMS/email. Execution evidence is retained for at least seven years.

## PDR-003 · Real-time paper trading precedes owner-authorized live trading
Status: Accepted · 2026-06-22

**Decision:** The first execution release is real-time IBKR paper trading: it consumes live broker market/order streams, evaluates only completed five-minute bars, submits to an IBKR paper account, and continuously reconciles MarketMind with IBKR orders, fills, positions, and cash. The first active strategy is one candidate that has passed governed research gates; it runs on US equities and ETFs. Capability sequence: US options, futures, FX, international equities, then crypto. Combining eligible strategies waits for a separate portfolio-allocation product.

**Why:** Paper trading must rehearse operational execution, not merely replay simulated fills. Completed bars preserve point-in-time inputs and deterministic replay; one strategy makes accountability and attribution tractable.

**Scope:** Paper promotion evidence requires at least 60 trading days and 100 submitted orders, continuous order/fill reconciliation, position/cash reconciliation every completed five-minute bar, zero critical risk-control failures, and tested kill-switch/recovery procedures. A mismatch blocks new orders. These gates inform, but do not replace, the owner's final live go/no-go decision.

## PDR-002 · Regime-aware policy/allocation layer (future product)
Status: Accepted · 2026-06-23 (gates met: PDR-001 complete, `panel policy-smoke` POC)

**Decision:** The intended product is a **regime-aware policy/allocation layer** on top of the
panel matrix: it consumes the panel product's predictions and decides routing, position sizing,
exposure, and abstention across market regimes (BOCPD regime detection + gating /
mixture-of-experts / Reptile policy adaptation, on the `meta_learning` substrate).

**Why:** The `meta/` + `meta_learning/` cluster is kept as bucket-2 (ADR-002) and layered as
substrate vs trading policy (ADR-003). This record names the *product* that justifies the keep,
so it is not a hoard.

**Scope:** Out now — no implementation work; the cluster stays dormant but kept, with the ADR-003
boundary as its internal contract. In, when revived — the re-shelving/dedup in ADR-003, then build
the policy layer against validated panel predictions.

**Horizon / Gate:** Gated, in order, on (1) the panel product being **validated** (the single
critical path, ADR-002) and (2) the productionization chain (`backtesting → portfolio →
strategies`, PDR-001) being in place. Further out than PDR-001; no active work until then. If the
panel product fails validation, this product and its cluster fall to bucket 3.

## PDR-001 · The active product is the walk-forward panel model matrix
Status: Accepted · 2026-06-21

**Decision:** The product is the **walk-forward panel model matrix** — train a matrix of models
across walk-forward folds on the canonical indicator panel and emit a per-(ticker, date)
`model_prediction_panel.parquet`. **Product behavior:** predicts **1-day-ahead returns**
(supervision target `forward_return_horizon`, horizon = 1). **User-facing surface (CLI):**
`dataprep run` (build the panel) and the `panel` command group (`train`, `train-matrix`,
`audit-*`, `investigate-targets`, `probe-persistence`); entry `pysrc/cli/panel_flow.py`.
**Prioritization rider (ADR-002):** finish demolition and comprehension recovery of this one
product *before* adding any new research product — new products require a new PDR.

**Why:** The repo had accumulated several competing research products (the date-level
meta-router; the W2/W3/W4 allocator-benchmark lanes) and the owner could no longer say which one
*was* the product. The one-product-flow cutover (ADR-001) and the demolition (ADR-002) require a
single, named, user-facing product surface.

**Scope:** In — panel model-matrix training + diagnostics on the canonical panel. Out / retired —
the date-level meta-router product and the W2/W3/W4 allocator benchmarks (historical
comparator/provenance only).

**Consequences:**
- Anything not serving this product is a demolition candidate (ADR-002 keep-list).
- **Production endgame:** once validated, the panel product's predictions feed `backtesting →
  portfolio → strategies` into production — strategies are the production surface. Gated on
  panel-product validation; until then this chain is dormant.
- The 1-day horizon is current product behavior, not a permanent law; changing it is a new PDR.

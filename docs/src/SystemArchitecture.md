# MarketMind System Architecture

**Status:** Authoritative single source of truth for package ownership and dependency boundaries.
**Scope:** The data → model → prediction → candidate-portfolio chain and the packages around it.

## Governing rule

```text
SystemArchitecture.md      defines what each package owns
Programming Guidelines.md  defines how code is structured
AGENTS.md                  defines how agents must respect those boundaries
CONTEXT.md                 defines the current operational state and commands
docs/DECISIONS.md          records the architecture (ADR) and product (PDR) decisions
Research notes             describe experiments; they cannot redefine architecture
```

Package READMEs, CLI help, and research notes may summarize this document. They must not redefine
package ownership or dependency boundaries independently.

> **Post-demolition note (ADR-002).** The repo was demolished to one product flow. The active
> product is the **walk-forward panel model matrix** (PDR-001). The former *local MetaRouter*
> orchestration package (`pysrc/pipeline/meta_router/`) and the old `pysrc/pipeline/router/` /
> `pysrc/meta/allocator_benchmark/` comparator lanes are **deleted**. The allocation/routing role
> they were meant to fill is the **future regime-aware policy layer** (PDR-002), which is
> *dormant* until the panel product validates. Some modules still carry the `meta_router` name as
> shared schemas/product IDs used by the active flow — that naming is tracked debt (PDR-001), not
> a live orchestrator.

---

## 1. Architectural intent

MarketMind is one production chain with explicit typed boundaries. The **active** chain:

```text
incoming market / macro / alternative data
→ PIT-safe ingestion
→ cleaning and preprocessing
→ canonical processed products (data/processed/)
→ ticker-level model-matrix training and inference
→ standardized prediction panel
→ candidate portfolio construction
```

Downstream of candidate portfolios — allocation/sizing, `PortfolioTargetPlan`, and economic
backtesting — is **kept but dormant** (ADR-002 bucket-2), to be re-wired onto the panel flow once
the product validates. The allocation layer itself is the future regime-aware policy product
(PDR-002), built on `pysrc/meta/` + `pysrc/meta_learning/`.

---

## 2. Active production flow

```text
1. DATA INGESTION            pysrc/pipeline/stages/market_data/ , pysrc/data/
                             SIP, FRED, fundamentals, alt-data, crypto, broker
                                   ↓ PIT-safe / as-of access
2. CLEANING + PREPROCESSING  pysrc/preprocessor/ ,
                             pysrc/pipeline/stages/cleaning/ , .../preprocessing/
                             (indicators now live under stages/preprocessing/indicators/)
                                   ↓
3. CANONICAL DATA PRODUCTS   pysrc/pipeline/materializers/ , products.py ,
                             dataprep_runtime.py  →  data/processed/
                                   ↓
4. MODEL-MATRIX TRAINING     orchestration: pysrc/pipeline/panel/
                             algorithms: pysrc/models/   selection: pysrc/tuning/
                             output: standardized model_prediction_panel.parquet
                                   ↓
5. CANDIDATE PORTFOLIOS      pysrc/pipeline/candidate_portfolios/ , pysrc/strategies/ ,
                             pysrc/portfolio/
                             predictions → strategies → TradeIntent → candidate positions
```

```text
DORMANT / kept (ADR-002 bucket-2, re-wire onto the panel flow before use):
  allocation layer (future PDR-002)  →  PortfolioTargetPlan
  →  pysrc/backtesting/  (fees, slippage, turnover, ledger — economic truth)
  →  pysrc/infra/brokers/  (execution)
```

A separate macro / alternative-data ensemble path is **future** (no active implementation): macro
features under `pysrc/pipeline/stages/cleaning/features/` would feed a macro-state product used as
optional context by the allocation layer.

---

## 3. Package ownership and boundaries

### 3.1 `pysrc/pipeline/stages/market_data/`, `pysrc/data/`
Source adapters; market/macro/fundamental/alt/crypto ingestion; point-in-time access; valid-time
and knowledge-time enforcement; corporate-action-aware handling.

> All downstream model, strategy, and backtest inputs must enter through the point-in-time data
> boundary (`DataView.as_of(T)`). Raw source files must not be read directly by models, policy
> selectors, or evaluators.

### 3.2 `pysrc/preprocessor/`, `pysrc/pipeline/stages/cleaning/`, `.../stages/preprocessing/`
Cleaning, normalization, anomaly/missing handling, time alignment, technical/macro/sentiment/alt
features, scaling, causal sequence construction, PIT-safe joins. `pysrc/preprocessor/` owns the
implementations; the `stages/` packages only resolve config, invoke plans, connect canonical
products, and emit stage artifacts — they do not define independent transformations. They do not
own model selection, allocation, or backtest execution.

### 3.3 `pysrc/pipeline/materializers/` and canonical products
Canonical product definitions, validated materialization, schemas, grain checks, lineage, atomic
persistence. Canonical data lives under `data/processed/` (e.g. `canonical_daily_panel`,
`full_indicator_feature_panel`, `target_label_panel`, `fold_split_panel`), unique by
`instrument + date + interval`. A research-run output is not canonical merely because it is a
Parquet file.

### 4. Model layer — `pysrc/models/`
`pysrc/ml/` no longer exists; its contents were absorbed here. `pysrc/models/` is the single model
domain: `base.py` (the `PanelModel` contract), `registry.py` (executable registry), `tabular.py`
(linear/tree/boosting), `mlp.py`, `lstm.py` (consolidated; not yet an executable `PanelModel`),
`runtime/torch.py`, `datasets/timeseries.py`.

**Registry rule: `registered family = executable family`.** A family must not be accepted by
production config if it only raises `NotImplementedError` later. Planned families may live in a
capability catalog but must not be runnable registry entries until they pass construction, fit,
predict, save/load, determinism, and panel-integration tests.

### 5. `pysrc/pipeline/panel/`
Orchestrates base-model training and inference: feature-universe discovery, eligibility reporting,
fold and target construction, tabular encoding, causal sequence materialization
(`sequence_data.py`), registry-driven model-matrix execution (`train_model_matrix.py`),
standardized prediction products, and optional tuning via `pysrc/tuning/` before the final refit.
May depend on `pysrc/models/`, canonical products, and artifact/run infrastructure. Must not depend
on backtesting engine internals. `pysrc/tuning/` may consume model public contracts; models must
not depend on tuning.

### 5.1 `pysrc/strategies/`
Reusable alpha-decision layer: named strategy rules, their input contracts, and conversion of
features or predictions into typed pre-sizing `TradeIntent` rows. Does not own sizing, target
weights, artifacts, or backtest execution.

### 6. Candidate portfolio layer — `pysrc/pipeline/candidate_portfolios/`, `pysrc/portfolio/`
Convert predictions / `TradeIntent` into target positions; lightweight portfolio construction;
turnover; research-stage cost/liquidity/capacity; standardized candidate outputs. This makes models
economically comparable. `pysrc/portfolio/` is a lightweight calculation domain; final economic
truth stays in `pysrc/backtesting/`. Must not own production backtesting engines or broker
execution.

### 7. Meta packages (kept, dormant — ADR-003)
- **`pysrc/meta/` = trading policy layer:** policy selection, gating, mixture-of-experts,
  abstention, exposure scaling, decision constraints, Reptile adaptation, trading-semantic regime
  definitions.
- **`pysrc/meta_learning/` = generic ML substrate:** meta-task/inference contracts, context
  encoders, regime vocabulary, generic task registries, coherence diagnostics.
- **Enforced direction:** `meta/` may import `meta_learning/`; `meta_learning/` must **never**
  import `meta/` (guarded by `tests/python/unit/architecture/`). These are the substrate for the
  future PDR-002 allocation product and get no active work until it is revived.

### 8. Backtesting and the portfolio handoff
`pysrc/backtesting/` is authoritative for fills, fees, slippage, latency, borrow, liquidity,
ledger, settlement, corporate actions, and risk constraints. The handoff contract
`PortfolioTargetPlan` lives in `pysrc/backtesting/contracts/portfolio_target.py` (target weights,
cash, exposure, decision timestamp, lineage, confidence, constraints). Its former builder adapter
lived in the deleted `pipeline/meta_router/`; wiring an allocation layer to this contract is future
PDR-002 work.

---

## 9. Contracts inventory

| Concern | Canonical module |
|---------|------------------|
| Strategy intent | `pysrc/contracts/trade_intent.py` |
| Candidate spec | `pysrc/contracts/candidate_spec.py` |
| Standardized product artifacts | `pysrc/contracts/product_artifacts.py` |
| Feature channels (incl. macro state) | `pysrc/contracts/feature_channel.py` |
| Shared policy/decision schemas (`meta_router`-named, used by the panel trainer + `meta/`) | `pysrc/contracts/meta_router.py` |
| Canonical product IDs / durability | `pysrc/pipeline/meta_router_products.py` |
| Portfolio handoff | `pysrc/backtesting/contracts/portfolio_target.py` |
| Artifact registry, CAS, attestation, persistence | `pysrc/artifact_registry/` |

The two `meta_router`-named modules are shared schemas/product-IDs still consumed by the active
flow; renaming them to panel-native names is tracked debt (PDR-001). Boundary enforcement tests
live under `tests/python/unit/architecture/`.

---

## 10. Dependency rules

Allowed:

```text
pipeline/panel             → models, canonical products, artifact_registry
pipeline/candidate_portfolios → standardized prediction contracts, portfolio calculations
meta                       → meta_learning, shared contracts
backtesting                → shared contracts
```

Forbidden:

```text
models / meta / meta_learning / pipeline/panel / pipeline/candidate_portfolios
    ✗ depend on any reintroduced router/allocator orchestration without an ADR
meta_learning              ✗ import meta
backtesting                ✗ depend on allocation-layer implementation details
any package                ✗ read raw market-data adapters outside the PIT boundary
```

Shared contracts may be imported without importing implementation modules.

---

## 11. Configuration and artifacts

- Tracked experiment specs live under `research/p2/configs/`.
- Generated run outputs live under `artifacts/runs/<run_id>/`; canonical data does not live there.
- Persistence: smoke runs are ephemeral by default; durable real-run products are allowlisted;
  intermediate panels require explicit persistence. No new uncontrolled artifact root may be created.

---

## 12. Extension rules

- **New model:** one implementation module under `pysrc/models/` + registry entry + YAML config
  entry + `PanelModel` tests + panel integration tests. Do not create a model-specific pipeline
  package.
- **New experiment:** one YAML under `research/p2/configs/`. Do not create a phase/version source
  directory.
- **New data modality:** source adapter + preprocessing feature channel + canonical product/schema.
- **New durable output:** canonical product ID + one registry entry + schema reference + durability
  class + an active consumer + tests.

---

## 13. Current status

**Implemented:** one model domain under `pysrc/models/`; executable registry (tabular + MLP);
consolidated LSTM source; causal sequence-materialization boundary; canonical panel path; candidate
portfolio stage; architecture boundary tests; ephemeral smoke-artifact policy.

**Not yet complete (and gated on the panel product validating — ADR-002, PDR-001):** LSTM/GRU/TCN/
Transformer/etc. as executable `PanelModel` families; re-wiring `backtesting`/`portfolio`/
`strategies` onto the panel flow; the regime-aware allocation layer (PDR-002); renaming the residual
`meta_router` shared modules; a real-data end-to-end pilot through backtesting.

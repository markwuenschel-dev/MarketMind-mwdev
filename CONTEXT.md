# MarketMind

Governed algorithmic trading **research** platform. One data path through `pysrc/pipeline/`. Experiments write to `artifacts/`. No script sprawl.

This file is the canonical vision and vocabulary for **operational state and commands**. If another doc disagrees on day-to-day execution, this file wins.

**Architecture SSOT:** [`docs/src/SystemArchitecture.md`](docs/src/SystemArchitecture.md) — package ownership and dependency boundaries. Research notes cannot redefine it.

**Agent policy:** [`AGENTS.md`](./AGENTS.md) — how agents must respect those boundaries.

**Engineering posture:** [`docs/Programming Guidelines.md`](docs/Programming%20Guidelines.md) — how code is structured (registries, factories, PIT, determinism, logging). Promotion-gate requirements are relaxed on the active P2 research lane; invariants still apply.

**Execution plan:** [`docs/src/ExecutionPlan.md`](docs/src/ExecutionPlan.md) — gate order, parallelism, stop/go decisions.

## Shell conventions

**Agents and local runbooks use PowerShell 7+ (`pwsh`) only** — not `cmd.exe` or bash.

```powershell
Set-Location "C:\Users\Nalakram\Documents\GitHub\MarketMind"

# Prefer repo venv (or py -3.12 -m ... for launcher parity with CI)
.\.venv\Scripts\python.exe -m pip install -e .

# Disable coverage addopts for narrow pytest runs
$env:PYTEST_ADDOPTS = ""
.\.venv\Scripts\python.exe -m pytest tests/python/unit/models/ --override-ini="addopts="
```

- Quote paths; use `` ` `` for line continuation.
- Git commit messages: here-strings (`git commit -m @'...'@`), not bash heredocs.
- **GitHub CLI:** copy `.env.example` → `.env` (gitignored), set `GH_TOKEN` and `GITHUB_TOKEN` to the same PAT. Agents load `.env` per `AGENTS.md` §11.7 before `gh` commands.

## North star

```text
market data  →  clean  →  preprocess  →  data/processed/   (pipeline products)
                                              ↓
                         experiments  →  artifacts/           (run outputs)
                                              ↓
                         backtests / bundles  →  artifacts/  (orchestrator path)
```

**Pipeline** (`pysrc/pipeline/`) owns stage execution: fetch, clean, preprocess, materialize. Config lives under `pysrc/pipeline/pipeline_config/`. Runtime is `dataprep_runtime.py`; read contracts are in `products.py`.

**Two dataprep lanes** share `DataPrepOrchestrator` but branch at preprocessing:

| Lane | Purpose | Config | Preprocessing |
|------|---------|--------|---------------|
| **Research dataprep** | Model selection, backtest, paper-trade candidates | `research_sip_indicators.yaml` | `indicator_engine` → W3-B indicator panel in `data/processed/` |
| **Production dataprep** | Multi-asset ingest (equities, alt, crypto), governed refresh, model-update substrate | `config.yaml` | `preprocessor_preset` / `preprocessor_grid` → `pysrc/preprocessor/` feature graph + CAS materialize |

Research is active today. Production is the long-term dataprep engine — not deprecated.

**Experiments** compose stable modules from `pysrc/models/`, `pysrc/meta/`, `pysrc/portfolio/`, and `pysrc/pipeline/panel/`. Experiment definitions live in `research/`; every workflow output is allocated and tracked by `pysrc/artifact_registry` under `artifacts/`.

**CLI** is the only entry surface:

```powershell
.\.venv\Scripts\python.exe -m pysrc.cli.marketmind dataprep run -c <config>
.\.venv\Scripts\python.exe -m pysrc.cli.marketmind backtest run --symbol AAA
.\.venv\Scripts\python.exe -m pysrc.cli.marketmind panel audit-features
.\.venv\Scripts\python.exe -m pysrc.cli.marketmind panel train-matrix
.\.venv\Scripts\python.exe -m pysrc.cli.marketmind strategies build --source-run-id <prediction-run-id>
.\.venv\Scripts\python.exe -m pysrc.cli.marketmind candidate-portfolios build --source-run-id <strategy-run-id>
.\.venv\Scripts\python.exe -m pysrc.cli.marketmind artifacts clean
```

The public CLI intentionally has no legacy command aliases and no caller-selected output directories.

## Directory contract

| Path | Role |
|------|------|
| `pysrc/` | Reusable implementation (pipeline, models, meta, portfolio, CLI) |
| `research/` | Tracked experiment configs and analyses — not generated outputs |
| `pysrc/pipeline/` | Stage pipeline — the centralized data path |
| `pysrc/models/` | Single model domain: registry, tabular/MLP estimators, runtime, dataset contracts — see SystemArchitecture.md |
| `pysrc/meta/` | Reusable meta-policy logic (pairwise advantage, local policy selector) |
| `pysrc/portfolio/` | Portfolio simulation and label utilities |
| `pysrc/backtesting/` | Backtest engines, PIT views, bundle contracts |
| `pysrc/strategies/` | Strategy implementations (momentum, stat_arb, pipeline bridge) |
| `pysrc/preprocessor/` | **Production dataprep** feature graph (preset/grid, multi-asset plans) |
| `archive/disabled_java/` | Orphan: 3 unused Java gRPC client stubs, not on the Python build path |
| `data/` | Raw and vendor panels (inputs) |
| `data/processed/` | Pipeline products (outputs of dataprep) |
| `artifacts/` | Experiment and run outputs — **never** commit bulk runs |
| `archive/` | Retired lanes, old scripts, historical docs — not default execution |
| `scripts/` | Empty of runners — see `scripts/README.md` |

Do not add new top-level `scripts/run_*.py` files. Add pipeline stages or CLI commands instead.

## Pipeline products

**Indicator feature panel** — primary research product:

```text
data/processed/full_indicator_feature_panel/panel.parquet
```

Ticker × date × interval grain with W3-B TA indicator columns plus a small supervision allowlist (`adjusted_return_1d`, `forward_return_horizon`, etc.) for panel training. Produced by:

```text
fetch → sip_adjusted_panel (market_data stage) → clean → IndicatorEngineStep → materialize
```

Backtest orchestrator (`pysrc/pipeline/orchestrator.py`) reads pipeline products via `pysrc/pipeline/products.py` when `pipeline_product=indicator_panel`.

**IndicatorEngine** — single interface to compute or load indicator columns (`pysrc/pipeline/stages/preprocessing/indicators/engine.py`). Pipeline materialization uses it.

## Experiment lanes (plain English)

P2 is a **research program** (`research/p2/`), not a source-package prefix. Lanes compose stable modules:

| Plain name | What it does | Code module | Typical artifact root |
|------------|--------------|-------------|------------------------|
| **dataprep** | Run the stage pipeline | `pysrc/pipeline/dataprep_runtime.py` | `data/processed/` |
| **SIP market data source** | Registered `sip_adjusted_panel` fetch stage | `pysrc/pipeline/stages/market_data/sources/sip_adjusted_panel.py` | reads `data/massive/.../adjusted_day_panel_v1/` |
| **strategies** | Convert standardized predictions into pre-sizing trade intents | `pysrc/strategies/` | registry run artifacts |
| **candidate portfolios** | Size standardized trade intents | `pysrc/pipeline/candidate_portfolios/` | registry run artifacts |
| **retired router comparator** | Historical provenance only | `archive/retired_product_flow/router/` | archive only |

### Primary forward research flow

```powershell
# 1. Pipeline substrate
.\.venv\Scripts\python.exe -m pysrc.cli.marketmind dataprep run `
  -c pysrc/pipeline/pipeline_config/research_sip_indicators.yaml

# 2. Panel audit + model matrix
.\.venv\Scripts\python.exe -m pysrc.cli.marketmind panel audit-features
.\.venv\Scripts\python.exe -m pysrc.cli.marketmind panel train-matrix

# 2b. Panel gates (on a completed model-matrix run)
.\.venv\Scripts\python.exe -m pysrc.cli.marketmind panel gate3-viability --run-dir artifacts/runs/<run_id>
.\.venv\Scripts\python.exe -m pysrc.cli.marketmind panel gate4-robustness --run-dir artifacts/runs/<run_id>
.\.venv\Scripts\python.exe -m pysrc.cli.marketmind panel production-smoke --run-dir artifacts/runs/<run_id> --model-id xgboost
.\.venv\Scripts\python.exe -m pysrc.cli.marketmind panel gate6-promotion --run-dir artifacts/runs/<run_id> --model-id xgboost

# 3. Strategy and candidate portfolio handoffs
.\.venv\Scripts\python.exe -m pysrc.cli.marketmind strategies build --source-run-id <prediction-run-id>
.\.venv\Scripts\python.exe -m pysrc.cli.marketmind candidate-portfolios build --source-run-id <strategy-run-id>

# 4. Optional backtest from pipeline product
.\.venv\Scripts\python.exe -m pysrc.cli.marketmind backtest run --symbol AAA
```

Smoke / CI (no real parquet):

```powershell
.\.venv\Scripts\python.exe -m pysrc.cli.marketmind panel train-matrix --smoke-test
```

## Legacy name glossary

Stop using these in new docs and user-facing text. They remain in code paths and artifact folders for compatibility.

| Legacy | Plain English |
|--------|---------------|
| P2, P2-MAP / MATRIX / NARROW / PORTFOLIO | **router experiment** stages |
| W2-V3M | **SIP adjusted day panel** loader |
| W3-B | **child-policy indicators** experiment |
| W4-A | **router supervision** audit |
| W4-B | learned router comparator (legacy, not active lane) |
| P2-PANEL | **panel model** experiment |
| canonical preprocessing | **pipeline dataprep** |
| preprocess build | **`dataprep run`** |

## P2 gate status (canonical run `model_matrix_685473afc3d0`)

| Gate | Status | Promotion model |
|------|--------|-----------------|
| 1–3 | PASS | — |
| 4 | PASS (outcome **D**) | `xgboost` |
| 5 | PASS (production smoke) | `xgboost` |
| 6 | PASS (statistical promotion) | `xgboost` |
| 7 | **PASS** (promotion bundle + mm-gate) | `xgboost` |

**PDR-002 meta-router eval: PASS** on `ee28207f3c27` (`validation_weighted_blend` Sharpe 10.18 vs `equal_blend` 7.44). See [`pdr002_meta_router_eval.md`](research/p2/notes/pdr002_meta_router_eval.md). Pin `685473afc3d0` retained (Gate 4 xgboost 8.40 vs 8.41).

### PDR-002 meta-router eval (primary)

```powershell
python -m pysrc.cli.marketmind panel meta-router-eval `
  --run-dir artifacts/runs/model_matrix_ee28207f3c27 `
  -c research/p2/configs/local_meta_router.yaml
```

Do **not** use the closed pin `685473afc3d0` for router tuning. `run full` is still a stub — use `panel meta-router-eval` instead.

### PDR-001b scratch unblock

Full `panel train-matrix` needs ~**46 GB** scratch; default run dir under `artifacts/` may be too small. Use external scratch config when ≥50 GB is available elsewhere:

```powershell
# Optional convention — edit scratch_dir in YAML to match (p2 config does not expand env vars)
$env:MARKETMIND_SCRATCH = "D:/MarketMindScratch"

python -m pysrc.cli.marketmind panel train-matrix `
  -c research/p2/configs/panel_model_matrix_external_scratch.yaml
```

See [`research/p2/notes/pdr001b_matrix_retrain.md`](research/p2/notes/pdr001b_matrix_retrain.md).

```powershell
# PDR-002 allocation POC
python -m pysrc.cli.marketmind panel policy-smoke --run-dir artifacts/runs/model_matrix_685473afc3d0

# PDR-001 finish (do not re-run on closed run except --run-stat-battery)
python -m pysrc.cli.marketmind panel promotion-finish `
  --run-dir artifacts/runs/model_matrix_685473afc3d0 `
  --model-id xgboost
```

## Invariants

- Point-in-time data access on governed paths
- Train / validation / test separation in experiments
- Pipeline products consumed via `pysrc/pipeline/products.py`, not ad hoc parquet paths
- Deterministic seeds in tests and configurable `random_seed` in experiments
- Forbidden leakage feature patterns (`hindsight`, `_net_utility`, `future_`, `oracle`, …)

## What this is not

- Not live trading or paper trading
- Not a promoted adaptive allocator (that is a future target if evidence earns it)
- Not governed by Resolution Ledger / GATE-II blocking (that governance era is retired; see git history)

## Further reading (optional)

| Doc | Role |
|-----|------|
| [`docs/src/ExecutionPlan.md`](docs/src/ExecutionPlan.md) | **Execution plan** — gates, parallelism, stop/go decisions |
| [`docs/src/SystemArchitecture.md`](docs/src/SystemArchitecture.md) | **Architecture SSOT** — package ownership, production chain, dependency rules |
| [`docs/Programming Guidelines.md`](docs/Programming%20Guidelines.md) | **Engineering posture** — registries, factories, PIT, determinism, logging, gates |
| [`docs/DECISIONS.md`](docs/DECISIONS.md) | **Decisions** — all architecture (ADR) and product (PDR) decisions |

Historical material — the vision/thesis/roadmap narrative, the companion suite, and retired lanes — lives in git history and `archive/`.

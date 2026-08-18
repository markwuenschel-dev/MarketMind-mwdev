# MarketMind Execution Plan

**Role:** When and depending on what — gates, parallel workstreams, stop/go decisions.
**Does not replace:** [`SystemArchitecture.md`](./SystemArchitecture.md) (package ownership),
[`CONTEXT.md`](../../CONTEXT.md) (commands), [`AGENTS.md`](../../AGENTS.md) (agent rules),
[`docs/DECISIONS.md`](../DECISIONS.md) (decisions).

## Governing rule

> Do not let the next layer compensate for a failure in the previous layer.

> **Single critical path (ADR-002, PDR-001):** validating the **walk-forward panel model matrix**
> product. Everything kept off the active flow (backtesting, portfolio, strategies, meta/
> meta_learning) is dormant and gated on that validation. The former local-MetaRouter gate and the
> old date-level router comparator are **deleted** — they are not part of this plan.

## Doc hierarchy

| Document | Owns |
|----------|------|
| `SystemArchitecture.md` | What each package owns |
| `Programming Guidelines.md` | How code is structured |
| `AGENTS.md` | How agents respect boundaries |
| `docs/DECISIONS.md` | Architecture (ADR) and product (PDR) decisions |
| `CONTEXT.md` | Operational state and commands |
| **`ExecutionPlan.md`** | Gate order, parallelism, outcomes |
| Research notes under `research/p2/notes/` | Experiment findings — cannot redefine architecture |

## Primary forward experiment

```text
canonical panel
→ model matrix (panel train-matrix)
→ candidate portfolios
→ validate: are the panel predictions economically meaningful?
```

If validated, the dormant productionization chain (`backtesting → portfolio → strategies`, PDR-001)
and later the regime-aware allocation layer (PDR-002) are revived — each its own gated effort.

## Main dependency tree

```text
Gate 1  Canonical data substrate
    ↓
Gate 2  Base-model matrix (diversity, not just executability)
    ↓
Gate 3  Candidate portfolio economic viability
    ↓
Gate 4  Panel-product validation  →  stop / simplify / expand / harden
    ↓ (only if outcome D)
Gate 5  Production chain smoke (PDR-001)  →  statistical backtesting battery
    ↓ (only if Gate 5 PASS)
Gate 6  Statistical promotion (DSR/PBO/Harvey)  →  promotion bundle + release planning
    ↓
Dormant: full productionization + allocation layer (PDR-002)
    ↓ (PDR-002 primary — not another train-matrix)
PDR-002a  Meta-router eval battery (`panel meta-router-eval`) on fixed model-matrix run
    ↓ pass if best gate beats equal_blend test Sharpe
PDR-002b  Policy selector sweep / fold attribution (optional)
```

## Parallel tracks (non-blocking)

| Track | When | Blocks main chain? |
|-------|------|-------------------|
| A — LSTM PanelModel prep | After Gate 1; notes in `research/p2/notes/` | No |
| C — Macro-state path audit | After Gate 1 | No |
| D — meta / meta_learning hygiene (ADR-003 re-shelving) | When PDR-002 is revived | No |

Re-enter the main chain only when a track addresses a diagnosed bottleneck (e.g. Track A after a
Gate 2 diversity gap).

## Gates

### Gate 1 — Canonical substrate

```bash
python -m pysrc.cli.marketmind dataprep run \
  -c pysrc/pipeline/pipeline_config/research_sip_indicators.yaml
python -m pysrc.cli.marketmind panel audit-features
```

Require: `full_indicator_feature_panel`, `grain_valid=true`, `duplicate_key_count=0`, resolvable
target.

### Gate 2 — Model matrix

```bash
python -m pysrc.cli.marketmind panel train-matrix \
  -c research/p2/configs/panel_model_matrix.yaml
```

Rule: `registered = executable`. Stop and fix if `model_diversity_report.json` shows
`low_diversity_warning` or redundant pairs dominate the meaningful child count. `equal_blend` is a
candidate-portfolio comparator, not a model family.

### Gate 3 — Candidate portfolios

Evaluate economic viability of the predictions turned into positions. Expression failure (the
prediction has no economic signal) is not an allocation problem — diagnose it here, before any
allocation work.

### Gate 4 — Panel-product validation (the critical path)

Decide whether the panel matrix is a real product:

| Outcome | Next |
|---------|------|
| A — no economic signal vs. baseline | Oracle-gap / target audit; may stop |
| B — near-miss | Attribution; not success |
| C — narrow win | Robustness battery |
| D — robust win | Revive the dormant productionization chain (PDR-001) |

### Gate 5 — Production chain smoke (PDR-001)

Wire the Gate 4 winner through the dormant production path without retrain or new data:

```text
predictions → threshold intents → positions → simulate
                                      ↓
                         PortfolioTargetPlan → BacktestSuiteRunner (smoke)
```

| Check | Pass criteria |
|-------|---------------|
| Parity | Strategy path matches Gate 3 direct path per `fold_id` (Sharpe ±0.05, cum log ±0.5) |
| Economics | Strategy-path metrics match Gate 4 `by_fold` for promotion model |
| Backtest smoke | `BacktestSuiteRunner.execute` completes with `mechanical.v1` validator |

CLI: `python -m pysrc.cli.marketmind panel production-smoke --run-dir <run> --model-id xgboost`

| Result | Next |
|--------|------|
| Parity PASS + backtest smoke PASS | Gate 5 PASS → statistical backtesting battery (DSR/PBO) |
| Parity FAIL | Fix `production_bridge.py`; do not wire backtesting |
| Backtest smoke FAIL | Diagnose `PortfolioTargetPlan` / engine registry |

Research notes: `research/p2/notes/gate5_production_smoke.md`. Not an architecture override — cite PDR-001.

### Gate 6 — Statistical promotion battery

Run DSR/PBO/Harvey/bootstrap on strategy-path daily returns with honest `n_trials` (model-matrix search breadth):

```text
strategy positions → simulate (10 bps) → pooled net_return series
                                      ↓
                         run_validity_report + evaluate_promotion_gate
                                      ↓
                         BacktestSuiteRunner + statistical.v1 (integration)
```

| Check | Pass criteria |
|-------|---------------|
| Returns | Pooled OOS daily `net_return` from Gate 5 strategy path |
| DSR | `n_trials` = trained model families (not 1); anti-Goodhart — do not tune thresholds to fit run |
| Harvey t-stat | ≥ 3.0 |
| PBO | Walk-forward fold holdout surface (directional; 3 folds is thin) |
| Backtest integration | `statistical.v1` emits `stat_validity_report.json` on promotion returns |

CLI: `python -m pysrc.cli.marketmind panel gate6-promotion --run-dir <run> --model-id xgboost`

| Result | Next |
|--------|------|
| `gate_pass: true` | Gate 6 PASS → Gate 7 promotion bundle + `mm-gate validate` |
| DSR/Harvey FAIL | Stop promotion; attribution — do not retrain |
| PBO WARN/FAIL only | Document; full CPCV is separate gated effort |

Research notes: `research/p2/notes/gate6_promotion.md`.

### Gate 7 — Promotion bundle (PDR-001 finish line)

Assemble Appendix C bundle for the Gate 4/6 winner and validate with `mm-gate`:

```text
gate6 stat_validity_report + splits manifest + plan/env/dataset/preprocessing
                                      ↓
                         promotion_model_ledger (all 10 models)
                                      ↓
                         bundles/promotion_{model}/ + mm-gate validate
                                      ↓
                         crisis holdout slice + pdr001_finish_report.json
```

| Check | Pass criteria |
|-------|---------------|
| Ledger | `promotion_model_ledger.json` ranks all trained families |
| Bundle | Appendix C required files + sidecars |
| mm-gate | `validate_bundle` exit 0 |
| Crisis holdout | Documented in finish report (non-blocking on research lane) |
| Pin | `panel_promotion_manifest.json` |

CLI:

```powershell
python -m pysrc.cli.marketmind panel promotion-finish --run-dir <run> --model-id xgboost
```

Sub-commands: `panel assemble-promotion-bundle --validate`.

| Result | Next |
|--------|------|
| `finish_pass: true` | **PDR-001 research lane complete** — no Gate 8 on this path |
| mm-gate FAIL | Fix bundle assembler; do not retune gates |
| New retrain / data | Start new PDR (PDR-002 or later) |

Research notes: `research/p2/notes/pdr001_closure.md`, `gate7_promotion_bundle.md`.

## What not to do

- No `pysrc/ml/`; no placeholder model registry entries (`registered = executable`).
- Do not reintroduce a router/allocator orchestration package without an ADR (ADR-002 deleted it).
- No architecture scaffolding instead of Gates 1–4 on real data.
- Research notes cannot override `SystemArchitecture.md`.

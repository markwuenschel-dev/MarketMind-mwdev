# MarketMind

Governed algorithmic trading research platform. Four-way separation:

```text
pysrc/      reusable implementation
research/   experiment definitions and analyses
data/       canonical processed datasets
artifacts/  disposable run outputs
```

`CONTEXT.md` is the canonical vision and vocabulary for operational state and commands. Package ownership and dependency boundaries live in [`docs/src/SystemArchitecture.md`](docs/src/SystemArchitecture.md).

This is not live trading or paper trading. Portfolio and backtest commands are research steps.

## Quick Start

Use Python 3.12 for local validation parity with CI. Run commands in **PowerShell 7+** (`pwsh`). Inside `.venv`, use `.\.venv\Scripts\python.exe -m ...` (or `py -3.12 -m ...` with the Windows launcher).

```powershell
Set-Location "C:\path\to\MarketMind"
.\.venv\Scripts\python.exe -m pip install -e .

# 1. Pipeline: market data -> clean -> indicators -> data/processed/
.\.venv\Scripts\python.exe -m pysrc.cli.marketmind dataprep run `
  -c pysrc/pipeline/pipeline_config/research_sip_indicators.yaml

# 2. Experiments -> artifacts/
.\.venv\Scripts\python.exe -m pysrc.cli.marketmind indicators build
.\.venv\Scripts\python.exe -m pysrc.cli.marketmind supervision audit
.\.venv\Scripts\python.exe -m pysrc.cli.marketmind router full

# 3. Optional backtest from the pipeline product
.\.venv\Scripts\python.exe -m pysrc.cli.marketmind backtest run --symbol AAA

# 4. Optional panel model
.\.venv\Scripts\python.exe -m pysrc.cli.marketmind panel audit-features
.\.venv\Scripts\python.exe -m pysrc.cli.marketmind panel train

# 5. Housekeeping
.\.venv\Scripts\python.exe -m pysrc.cli.marketmind artifacts clean --keep-latest 5 --delete-smoke
```

CI or laptop smoke run without real supervision data:

```powershell
.\.venv\Scripts\python.exe -m pysrc.cli.marketmind router full --smoke-test --max-candidates 8
.\.venv\Scripts\python.exe -m pysrc.cli.marketmind panel audit-features --smoke-test
```

## Active Architecture

See [`docs/src/SystemArchitecture.md`](docs/src/SystemArchitecture.md) for the production chain, package ownership, and dependency rules. This README does not duplicate that document.

**Entrypoint:** `.\.venv\Scripts\python.exe -m pysrc.cli.marketmind` (or `py -3.12 -m pysrc.cli.marketmind` with the Windows launcher). Use PowerShell 7+ — see `CONTEXT.md` § Shell conventions.

Do not add new `scripts/run_*.py` runners. Add pipeline stages or CLI commands instead. Retired script shims live under `archive/scripts_shims/`.

## Pipeline Product Contract

Primary product:

```text
data/processed/full_indicator_feature_panel/panel.parquet
```

Expected producer:

```text
dataprep run -c pysrc/pipeline/pipeline_config/research_sip_indicators.yaml
```

Expected consumer paths:

- `pysrc/pipeline/products.py` resolves product locations and loading.
- `pysrc/meta/allocator_benchmark/w3_b_pandas_ta.py` can reuse the pipeline indicator panel.
- `pysrc/pipeline/panel/` audits and trains on the panel.
- `research/p2/configs/` holds P2 experiment matrices (panel baselines, router matrix, local meta-router).
- `pysrc/pipeline/orchestrator.py` reads it for `backtest run --pipeline-product indicator_panel`.

## Command Inputs And Outputs

| Command | Required input | Main output |
|---|---|---|
| `dataprep run -c <config>` | Pipeline YAML/JSON config | `data/processed/full_indicator_feature_panel/panel.parquet` plus manifest |
| `indicators build` | SIP adjusted panel or pipeline indicator panel | `artifacts/runs/<run_id>/` (legacy read: `artifacts/phase_ii/w3_b_pandas_ta/`) |
| `supervision audit` | Child-policy artifacts and adjusted panel | `artifacts/runs/<run_id>/` (legacy read: `artifacts/phase_ii/w4_a_router_opportunity/`) |
| `router full` | Supervision artifacts or `--smoke-test` | `artifacts/runs/<run_id>/` (legacy read: `artifacts/phase_ii/p2_broad_reset/`) |
| `router narrow` | Router candidate matrix | Narrowing report and switch diagnostics under router artifacts |
| `backtest run` | Pipeline product or CSV input | Bundle under `bundles/<timestamp>/` or `--bundle-dir` |
| `panel audit-features` | Pipeline indicator panel or `--smoke-test` | `artifacts/runs/<run_id>/` feature universe report |
| `panel train` | Pipeline indicator panel | `artifacts/runs/<run_id>/` panel model report |
| `artifacts clean` | Existing `artifacts/runs/<run_id>/` tree | Retention report and pruned run dirs |

Legacy aliases still work for compatibility: `phase2` = `router`, `w3` = `indicators`, `w4` = `supervision`. Prefer the plain command names above.

## Leakage And Determinism Protections

- Governed data access uses point-in-time boundaries.
- Experiments preserve train, validation, and test separation.
- Router matrix construction blocks forbidden leakage feature names such as `hindsight`, `_net_utility`, `future_`, and `oracle`.
- Tests use determinism markers and the deterministic seed fixture.
- Experiment CLIs expose deterministic seed options where the lane needs them.

## Documentation Policy

- **Architecture:** [`docs/src/SystemArchitecture.md`](docs/src/SystemArchitecture.md) — package ownership and boundaries (SSOT).
- **Operations:** `CONTEXT.md` — current commands, directory contract, experiment lanes.
- **Agents:** `AGENTS.md` — how agents must respect architecture boundaries.
- **Decisions:** [`docs/DECISIONS.md`](docs/DECISIONS.md) — all architecture (ADR) and product (PDR) decisions.
- **Engineering posture:** [`docs/Programming Guidelines.md`](docs/Programming%20Guidelines.md) — registries, factories, PIT, determinism, logging.
- **Execution:** [`docs/src/ExecutionPlan.md`](docs/src/ExecutionPlan.md) — gate order, parallelism, stop/go.
- **Research lane:** process docs under `docs/research/`.
- Historical material lives in `archive/` and git history — not the default execution path unless `CONTEXT.md` points at it.

For material research runs, record the command, input path, output path, candidate count, baselines, result table, and next debugging step.

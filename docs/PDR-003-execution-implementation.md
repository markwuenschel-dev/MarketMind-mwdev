# PDR-003 — Execution implementation plan

**Status:** Wave 3 scaffold · 2026-06-23  
**Decisions:** PDR-003, ADR-005, ADR-006 in [`docs/DECISIONS.md`](DECISIONS.md)

## Goal

Real-time IBKR **paper** trading for one promoted strategy (`prediction_threshold_xgboost`) with continuous reconciliation and kill switch.

## Current scaffold (Wave 3)

| Component | Path | Status |
|-----------|------|--------|
| Promotion bundle runtime | `pysrc/strategies/prediction_threshold.py` | Implemented |
| Shadow replay v2 | `pysrc/tuning/execution/run_shadow_plan.py` | Per-bar provenance, n_days ≤ 60 |
| Inference adapter | `pysrc/tuning/live/inference_adapter.py` | xgboost load + PIT feature assembly |
| Reconciliation schema | `pysrc/tuning/execution/reconciliation.py` | Stub |
| Kill switch | `pysrc/tuning/execution/kill_switch.py` | Stub |
| Paper loop dry-run | `pysrc/tuning/execution/paper_loop.py` | Skeleton |
| IBKR orders interface | `pysrc/infra/brokers/ibkr/orders.py` | Schema + stub |
| CLI | `execution shadow-run`, `execution paper-dry-run` | Wired |
| Kotlin adapter | Out of repo | ADR-006 |

## Paper promotion evidence (from PDR-003)

- ≥ 60 trading days, ≥ 100 submitted orders
- Continuous order/fill reconciliation
- Position/cash reconciliation every completed 5-minute bar
- Zero critical risk-control failures
- Tested kill-switch / recovery

## 60-day paper runbook

### Prerequisites

1. Gate 7 promotion bundle assembled and `mm-gate validate` passes.
2. Model-matrix run contains `predictions/model_prediction_panel.parquet` and `models/xgboost/`.
3. IBKR paper gateway running (target: `127.0.0.1:7497` per `IBKRPaperAccountConfig`).

### Phase A — Offline shadow evidence (days 1–60 replay)

```powershell
python -m pysrc.cli.marketmind execution shadow-run `
  --bundle-dir artifacts/runs/<run_id>/bundles/promotion_xgboost `
  --run-dir artifacts/runs/<run_id> `
  --n-days 60
```

Inspect `artifacts/runs/<run_id>/reports/shadow_replay_log.json`:

- `schema_version` = `shadow_replay_log.v2`
- `bars` length equals replay window (up to 60)
- Each bar records `intent_count`, `instruments`, and PIT lineage marker

### Phase B — Live paper loop (Kotlin target per ADR-006)

**Python dry-run skeleton** (`paper_loop_dry_run`) simulates the bar cycle offline before Kotlin wiring:

```powershell
python -m pysrc.cli.marketmind execution paper-dry-run `
  --bundle-dir artifacts/runs/<run_id>/bundles/promotion_xgboost `
  --run-dir artifacts/runs/<run_id> `
  --n-bars 78
```

- `PAPER_TRADING_ENABLED` defaults to `0` — no IBKR submission from Python.
- Output: `reports/paper_loop_log.json` (`paper_loop_log.v1`) with per-bar reconciliation and kill-switch state.
- Wires `InferenceAdapter`, `load_promotion_bundle_runtime`, `compare_ledger_to_broker`, and `KillSwitchState`.

**Live target flow:**

```text
IBKR streams (bars, orders, fills, positions, cash)
        ↓
Execution service (Kotlin)
        ↓
InferenceAdapter.assemble_features_as_of + predict (Python bridge or pre-materialized)
        ↓
Internal ledger vs broker state (every 5m bar) — reconciliation.py
        ↓
Mismatch → KillSwitchState.engage → block_new_orders
```

### Phase C — Evidence collection checklist

| Day bucket | Action | Artifact |
|------------|--------|----------|
| Daily | Shadow or live replay | `shadow_replay_log.json` append / daily slice |
| Every 5m bar | Position/cash reconcile | `reconciliation_diff.v1` payload |
| On mismatch | Engage kill switch | `block_new_orders=true` + owner alert |
| Weekly | Review intent vs fill parity | run notes under `research/p2/notes/` |
| Day 60 | Gate evidence summary | PDR-003 promotion packet |

### Stop / go

- **Go:** 60 trading days, ≥ 100 orders, zero critical risk failures, kill-switch drill documented.
- **Stop:** Any unresolved ledger/broker mismatch or untested kill-switch recovery.

## Reconciliation loop (to implement)

```text
IBKR streams (bars, orders, fills, positions, cash)
        ↓
Execution service (Kotlin target per ADR-006)
        ↓
Internal ledger vs broker state (every 5m bar)
        ↓
Mismatch → block new orders + owner alert
```

## Python ↔ Kotlin boundary

- Python: research, bundle validation, shadow replay, training artifacts, reconciliation/kill-switch schema
- Kotlin/Spring: order submission, broker streams, kill switch persistence, reconciliation persistence

## Next sprint

1. Wire `IBKROrderExecutor.submit` (ib_insync) against paper account
2. Persist reconciliation diffs and kill-switch transitions
3. Connect live bar feed to `InferenceAdapter` on completed 5-minute bars
4. Collect 60-day paper evidence against PDR-003 gates

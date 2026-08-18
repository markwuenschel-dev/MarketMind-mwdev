# ADR-004 — PDR-002 orchestrator boundary

**Status:** Accepted · 2026-06-22  
**Context:** PDR-001 panel product validated (Gate 4 outcome D). Production chain smoke (Gate 5–7) complete. PDR-002 revival requires a named orchestrator home without resurrecting deleted `pipeline/meta_router/`.

## Decision

The **regime-aware policy/allocation orchestrator** for the panel matrix lives in:

`pysrc/pipeline/candidate_portfolios/policy_bridge.py`

It sits beside `production_bridge.py` (Gate 5) and consumes:

- `model_prediction_panel.parquet` (all candidates)
- `meta/local_policy_selector.py` (boring allocator first)
- `portfolio/labels.py` (`delta_utility_vs_default` training target)
- `production_backtest.py` for `PortfolioTargetPlan` emission (future step)

## Boundary rules

| Layer | Owns |
|-------|------|
| `meta_learning/` | Task identity math, regime vocabulary, encoder contracts |
| `meta/` | Trading policy primitives (gates, BOCPD labels, abstention) |
| `policy_bridge.py` | Wiring predictions → policy training frame → routed economics |
| `production_bridge.py` | Single-model threshold strategy path (PDR-001) |

**Import direction:** `policy_bridge` may import `meta/` and `meta_learning/` substrate types. `meta_learning/` must not import `meta/` (ADR-003).

## Consequences

- CLI entry: `panel policy-smoke`
- No new top-level package; no MetaRouter runner resurrection
- Neural gate / MoE spikes stay quarantined until boring path passes
- ADR-003 MetaTask SSOT remains `pysrc/meta_learning/task_generator.py`; `pysrc/meta/task.py` is a compatibility re-export shim

## Alternatives rejected

- **Revive `pipeline/meta_router/`** — deleted per ADR-002; would recreate parallel hierarchy
- **Orchestrator in `meta/`** — violates production chain placement next to candidate portfolios

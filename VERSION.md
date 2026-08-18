# VERSION

MarketMind no longer maintains a governance release ledger in the active research path. Git history is the code-history source of truth; this file is a lightweight research-state note.

## Research State — 2026-06-23

**PDR-001 research lane: COMPLETE** on canonical run `model_matrix_685473afc3d0` (promotion model `xgboost`, Gate 7 PASS). See [`research/p2/notes/pdr001_closure.md`](research/p2/notes/pdr001_closure.md).

**Sprint 2 (2026-06-23):** PDR-002/003 code landed in `pysrc/`; PDR-001b full retrain blocked on scratch disk. Closed run `685473afc3d0` unchanged.

**Next product tracks (each its own PDR):**

- **PDR-002** — regime-aware policy/allocation layer (`panel policy-smoke`)
- **PDR-003** — IBKR paper execution scaffold (shadow replay, strategy bridge)
- **PDR-001b** — target canary + optional full matrix retrain (after panel rebuild)
- Parallel prep: LSTM PanelModel Phase 1, macro_state_panel fixture

Finish-line command (PDR-001, do not re-run on closed run except skepticism flags):

```powershell
python -m pysrc.cli.marketmind panel promotion-finish `
  --run-dir artifacts/runs/model_matrix_685473afc3d0 `
  --model-id xgboost
```

Validation target:

- [ ] `python -m ruff check . --no-fix`
- [ ] `python -m mypy pysrc/`
- [ ] `python -m pytest tests/python/ --no-cov`
- [ ] `mvn -B compile -DskipTests -q`

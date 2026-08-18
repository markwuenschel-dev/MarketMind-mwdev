from __future__ import annotations

import json
from pathlib import Path

import pytest

from pysrc.meta.anti_signal_autopsy import (
    ALERT_MANY_TO_ONE_JOIN_EXPANSION,
    ALERT_TRAIN_EVAL_OVERLAP,
    assemble_final_report,
    build_canonical_task_table,
    choose_latest_governed_w1_run,
    load_autopsy_context,
    run_alignment_audit,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_PHASE_II_ROOT = _REPO_ROOT / "artifacts" / "phase_ii"


@pytest.mark.determinism("d0")
def test_choose_latest_governed_w1_run_prefers_latest_guardrailed_bundle(
    deterministic_seed: int,
) -> None:
    _ = deterministic_seed
    path = choose_latest_governed_w1_run(_PHASE_II_ROOT)
    assert path == _PHASE_II_ROOT / "w1_real_learned_guardrailed"


@pytest.mark.determinism("d0")
def test_canonical_task_table_and_alignment_audit_flag_duplicate_surface_predictions(
    deterministic_seed: int,
) -> None:
    _ = deterministic_seed
    ctx = load_autopsy_context(_PHASE_II_ROOT / "w1_real_learned_guardrailed")
    table = build_canonical_task_table(ctx)
    assert table.rows
    assert any(int(row["challenger_prediction_record_count"]) > 1 for row in table.rows)

    audit = run_alignment_audit(ctx, table)
    alerts = {str(x) for x in audit["hard_alerts"]}
    assert ALERT_MANY_TO_ONE_JOIN_EXPANSION in alerts
    assert ALERT_TRAIN_EVAL_OVERLAP in alerts


@pytest.mark.determinism("d0")
def test_assemble_final_report_preserves_agent_disagreements(deterministic_seed: int) -> None:
    _ = deterministic_seed
    inputs = {
        "agent_2": {
            "hard_alerts": [ALERT_MANY_TO_ONE_JOIN_EXPANSION],
            "primary_candidate_classification": "INVALID_EVAL_ALIGNMENT",
        },
        "agent_3": {
            "primary_candidate_classification": "POSSIBLE_SIGN_OR_OBJECTIVE_INVERSION",
        },
        "agent_4": {
            "primary_candidate_classification": "SUPPORT_QUERY_REVERSAL",
        },
    }
    report, _summary, alerts = assemble_final_report(inputs)
    assert report["primary_classification"] == "INVALID_EVAL_ALIGNMENT"
    assert report["agent_disagreements"]
    disagree = report["agent_disagreements"][0]
    assert disagree["field"] == "primary_classification"
    assert disagree["agent_values"]["agent_3"] == "POSSIBLE_SIGN_OR_OBJECTIVE_INVERSION"
    assert disagree["agent_values"]["agent_4"] == "SUPPORT_QUERY_REVERSAL"
    assert ALERT_MANY_TO_ONE_JOIN_EXPANSION in alerts["hard_alerts"]
    json.dumps(report)
    json.dumps(alerts)

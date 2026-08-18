from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from pysrc.meta.task import MAX_SIGNALS, MetaTask
from pysrc.meta.w1_baseline_config import W1BaselineConfig
from pysrc.meta.w1_challenger_surface import (
    W1_CHALLENGER_SURFACE_SCHEMA_VERSION,
    W1ChallengerPrediction,
    W1ChallengerSurface,
)
from pysrc.meta.w1_clean_rerun import (
    build_w1_clean_baseline_predictions,
    build_w1_clean_challenger_predictions,
    build_w1_clean_comparison_table,
    build_w1_clean_gate_report,
    build_w1_clean_metrics,
    build_w1_clean_task_universe,
    build_w1_clean_walk_forward_splits,
)
from pysrc.meta_learning.regime_vocabulary import REGIME_CLASS_ORDER


def _ids() -> tuple[str, ...]:
    return tuple(f"s{i}" for i in range(MAX_SIGNALS))


def _mask() -> tuple[bool, ...]:
    return tuple(i == 0 for i in range(MAX_SIGNALS))


def _active_mask(active_k: int) -> tuple[bool, ...]:
    return tuple(i < active_k for i in range(MAX_SIGNALS))


def _task(i: int) -> MetaTask:
    day = datetime(2024, 1, 1, tzinfo=UTC) + timedelta(days=i * 7)
    support = tuple((day + timedelta(days=j)).isoformat().replace("+00:00", "Z") for j in range(6))
    query = tuple(
        (day + timedelta(days=20 + j)).isoformat().replace("+00:00", "Z") for j in range(4)
    )
    emb = np.full(4, 0.01 * float(i + 1), dtype=np.float32)
    return MetaTask(
        task_id=f"w1-clean-{i:05d}",
        regime_id=f"{REGIME_CLASS_ORDER[i % len(REGIME_CLASS_ORDER)]}__regime",
        regime_class=REGIME_CLASS_ORDER[i % len(REGIME_CLASS_ORDER)],
        t0=support[0],
        t1=(day + timedelta(days=45)).isoformat().replace("+00:00", "Z"),
        pit_boundary=support[-1],
        support_set=support,
        query_set=query,
        signal_ids=_ids(),
        signal_mask=_active_mask((i % 5) + 1),
        signal_set_version="1",
        signal_ids_hash="sha256:w1_clean_test",
        horizon=5,
        active_k=(i % 5) + 1,
        regime_embedding=emb,
    )


class _FakeIncumbent:
    def __init__(self) -> None:
        self._fitted = False

    def fit_for_training_tasks(
        self,
        tasks: list[MetaTask],
        *,
        rng_seed: int,
        training_targets: list[float] | None = None,
    ) -> None:
        _ = (tasks, rng_seed, training_targets)
        self._fitted = True

    def predict_scores(
        self,
        task: MetaTask,
        *,
        fold_index: int,
        rng_seed: int,
        use_hash_score_jitter: bool = True,
    ) -> np.ndarray:
        assert self._fitted
        _ = (rng_seed, use_hash_score_jitter)
        base = float(fold_index) + float(task.active_k) * 0.01
        return np.full(len(task.query_set), base, dtype=np.float64)


def _query_targets(tasks: tuple[MetaTask, ...]) -> dict[str, float]:
    return {task.task_id: float(idx) / 100.0 for idx, task in enumerate(tasks)}


def _support_targets(tasks: tuple[MetaTask, ...]) -> dict[str, float]:
    return {task.task_id: -0.05 + float(idx) / 200.0 for idx, task in enumerate(tasks)}


def _challenger_surface(
    tasks: tuple[MetaTask, ...], splits: dict[str, object]
) -> W1ChallengerSurface:
    predictions: list[W1ChallengerPrediction] = []
    folds = splits["folds"]
    assert isinstance(folds, list)
    by_id = {task.task_id: task for task in tasks}
    for fold in folds:
        assert isinstance(fold, dict)
        fold_id = int(fold["fold_id"])
        eval_ids = fold["eval_task_ids"]
        assert isinstance(eval_ids, list)
        for task_id in eval_ids:
            task = by_id[str(task_id)]
            predictions.append(
                W1ChallengerPrediction(
                    task_id=task.task_id,
                    fold_index=fold_id,
                    query_scores=tuple(
                        0.02 * float(fold_id + j + 1) for j in range(len(task.query_set))
                    ),
                    prediction_time_utc="2026-04-25T12:00:00Z",
                    pit_boundary=task.pit_boundary,
                )
            )
    return W1ChallengerSurface(
        schema_version=W1_CHALLENGER_SURFACE_SCHEMA_VERSION,
        source="reptile_learned_checkpoint_w1.v1",
        model_family="reptile_meta_allocator_checkpoint",
        predictions=tuple(predictions),
        task_pool_hash="sha256:" + "a" * 64,
        data_fingerprint="sha256:" + "b" * 64,
        splits_fingerprint="sha256:" + "c" * 64,
        cost_assumptions_fingerprint="sha256:" + "d" * 64,
        signal_set_version="1",
        created_at_utc="2026-04-25T12:00:00Z",
        leakage_policy="pit_support_only_no_query_labels",
        uses_query_labels=False,
        uses_xgboost_outputs=False,
        uses_post_query_metrics=False,
        model_state_hash="sha256:" + "e" * 64,
        governed_checkpoint_lineage_verified=True,
    )


@pytest.mark.determinism("d1")
def test_w1_clean_rerun_contract_produces_one_row_per_eval_task(deterministic_seed: int) -> None:
    _ = deterministic_seed
    tasks = tuple(_task(i) for i in range(120))
    cfg = W1BaselineConfig()
    query_targets = _query_targets(tasks)
    support_targets = _support_targets(tasks)
    task_universe, _schema, agent_1_audit = build_w1_clean_task_universe(
        tasks=tasks,
        task_query_targets=query_targets,
        task_support_targets=support_targets,
        data_fingerprint="sha256:" + "b" * 64,
        cost_assumptions_fingerprint="sha256:" + "d" * 64,
        splits_fingerprint="sha256:" + "c" * 64,
        task_pool_hash="sha256:" + "a" * 64,
        cost_per_bar=0.001,
    )
    splits_doc, agent_2_audit, _overlap_rows = build_w1_clean_walk_forward_splits(
        tasks=tasks, config=cfg
    )
    baseline_doc, agent_3_audit = build_w1_clean_baseline_predictions(
        tasks=tasks,
        splits_doc=splits_doc,
        config=cfg,
        incumbent=_FakeIncumbent(),
        task_query_targets=query_targets,
        data_fingerprint="sha256:" + "b" * 64,
        splits_fingerprint="sha256:" + "c" * 64,
        cost_assumptions_fingerprint="sha256:" + "d" * 64,
        task_pool_hash="sha256:" + "a" * 64,
        prediction_time_utc="2026-04-25T12:00:00Z",
    )
    challenger_doc, agent_4_audit = build_w1_clean_challenger_predictions(
        surface=_challenger_surface(tasks, splits_doc),
        splits_doc=splits_doc,
        challenger_model_id="w1_7d_meta_head_or_successor",
    )
    comparison_rows, agent_5_audit, _join_rows = build_w1_clean_comparison_table(
        task_universe_rows=task_universe,
        splits_doc=splits_doc,
        baseline_doc=baseline_doc,
        challenger_doc=challenger_doc,
        expected_data_fingerprint="sha256:" + "b" * 64,
        expected_splits_fingerprint="sha256:" + "c" * 64,
        expected_cost_assumptions_fingerprint="sha256:" + "d" * 64,
        expected_task_pool_hash="sha256:" + "a" * 64,
    )
    metrics_doc, fold_metrics, regime_metrics, agent_6_audit = build_w1_clean_metrics(
        comparison_rows=comparison_rows,
        expected_eval_rows=100,
    )
    report, _summary, gate_audit, _supersession_note = build_w1_clean_gate_report(
        agent_1_audit=agent_1_audit,
        agent_2_audit=agent_2_audit,
        agent_3_audit=agent_3_audit,
        agent_4_audit=agent_4_audit,
        agent_5_audit=agent_5_audit,
        agent_6_audit=agent_6_audit,
        metrics_doc=metrics_doc,
        declared_unique_eval_tasks=100,
        declared_task_universe_size=120,
        run_id="w1-clean-test",
        data_fingerprint="sha256:" + "b" * 64,
        splits_fingerprint="sha256:" + "c" * 64,
        cost_assumptions_fingerprint="sha256:" + "d" * 64,
        task_pool_hash="sha256:" + "a" * 64,
    )

    assert len(task_universe) == 120
    assert len(comparison_rows) == 100
    assert agent_1_audit["status"] == "PASS"
    assert agent_2_audit["status"] == "PASS"
    assert agent_3_audit["status"] == "PASS"
    assert agent_4_audit["status"] == "PASS"
    assert agent_5_audit["status"] == "PASS"
    assert agent_6_audit["status"] == "PASS"
    assert gate_audit["hard_alerts"] == []
    assert report["w1_gate_closure_eligible"] is True
    assert report["counts"]["n_final_eval_rows"] == 100
    assert report["counts"]["n_unique_eval_task_ids"] == 100
    assert len(fold_metrics) == cfg.n_walk_forward_folds
    assert regime_metrics


@pytest.mark.determinism("d1")
def test_w1_clean_rerun_gate_fails_when_task_count_claim_is_wrong(deterministic_seed: int) -> None:
    _ = deterministic_seed
    report, _summary, gate_audit, _supersession_note = build_w1_clean_gate_report(
        agent_1_audit={
            "status": "PASS",
            "n_tasks": 5,
            "n_unique_task_ids": 5,
            "hard_alerts": [],
            "soft_alerts": [],
        },
        agent_2_audit={
            "status": "PASS",
            "n_train_eval_overlap": 0,
            "n_duplicate_eval_task_ids_within_fold": 0,
            "expected_eval_rows": 5,
            "expected_unique_eval_tasks": 5,
            "hard_alerts": [],
            "soft_alerts": [],
        },
        agent_3_audit={
            "status": "FAIL",
            "hard_alerts": ["MISSING_BASELINE_PREDICTION_ARTIFACT"],
            "soft_alerts": [],
        },
        agent_4_audit={"status": "PASS", "hard_alerts": [], "soft_alerts": []},
        agent_5_audit={
            "status": "FAIL",
            "n_final_eval_rows": 5,
            "n_unique_eval_task_ids": 5,
            "n_missing_baseline_scores": 5,
            "n_missing_challenger_scores": 0,
            "n_missing_query_targets": 0,
            "n_duplicate_eval_task_ids_within_fold": 0,
            "n_join_expansion_alerts": 0,
            "baseline_eval_task_set_matches_challenger": True,
            "hard_alerts": ["MISSING_BASELINE_SCORE"],
            "soft_alerts": [],
        },
        agent_6_audit={
            "status": "FAIL",
            "hard_alerts": ["TASK_LEVEL_TABLE_INVALID"],
            "soft_alerts": [],
        },
        metrics_doc={
            "baseline_metrics": {
                "mean_gross_ic": None,
                "mean_net_sharpe": None,
                "mean_turnover": None,
            },
            "challenger_metrics": {
                "mean_gross_ic": None,
                "mean_net_sharpe": None,
                "mean_turnover": None,
            },
            "diagnostics": {
                "challenger_flipped_ic": None,
                "support_query_transfer_ic": None,
                "baseline_challenger_score_ic": None,
            },
        },
        declared_unique_eval_tasks=120,
        declared_task_universe_size=5,
        run_id="w1-clean-test-invalid",
        data_fingerprint="sha256:" + "b" * 64,
        splits_fingerprint="sha256:" + "c" * 64,
        cost_assumptions_fingerprint="sha256:" + "d" * 64,
        task_pool_hash="sha256:" + "a" * 64,
    )

    assert report["w1_gate_closure_eligible"] is False
    assert report["model_comparison_decision"] == "NO_DECISION_INVALID_EVAL_SURFACE"
    assert "TASK_COUNT_CLAIM_MISMATCH" in gate_audit["hard_alerts"]
    assert "MISSING_BASELINE_PREDICTION_ARTIFACT" in gate_audit["hard_alerts"]

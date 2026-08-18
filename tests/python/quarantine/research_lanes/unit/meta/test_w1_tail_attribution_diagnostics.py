from __future__ import annotations

from typing import Any

import pytest

from pysrc.meta.w1_ranking_vs_expression_decomposition import (
    build_w1_ranking_vs_expression_decomposition,
)


def _comparison_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    gross_by_rank = [0.10, 0.08, 0.05, -0.01, -0.04]
    turnover_by_rank = [0.05, 0.10, 0.25, 0.45, 0.70]
    baseline_scores = [
        [0.55, 0.90, 0.10, 0.70, 0.30],
        [0.50, 0.85, 0.15, 0.65, 0.25],
    ]
    challenger_scores = [
        [0.95, 0.80, 0.60, 0.30, 0.10],
        [0.92, 0.82, 0.58, 0.28, 0.08],
    ]
    for fold_id in range(2):
        for rank in range(5):
            gross = gross_by_rank[rank] - 0.01 * float(fold_id)
            turnover = turnover_by_rank[rank]
            rows.append(
                {
                    "fold_id": fold_id,
                    "task_id": f"w1-tail-{fold_id}-{rank}",
                    "regime_class": "bull" if rank % 2 == 0 else "bear",
                    "baseline_score": baseline_scores[fold_id][rank],
                    "challenger_score": challenger_scores[fold_id][rank],
                    "query_gross_utility": gross,
                    "query_net_utility": gross - 0.03 * turnover,
                    "query_turnover": turnover,
                }
            )
    return rows


def _run(
    rows: list[dict[str, Any]],
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    str,
    dict[str, Any],
]:
    return build_w1_ranking_vs_expression_decomposition(
        comparison_rows=rows,
        run_identity={
            "run_id": "w1-tail-test",
            "data_fingerprint": "sha256:" + "b" * 64,
            "splits_fingerprint": "sha256:" + "c" * 64,
            "cost_assumptions_fingerprint": "sha256:" + "d" * 64,
            "task_pool_hash": "sha256:" + "a" * 64,
        },
    )


@pytest.mark.determinism("d1")
def test_tail_attribution_emits_deciles_and_grouped_quantile_rows(deterministic_seed: int) -> None:
    _ = deterministic_seed
    (doc, *_unused, audit) = _run(_comparison_rows())

    assert audit["status"] == "PASS"
    tail_doc = doc["tail_attribution_diagnostics"]
    assert tail_doc["target_surfaces"] == ["gross", "net"]
    decile_rows = tail_doc["decile_calibration_rows"]
    assert len(decile_rows) == 40
    assert {row["model_id"] for row in decile_rows} == {"baseline", "challenger"}
    assert {row["target_surface"] for row in decile_rows} == {"gross", "net"}
    gross_baseline_rows = [
        row
        for row in decile_rows
        if row["model_id"] == "baseline" and row["target_surface"] == "gross"
    ]
    assert sum(int(row["n_rows"]) for row in gross_baseline_rows) == 10
    quantile_rows = tail_doc["target_quantile_ic_rows"]
    assert any(row["quantile_type"] == "realized_target_decile" for row in quantile_rows)
    assert any(
        row["quantile_type"] == "tail_bucket" and row["quantile_label"] == "top_tail"
        for row in quantile_rows
    )
    assert any(
        row["quantile_type"] == "tail_bucket" and row["quantile_label"] == "middle"
        for row in quantile_rows
    )
    assert any(
        row["quantile_type"] == "tail_bucket" and row["quantile_label"] == "bottom_tail"
        for row in quantile_rows
    )
    assert doc["tail_attribution_audit"]["status"] == "PASS"


@pytest.mark.determinism("d1")
def test_tail_attribution_hit_rate_regret_and_overlap_match_fold_local_oracle(
    deterministic_seed: int,
) -> None:
    _ = deterministic_seed
    (doc, *_unused, audit) = _run(_comparison_rows())

    assert audit["status"] == "PASS"
    tail_doc = doc["tail_attribution_diagnostics"]
    hit_rate_rows = tail_doc["top_k_hit_rate_rows"]
    regret_rows = tail_doc["oracle_regret_rows"]
    overlap_rows = tail_doc["selected_task_overlap_rows"]

    challenger_global_k3_gross = next(
        row
        for row in hit_rate_rows
        if row["scope_type"] == "global"
        and row["model_id"] == "challenger"
        and row["k"] == 3
        and row["target_surface"] == "gross"
    )
    assert challenger_global_k3_gross["mean_hit_rate"] == pytest.approx(1.0)

    challenger_global_k3_regret = next(
        row
        for row in regret_rows
        if row["scope_type"] == "global"
        and row["model_id"] == "challenger"
        and row["k"] == 3
        and row["target_surface"] == "gross"
    )
    assert challenger_global_k3_regret["mean_oracle_regret"] == pytest.approx(0.0)

    assert any(
        row["overlap_type"] == "baseline_vs_challenger" and row["scope_type"] == "global"
        for row in overlap_rows
    )
    assert any(
        row["overlap_type"] == "model_vs_oracle"
        and row["model_id"] == "challenger"
        and row["target_surface"] == "gross"
        and row["scope_type"] == "global"
        for row in overlap_rows
    )


@pytest.mark.determinism("d1")
def test_tail_attribution_missing_required_target_fails_audit(deterministic_seed: int) -> None:
    _ = deterministic_seed
    rows = _comparison_rows()
    rows[0] = dict(rows[0])
    rows[0]["query_gross_utility"] = None
    (doc, *_unused, audit) = _run(rows)

    assert audit["status"] == "FAIL"
    assert doc["tail_attribution_audit"]["status"] == "FAIL"
    assert (
        "MISSING_QUERY_GROSS_UTILITY_FOR_DECOMPOSITION"
        in doc["tail_attribution_audit"]["hard_alerts"]
    )

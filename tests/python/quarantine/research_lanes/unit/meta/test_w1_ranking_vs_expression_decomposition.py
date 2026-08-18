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
                    "task_id": f"w1-dec-{fold_id}-{rank}",
                    "regime_class": "bull" if rank % 2 == 0 else "bear",
                    "baseline_score": baseline_scores[fold_id][rank],
                    "challenger_score": challenger_scores[fold_id][rank],
                    "query_gross_utility": gross,
                    "query_net_utility": gross - 0.03 * turnover,
                    "query_turnover": turnover,
                    "query_scores": [999.0, -999.0],
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
            "run_id": "w1-clean-test",
            "data_fingerprint": "sha256:" + "b" * 64,
            "splits_fingerprint": "sha256:" + "c" * 64,
            "cost_assumptions_fingerprint": "sha256:" + "d" * 64,
            "task_pool_hash": "sha256:" + "a" * 64,
        },
    )


@pytest.mark.determinism("d1")
def test_decomposition_emits_expected_sweeps_and_transform_invariants(
    deterministic_seed: int,
) -> None:
    _ = deterministic_seed
    (
        doc,
        fold_rows,
        regime_rows,
        bucket_rows,
        transform_rows,
        threshold_rows,
        top_k_rows,
        penalty_rows,
        summary_md,
        audit,
    ) = _run(_comparison_rows())

    assert audit["status"] == "PASS"
    assert doc["schema_version"] == "w1_ranking_vs_expression_decomposition.v1"
    assert doc["threshold_percentiles"] == [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    assert doc["top_k_values"] == [1, 3, 5, 10, 15, 20]
    assert doc["cost_penalty_lambdas"] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert {row["regime_class"] for row in regime_rows} == {"bear", "bull"}
    assert {row["fold_id"] for row in fold_rows} == {0, 1}
    assert {row["model_id"] for row in bucket_rows} == {"baseline", "challenger"}
    assert {row["transform_name"] for row in transform_rows} == {
        "cost_penalized_rank_weight",
        "rank_normalized_weight",
        "threshold_select_equal_weight",
        "top_k_equal_weight",
    }
    rank_rows = [row for row in transform_rows if row["transform_name"] == "rank_normalized_weight"]
    assert rank_rows
    assert all(row["min_weight_sum"] == pytest.approx(1.0) for row in rank_rows)
    assert all(row["max_weight_sum"] == pytest.approx(1.0) for row in rank_rows)
    k3_rows = [row for row in top_k_rows if row["k"] == 3]
    assert k3_rows
    assert all(row["selected_row_count"] == 6 for row in k3_rows)
    k10_rows = [row for row in top_k_rows if row["k"] == 10]
    assert k10_rows
    assert all(row["n_slices_evaluated"] == 0 for row in k10_rows)
    assert all(row["n_slices_insufficient"] == 2 for row in k10_rows)
    assert "ranking-vs-expression decomposition" in summary_md.lower()


@pytest.mark.determinism("d1")
def test_threshold_sweep_is_invariant_to_monotone_score_rescaling(deterministic_seed: int) -> None:
    _ = deterministic_seed
    rows = _comparison_rows()
    scaled_rows = [
        dict(row, challenger_score=float(row["challenger_score"]) * 10.0 + 7.0) for row in rows
    ]
    base = _run(rows)
    scaled = _run(scaled_rows)

    base_threshold_rows = {
        (row["model_id"], row["threshold_percentile"]): row
        for row in base[5]
        if row["model_id"] == "challenger"
    }
    scaled_threshold_rows = {
        (row["model_id"], row["threshold_percentile"]): row
        for row in scaled[5]
        if row["model_id"] == "challenger"
    }
    assert set(base_threshold_rows) == set(scaled_threshold_rows)
    for key, base_row in base_threshold_rows.items():
        scaled_row = scaled_threshold_rows[key]
        assert scaled_row["selected_row_count"] == base_row["selected_row_count"]
        assert scaled_row["mean_net_utility"] == pytest.approx(base_row["mean_net_utility"])


@pytest.mark.determinism("d1")
def test_penalty_lambda_zero_matches_unpenalized_ranking_and_missing_fields_fail(
    deterministic_seed: int,
) -> None:
    _ = deterministic_seed
    (
        doc,
        _fold_rows,
        _regime_rows,
        _bucket_rows,
        _transform_rows,
        _threshold_rows,
        _top_k_rows,
        penalty_rows,
        _summary_md,
        audit,
    ) = _run(_comparison_rows())

    assert audit["status"] == "PASS"
    base_hashes = {
        row["model_id"]: row["ranking_hash"]
        for row in doc["expression_diagnostics"]["base_rankings"]
    }
    lambda_zero_rows = [row for row in penalty_rows if row["penalty_lambda"] == 0.0]
    assert lambda_zero_rows
    for row in lambda_zero_rows:
        assert row["ranking_hash"] == base_hashes[row["model_id"]]

    bad_rows = _comparison_rows()
    bad_rows[0] = dict(bad_rows[0])
    bad_rows[0]["query_turnover"] = None
    *_unused, bad_audit = _run(bad_rows)
    assert bad_audit["status"] == "FAIL"
    assert "MISSING_QUERY_TURNOVER_FOR_DECOMPOSITION" in bad_audit["hard_alerts"]

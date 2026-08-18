from __future__ import annotations

import json
from datetime import date, timedelta

import pandas as pd
import pytest

from pysrc.artifact_registry.run_registry import RunRegistry
from pysrc.backtesting.contracts.types import PitMeta
from pysrc.backtesting.validation.statistical.pbo import compute_pbo
from pysrc.backtesting.validation.statistical.pbo_bridge import (
    CANONICAL_PBO_MODE,
    build_pbo_path_pairs,
)
from pysrc.preprocessor.graph.factory import register_builtin_ops
from pysrc.strategies.momentum.artifacts.cpcv_path_scores import compute_payload_hash
from pysrc.strategies.momentum.comparison import run_variant_comparison
from pysrc.strategies.pipeline_strategy import StrategyContext

pytestmark = [pytest.mark.integration, pytest.mark.determinism("d1")]


def _comparison_corpus() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    symbols = ("AAA", "BBB", "CCC")
    start = date(2023, 1, 1)
    for day_idx in range(320):
        valid_time = start + timedelta(days=day_idx)
        for symbol_idx, symbol in enumerate(symbols):
            oscillation = ((day_idx % 11) - 5) * (symbol_idx + 1) * 0.7
            reversal = (-1.0 if day_idx % 2 == 0 else 1.0) * (symbol_idx + 1) * 0.35
            trend = day_idx * (1.0 + (symbol_idx * 0.05))
            rows.append(
                {
                    "symbol": symbol,
                    "valid_time": valid_time,
                    "knowledge_time": valid_time,
                    "close": 100.0 + (symbol_idx * 8.0) + trend + oscillation + reversal,
                }
            )
    return pd.DataFrame(rows)


def _ctx(tmp_path) -> StrategyContext:
    return StrategyContext(
        prices=_comparison_corpus(),
        backend="pandas",
        cache_dir=tmp_path / "comparison-root",
        pit_provenance=PitMeta(
            as_of="2023-11-17T00:00:00",
            source="pysrc.data.dataview.DataView",
            knowledge_cutoff="2023-11-17",
        ),
    )


def test_run_variant_comparison_emits_parent_and_child_bundles(
    tmp_path, deterministic_seed
) -> None:
    _ = deterministic_seed
    register_builtin_ops()
    registry = RunRegistry(tmp_path / "registry")
    ctx = _ctx(tmp_path)

    result = run_variant_comparison(
        ctx,
        bundle_dir=tmp_path / "comparison-bundle",
        run_registry=registry,
    )

    root_bundle = result.bundle_dir
    report = json.loads(
        (root_bundle / "momentum_variant_comparison_report.json").read_text(encoding="utf-8")
    )
    comparison_stat_validity = json.loads(
        (root_bundle / "comparison_stat_validity.json").read_text(encoding="utf-8")
    )
    parent_plan = json.loads((root_bundle / "plan.json").read_text(encoding="utf-8"))
    root_splits = json.loads((root_bundle / "splits_manifest.json").read_text(encoding="utf-8"))
    expected_evaluations: list[dict[str, object]] = []

    assert parent_plan["config"]["strategy"] == "momentum_variant_comparison"
    assert report["schema_version"] == "1.0.0"
    assert report["cost_identity_valid"] is True
    assert report["split_identity_valid"] is True
    assert report["shared_cpcv"]["n_splits"] == 6
    assert report["shared_cpcv"]["n_test_splits"] == 2
    assert root_splits["split_method"] == "cpcv"
    assert report["comparison_stat_validity_path"] == "comparison_stat_validity.json"
    assert comparison_stat_validity["pbo"] == report["shared_pbo"]

    variant_summaries: dict[str, float] = {}
    for variant in ("xsec", "tsmom", "dual"):
        child_bundle = root_bundle / "variants" / variant
        child_plan = json.loads((child_bundle / "plan.json").read_text(encoding="utf-8"))
        child_splits = json.loads(
            (child_bundle / "splits_manifest.json").read_text(encoding="utf-8")
        )
        stat_validity = json.loads(
            (child_bundle / "stat_validity_report.json").read_text(encoding="utf-8")
        )
        execution_assumptions = json.loads(
            (child_bundle / "execution_assumptions.json").read_text(encoding="utf-8")
        )
        cpcv_path_scores = json.loads(
            (child_bundle / "cpcv_path_scores.json").read_text(encoding="utf-8")
        )
        expected_evaluations.extend(cpcv_path_scores["evaluations"])
        variant_summaries[variant] = cpcv_path_scores["summary"]["mean_out_of_sample_net_sharpe"]

        assert child_plan["config"]["strategy"] == "momentum"
        assert child_splits == root_splits
        assert stat_validity["schema_version"] == "v1"
        assert stat_validity["pbo"]["method"] == "unavailable"
        assert (
            execution_assumptions["cost_model_id"] == report["shared_cost_model"]["cost_model_id"]
        )
        assert report["variants"][variant]["cpcv_path_scores_path"] == str(
            child_bundle.relative_to(root_bundle) / "cpcv_path_scores.json"
        )
        assert report["variants"][variant]["child_stat_validity_pbo"] == stat_validity["pbo"]
        assert report["variants"][variant]["mean_in_sample_net_sharpe"] == pytest.approx(
            cpcv_path_scores["summary"]["mean_in_sample_net_sharpe"]
        )
        assert report["variants"][variant]["mean_out_of_sample_net_sharpe"] == pytest.approx(
            cpcv_path_scores["summary"]["mean_out_of_sample_net_sharpe"]
        )
        assert report["variants"][variant]["mean_turnover"] == pytest.approx(
            cpcv_path_scores["summary"]["mean_turnover"]
        )
        assert report["variants"][variant]["total_costs"] == pytest.approx(
            cpcv_path_scores["summary"]["total_costs"]
        )

    expected_path_pairs = build_pbo_path_pairs(expected_evaluations)
    expected_pbo = compute_pbo(expected_path_pairs, mode=CANONICAL_PBO_MODE)
    expected_ranking = [
        variant
        for variant, _score in sorted(
            variant_summaries.items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ]

    assert report["shared_path_score_surface_hash"] == compute_payload_hash(
        {"evaluations": expected_evaluations}
    )
    assert report["shared_pbo"] == expected_pbo
    assert [item["variant"] for item in report["ranking"]] == expected_ranking

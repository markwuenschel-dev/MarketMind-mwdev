from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pysrc.pipeline.contracts.p2 import P2Config
from pysrc.pipeline.panel.feature_grain_audit import audit_panel_grain
from pysrc.pipeline.panel.indicator_universe_builder import (
    build_indicator_universe,
    build_panel_supervision_frame,
    generate_synthetic_panel_frame,
    write_feature_universe_artifacts,
)
from pysrc.pipeline.panel.panel_feature_registry import (
    FeatureExclusionReason,
    classify_column,
)


@pytest.mark.determinism("d1")
def test_classify_identity_and_leakage_columns() -> None:
    frame = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "future_return": [0.1],
            "hindsight_best_child": ["alpha"],
            "ta_rsi_14": [55.0],
        }
    )
    assert (
        classify_column("ticker", frame["ticker"], source="t").exclusion_reason
        == FeatureExclusionReason.IDENTITY_COLUMN
    )
    assert (
        classify_column("future_return", frame["future_return"], source="t").exclusion_reason
        == FeatureExclusionReason.FUTURE_OR_LEAKAGE_COLUMN
    )
    assert (
        classify_column(
            "hindsight_best_child", frame["hindsight_best_child"], source="t"
        ).exclusion_reason
        == FeatureExclusionReason.ORACLE_COLUMN
    )
    assert (
        classify_column("ta_rsi_14", frame["ta_rsi_14"], source="t").exclusion_reason
        == FeatureExclusionReason.ELIGIBLE_FEATURE
    )


@pytest.mark.determinism("d1")
def test_classify_constant_numeric_column() -> None:
    frame = pd.DataFrame({"flat_feat": [1.0, 1.0, 1.0, 1.0]})
    assert (
        classify_column("flat_feat", frame["flat_feat"], source="t").exclusion_reason
        == FeatureExclusionReason.CONSTANT_COLUMN
    )


@pytest.mark.determinism("d1")
def test_synthetic_universe_discovers_all_ta_columns_not_shortlist(
    deterministic_seed: int,
) -> None:
    _ = deterministic_seed
    frame = generate_synthetic_panel_frame(n_rows=50, n_indicators=40, random_seed=42)
    config = P2Config(smoke_test=True, panel_model_output_dir="ignored")
    universe = build_indicator_universe(config, smoke_frame=frame)

    assert universe.eligible_feature_count >= 35
    assert "ta_feat_000" in universe.eligible_features
    assert "future_return" not in universe.eligible_features
    assert "hindsight_best_child" not in universe.eligible_features
    assert "ticker" not in universe.eligible_features


@pytest.mark.determinism("d1")
def test_write_feature_universe_artifacts_schema(tmp_path: Path) -> None:
    frame = generate_synthetic_panel_frame(n_rows=30, n_indicators=10, random_seed=7)
    config = P2Config(smoke_test=True)
    universe = build_indicator_universe(config, smoke_frame=frame)
    paths = write_feature_universe_artifacts(universe, tmp_path)

    report = json.loads(paths["feature_universe_report"].read_text(encoding="utf-8"))
    assert report["uses_full_discovered_feature_universe"] is True
    assert report["uses_full_indicator_universe"] is False
    assert report["grain_valid"] is True
    assert report["key_columns_used"] == ["date", "instrument", "interval"]
    assert report["row_grain"] == "ticker_date_interval"
    assert paths["unsupported_categorical_features"].is_file()
    assert paths["source_column_collision_report"].is_file()
    assert report["eligible_feature_count"] == universe.eligible_feature_count
    assert paths["eligible_features"].is_file()
    assert paths["excluded_features"].is_file()
    assert paths["model_feature_usage"].is_file()


@pytest.mark.determinism("d1")
def test_build_panel_supervision_frame_smoke(tmp_path: Path) -> None:
    config = P2Config(smoke_test=True, panel_model_output_dir=str(tmp_path / "panel_audit"))
    panel = build_panel_supervision_frame(config)

    assert panel.grain == "ticker_date_interval"
    assert panel.feature_names
    assert panel.provenance["feature_policy"] == "full_indicator_universe_v1"
    assert panel.provenance["uses_full_indicator_universe"] is False
    assert panel.provenance["uses_full_discovered_feature_universe"] is True
    assert (tmp_path / "panel_audit" / "feature_universe_report.json").is_file()


@pytest.mark.determinism("d1")
def test_grain_audit_flags_duplicate_keys() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01"],
            "instrument": ["AAA", "AAA"],
            "interval": ["daily", "daily"],
            "fold_id": ["fold_0", "fold_1"],
            "split": ["train", "test"],
            "x": [1.0, 2.0],
        }
    )
    audit = audit_panel_grain(frame)
    assert audit.valid is False
    assert audit.duplicate_key_count == 1
    assert list(audit.key_columns_used) == ["date", "instrument", "interval"]

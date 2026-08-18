from __future__ import annotations

import json
from pathlib import Path

import pytest

from pysrc.pipeline.contracts.p2 import P2Config
from pysrc.pipeline.panel.indicator_universe_builder import default_panel_model_output_dir
from pysrc.pipeline.panel.panel_model_runner import run_p2_panel_model


@pytest.mark.determinism("d1")
def test_panel_model_smoke_uses_full_eligible_universe(
    tmp_path: Path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    config = P2Config(
        smoke_test=True,
        panel_model_output_dir=str(tmp_path / "panel"),
        panel_model_families=("ridge",),
        random_seed=42,
    )
    paths = run_p2_panel_model(config)
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    assert report["eligible_feature_count"] > 0
    trained = [item for item in report["candidates"] if item.get("status") != "FAILED_TRAINING"]
    assert trained
    first = trained[0]
    assert first["feature_count"] == report["eligible_feature_count"]
    assert first["feature_policy"] == "full_indicator_universe_v1"

    usage_path = tmp_path / "panel" / "model_feature_usage.csv"
    assert usage_path.is_file()
    usage_text = usage_path.read_text(encoding="utf-8")
    assert "ridge" in usage_text
    assert "used_as_input" in usage_text


@pytest.mark.determinism("d1")
def test_panel_model_refuses_invalid_grain_report(tmp_path: Path) -> None:
    from pysrc.artifact_registry._atomic import atomic_write_json

    config = P2Config(
        smoke_test=False,
        panel_model_output_dir=str(tmp_path / "panel"),
    )
    audit_dir = default_panel_model_output_dir(config)
    audit_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        audit_dir / "feature_universe_report.json",
        {"grain_valid": False, "duplicate_key_sample_path": "duplicate_key_sample.csv"},
    )

    with pytest.raises(ValueError, match="grain_valid=false"):
        run_p2_panel_model(config)

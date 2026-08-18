from __future__ import annotations

import json
from pathlib import Path

import pytest

from pysrc.pipeline.materializers.indicator_panel import materialize_indicator_panel_from_frame
from pysrc.pipeline.orchestrator import OrchestratorConfig, run_orchestration
from tests.python.unit.pipeline.test_indicator_materializer import _synthetic_base_panel


@pytest.mark.determinism("d1")
def test_orchestrator_consumes_pipeline_indicator_product(tmp_path: Path) -> None:
    pytest.importorskip("pandas_ta_classic")
    processed_root = tmp_path / "processed"
    base = _synthetic_base_panel(rows=40)
    materialize_indicator_panel_from_frame(
        base,
        {"enabled": True, "processed_data_root": str(processed_root)},
    )

    exit_code, payload = run_orchestration(
        OrchestratorConfig(
            pipeline_product="indicator_panel",
            processed_data_root=processed_root,
            symbol="AAA",
            bundle_dir=tmp_path / "bundle",
            fast_sma=2,
            slow_sma=3,
        )
    )

    assert exit_code in {0, 1}
    assert payload["success"] is True
    assert (tmp_path / "bundle" / "plan.json").exists()
    plan = json.loads((tmp_path / "bundle" / "plan.json").read_text(encoding="utf-8"))
    assert plan["config"]["pipeline_product"] == "indicator_panel"
    assert (tmp_path / "bundle" / "backtest_result.json").exists()

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from pysrc.cli.marketmind import cli
from pysrc.pipeline.materializers.indicator_panel import materialize_indicator_panel_from_frame
from tests.python.unit.pipeline.test_indicator_materializer import _synthetic_base_panel


@pytest.mark.determinism("d1")
def test_backtest_run_from_pipeline_product(tmp_path: Path) -> None:
    pytest.importorskip("pandas_ta_classic")
    processed_root = tmp_path / "processed"
    bundle_dir = tmp_path / "bundle"
    materialize_indicator_panel_from_frame(
        _synthetic_base_panel(rows=40),
        {"enabled": True, "processed_data_root": str(processed_root)},
    )

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "backtest",
            "run",
            "--pipeline-product",
            "indicator_panel",
            "--processed-data-root",
            str(processed_root),
            "--symbol",
            "AAA",
            "--bundle-dir",
            str(bundle_dir),
            "--fast-sma",
            "2",
            "--slow-sma",
            "3",
        ],
    )

    assert result.exit_code in {0, 1}, result.output
    assert "bundle_path" in result.output
    assert (bundle_dir / "plan.json").is_file()


@pytest.mark.determinism("d1")
def test_backtest_run_rejects_dual_input_modes(tmp_path: Path) -> None:
    csv_path = tmp_path / "prices.csv"
    csv_path.write_text("date,close,symbol\n2024-01-01,100,SPY\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "backtest",
            "run",
            "--input-path",
            str(csv_path),
            "--pipeline-product",
            "indicator_panel",
        ],
    )

    assert result.exit_code != 0
    assert "not both" in result.output.lower()


@pytest.mark.determinism("d1")
def test_dataprep_run_accepts_research_config_without_backtest_metric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    deterministic_seed: int,
) -> None:
    _ = deterministic_seed
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text("version: 1.0.0" + chr(10) + "pipeline: {}" + chr(10), encoding="utf-8")
    calls: list[dict[str, object]] = []

    def fake_run_dataprep(run_cfg: object, backtest_metric: object = None) -> dict[str, str]:
        calls.append({"run_cfg": run_cfg, "backtest_metric": backtest_metric})
        return {"status": "success"}

    monkeypatch.setattr("pysrc.pipeline.dataprep_runtime.run_dataprep", fake_run_dataprep)

    runner = CliRunner()
    result = runner.invoke(cli, ["dataprep", "run", "-c", str(config_path)])

    assert result.exit_code == 0, result.output
    assert "success" in result.output
    assert calls == [{"run_cfg": {"version": "1.0.0", "pipeline": {}}, "backtest_metric": None}]

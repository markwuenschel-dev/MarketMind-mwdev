"""Executable boundary tests for the active product flow."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from pysrc.cli.marketmind import cli


@pytest.mark.determinism("d0")
def test_public_cli_tree_excludes_retired_workflows(deterministic_seed: int) -> None:
    _ = deterministic_seed

    assert set(cli.list_commands(None)) == {
        "artifacts",
        "backtest",
        "capabilities",
        "candidate-portfolios",
        "config",
        "dataprep",
        "panel",
        "run",
        "strategies",
        "tuning",
    }

    runner = CliRunner()
    result = runner.invoke(cli, ["candidate-portfolios", "build", "--help"])
    assert result.exit_code == 0
    assert "--source-run-id" in result.output
    assert "--output-dir" not in result.output


@pytest.mark.determinism("d0")
def test_prediction_registry_resolution_requires_a_complete_run(
    tmp_path, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    from pysrc.artifact_registry import ArtifactRegistry
    from pysrc.contracts import StandardizedPredictionArtifact

    registry = ArtifactRegistry(tmp_path)
    run_id = registry.begin_run({"lane": "panel"})
    prediction = StandardizedPredictionArtifact(
        schema_version="1",
        as_of="2026-01-02T00:00:00+00:00",
        data_lineage={"panel": "processed:panel-v1"},
        model_id="ridge",
        fold_id="fold-0",
        split="test",
        predictions=(
            {
                "instrument": "AAA",
                "decision_time": "2026-01-02T00:00:00+00:00",
                "value": 0.1,
            },
        ),
    )
    registry.register_json(run_id, "standardized_prediction", prediction)

    with pytest.raises(Exception, match="COMPLETE"):
        registry.resolve(run_id, "standardized_prediction", StandardizedPredictionArtifact)

    registry.complete_run(run_id)
    resolved = registry.resolve(run_id, "standardized_prediction", StandardizedPredictionArtifact)
    assert resolved.payload == prediction
    assert resolved.role == "standardized_prediction"

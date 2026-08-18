from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pysrc.artifact_registry.run_layout import (
    allocate_run_dir,
    cleanup_runs,
    resolve_artifact_roots,
)
from pysrc.pipeline.contracts.p2 import P2Config
from pysrc.pipeline.products import (
    pipeline_indicator_panel_path,
    require_pipeline_indicator_panel,
    resolve_pipeline_indicator_features_path,
    resolve_pipeline_product_paths,
)
from pysrc.pipeline.stages.preprocessing.indicators.pandas_ta_classic_provider import (
    load_pipeline_indicator_features,
)


@pytest.mark.determinism("d1")
def test_resolve_pipeline_product_paths_defaults() -> None:
    paths = resolve_pipeline_product_paths()
    assert paths.full_indicator_feature_panel.name == "full_indicator_feature_panel"
    assert paths.indicator_panel_parquet.name == "panel.parquet"


@pytest.mark.determinism("d1")
def test_resolve_pipeline_indicator_features_path_prefers_pipeline(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed"
    panel_dir = processed_root / "full_indicator_feature_panel"
    panel_dir.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-03"],
            "instrument": ["AAA", "AAA"],
            "rsi_14": [50.0, 55.0],
        }
    )
    panel_path = panel_dir / "panel.parquet"
    frame.to_parquet(panel_path, index=False)

    config = P2Config(processed_data_root=str(processed_root))
    path, source_kind = resolve_pipeline_indicator_features_path(config)
    assert source_kind == "pipeline_preprocessing"
    assert path == panel_path


@pytest.mark.determinism("d1")
def test_load_pipeline_indicator_features(tmp_path: Path) -> None:
    panel_path = tmp_path / "panel.parquet"
    pd.DataFrame(
        {
            "date": ["2024-01-02"],
            "instrument": ["AAA"],
            "rsi_14": [50.0],
        }
    ).to_parquet(panel_path, index=False)

    result = load_pipeline_indicator_features(panel_path)
    assert result.indicator_columns == ("rsi_14",)
    assert len(result.features) == 1


@pytest.mark.determinism("d1")
def test_require_pipeline_indicator_panel_raises_when_missing(
    tmp_path: Path,
) -> None:
    config = P2Config(processed_data_root=str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match="dataprep run"):
        require_pipeline_indicator_panel(config)


@pytest.mark.determinism("d1")
def test_pipeline_indicator_panel_path() -> None:
    config = P2Config(processed_data_root="data/processed")
    assert (
        pipeline_indicator_panel_path(config)
        .as_posix()
        .endswith("full_indicator_feature_panel/panel.parquet")
    )


@pytest.mark.determinism("d1")
def test_allocate_run_dir_and_cleanup(tmp_path: Path) -> None:
    roots = resolve_artifact_roots(tmp_path / "artifacts")
    smoke_dir = allocate_run_dir(lane="panel", smoke=True, roots=roots)
    regular_dir = allocate_run_dir(lane="panel", smoke=False, roots=roots)
    assert (smoke_dir / "reports").is_dir()
    assert (regular_dir / "run_meta.json").is_file()

    report = cleanup_runs(keep_latest=0, delete_smoke=True, roots=roots)
    assert smoke_dir.name in report.deleted_run_ids
    assert regular_dir.name in report.deleted_run_ids

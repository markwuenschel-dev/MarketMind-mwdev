from __future__ import annotations

import importlib
import json
import logging
import sys
from datetime import UTC, date, datetime

import pytest

pytestmark = pytest.mark.determinism("d1")


def test_loader_import_does_not_require_boto3(
    monkeypatch: pytest.MonkeyPatch, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    module_name = "pysrc.pipeline.stages.market_data.sources.massive_day_aggs_loader"
    monkeypatch.setitem(sys.modules, "boto3", None)
    sys.modules.pop(module_name, None)

    module = importlib.import_module(module_name)

    assert module.LOADER_VERSION


def test_parse_args_preserves_verbose_and_rejects_nonpositive_workers(
    deterministic_seed: int,
) -> None:
    _ = deterministic_seed
    from pysrc.pipeline.stages.market_data.sources.massive_day_aggs_loader import parse_args

    config = parse_args(["--start", "2024-01-02", "--end", "2024-01-03", "--verbose"])

    assert config.verbose is True
    with pytest.raises(SystemExit):
        parse_args(["--year", "2024", "--workers", "0"])


def test_write_manifest_uses_atomic_json_format(tmp_path, deterministic_seed: int) -> None:
    _ = deterministic_seed
    from pysrc.pipeline.stages.market_data.sources.massive_day_aggs_loader import (
        FileResult,
        LoaderConfig,
        write_manifest,
    )

    config = LoaderConfig(
        start_date=date(2024, 1, 2),
        end_date=date(2024, 1, 2),
        output_root=tmp_path,
        workers=1,
        overwrite=False,
        verbose=False,
    )
    result = FileResult(
        trade_date="2024-01-02",
        s3_key="key",
        parquet_path="out.parquet",
        status="downloaded",
        rows=1,
        bytes_compressed=2,
        bytes_parquet=3,
        s3_etag="etag",
        content_blake2b="hash",
        error=None,
        duration_seconds=0.1,
        partition_identity="massive.partition.v1:abc",
        knowledge_time_utc="2024-01-03T00:00:00+00:00",
    )

    manifest_path = write_manifest(config, [result])

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "massive_day_aggs_manifest.v1"
    assert payload["role"] == "raw_market_data_ingestion"
    assert payload["price_basis"] == "raw_unadjusted"
    assert payload["adjusted_ohlcv_source"] == "corporate_action_adjuster_only"
    assert payload["determinism_tier"] == "D1"
    assert payload["pit_contract"]["valid_time_column"] == "valid_time"
    assert payload["pit_contract"]["knowledge_time_column"] == "knowledge_time"
    assert payload["pit_contract"]["strategy_access"] == "DataView.as_of(T)_required"
    assert payload["execution_plan"]["cli_module"] == (
        "pysrc.pipeline.stages.market_data.sources.massive_day_aggs_loader"
    )
    assert "python" in payload["dependency_versions"]
    assert payload["summary"]["by_status"] == {"downloaded": 1}
    assert payload["files"][0]["partition_identity"] == "massive.partition.v1:abc"
    assert payload["files"][0]["knowledge_time_utc"] == "2024-01-03T00:00:00+00:00"
    assert not list(manifest_path.parent.glob("*.tmp"))


def test_main_treats_missing_vendor_objects_as_nonfatal(
    monkeypatch: pytest.MonkeyPatch, deterministic_seed: int
) -> None:
    _ = deterministic_seed
    from pysrc.pipeline.stages.market_data.sources.massive_day_aggs_loader import (
        FileResult,
        main,
    )

    monkeypatch.setattr(
        "pysrc.pipeline.stages.market_data.sources.massive_day_aggs_loader.run",
        lambda _config: [
            FileResult(
                trade_date="2025-01-09",
                s3_key="key",
                parquet_path="out.parquet",
                status="missing",
                rows=0,
                bytes_compressed=0,
                bytes_parquet=0,
                s3_etag="",
                content_blake2b="",
                error="S3 ClientError NoSuchKey",
                duration_seconds=0.1,
                partition_identity="massive.partition.v1:missing",
                knowledge_time_utc=None,
            )
        ],
    )
    monkeypatch.setattr(
        "pysrc.pipeline.stages.market_data.sources.massive_day_aggs_loader.write_manifest",
        lambda _config, _results: None,
    )

    assert main(["--start", "2025-01-09", "--end", "2025-01-09"]) == 0


def test_csv_conversion_adds_modular_pit_columns(deterministic_seed: int) -> None:
    _ = deterministic_seed
    gzip = pytest.importorskip("gzip")
    pa = pytest.importorskip("pyarrow")
    from pysrc.pipeline.stages.market_data.sources.massive_day_aggs_loader import (
        csv_gz_bytes_to_parquet_table,
    )

    raw_csv = (
        b"ticker,volume,open,close,high,low,window_start,transactions\n"
        b"SPY,100,10.0,10.5,10.7,9.9,1704153600000000000,12\n"
    )

    table = csv_gz_bytes_to_parquet_table(
        gzip.compress(raw_csv),
        trade_date=date(2024, 1, 2),
        knowledge_time=datetime(2024, 1, 3, tzinfo=UTC),
    )

    assert {"symbol", "date", "valid_time", "knowledge_time"} <= set(table.column_names)
    assert table.column("symbol").to_pylist() == ["SPY"]
    assert table.column("date").type == pa.date32()
    assert table.column("valid_time").type == pa.date32()
    assert table.column("knowledge_time").to_pylist() == [datetime(2024, 1, 3, tzinfo=UTC)]


def test_configure_logging_honors_verbose(deterministic_seed: int) -> None:
    _ = deterministic_seed
    from pysrc.pipeline.stages.market_data.sources.massive_day_aggs_loader import (
        configure_logging,
    )

    configure_logging(verbose=True)

    assert logging.getLogger().level == logging.DEBUG

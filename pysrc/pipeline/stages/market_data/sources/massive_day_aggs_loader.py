"""
Massive (formerly Polygon) flat-files loader for US Stocks SIP day aggregates.

Downloads a configurable date range of `us_stocks_sip/day_aggs_v1/YYYY/MM/YYYY-MM-DD.csv.gz`
from the Massive S3-compatible endpoint, converts each file to Parquet partitioned by date,
and emits a per-run manifest for reproducibility.

Endpoint reference: https://massive.com/docs/flat-files/quickstart

Design notes:
- Functional core / imperative shell: pure transform `csv_gz_bytes_to_parquet_table` is
  separated from S3 I/O and disk writes.
- Idempotent resume: skips files whose Parquet output already exists.
- Structured logging only; no print; no bare except.
- Credentials are read from environment variables, never hardcoded.
- Emits a manifest with per-file content hashes (BLAKE2b) and S3 ETags for lineage.
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import hashlib
import importlib.metadata
import io
import json
import logging
import os
import platform
import sys
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Final, Protocol, TypedDict, cast

from pysrc.ops.mm_logkit import configure_logger, get_logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MASSIVE_ENDPOINT: Final[str] = "https://files.massive.com"
MASSIVE_BUCKET: Final[str] = "flatfiles"
DATASET_PREFIX: Final[str] = "us_stocks_sip/day_aggs_v1"
SOURCE_ID: Final[str] = "massive.us_stocks_sip.day_aggs_v1"
MANIFEST_SCHEMA_VERSION: Final[str] = "massive_day_aggs_manifest.v1"
EXECUTION_PLAN_VERSION: Final[str] = "massive_day_aggs_ingestion_plan.v1"
DETERMINISM_TIER: Final[str] = "D1"
S3_MISSING_OBJECT_CODES: Final[frozenset[str]] = frozenset({"NoSuchKey", "404", "NotFound"})
DEFAULT_WORKERS: Final[int] = 8
DEFAULT_OUTPUT_ROOT: Final[Path] = Path("./data/massive/us_stocks_sip/day_aggs_v1")
LOADER_VERSION: Final[str] = "1.0.0"

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Typed errors
# ---------------------------------------------------------------------------


class LoaderError(Exception):
    """Base loader error."""


class CredentialsMissingError(LoaderError):
    """Raised when required env vars for S3 credentials are absent."""


class DownloadError(LoaderError):
    """Raised when an S3 object fails to download or decode."""


class ParquetConversionError(LoaderError):
    """Raised when CSV-to-Parquet conversion fails."""


class MissingDependencyError(LoaderError):
    """Raised when an optional loader dependency is absent."""


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FileResult:
    """Per-file outcome record for the manifest."""

    trade_date: str  # YYYY-MM-DD
    s3_key: str
    parquet_path: str
    status: str  # "downloaded" | "skipped_existing" | "missing" | "failed"
    rows: int
    bytes_compressed: int
    bytes_parquet: int
    s3_etag: str
    content_blake2b: str
    error: str | None
    duration_seconds: float
    partition_identity: str
    knowledge_time_utc: str | None


@dataclass(frozen=True)
class LoaderConfig:
    """Loader configuration; immutable."""

    start_date: date
    end_date: date
    output_root: Path
    workers: int
    overwrite: bool
    verbose: bool


class StreamingBody(Protocol):
    """Small S3 body protocol used by the loader."""

    def read(self) -> bytes:
        """Return object bytes."""


class S3ObjectResponse(TypedDict, total=False):
    """Subset of the S3 get_object response consumed by the loader."""

    Body: StreamingBody
    ETag: str


class S3Client(Protocol):
    """Small S3 client protocol used by the loader."""

    def get_object(self, *, Bucket: str, Key: str) -> S3ObjectResponse:
        """Return an S3 object response."""


class Boto3Session(Protocol):
    """Subset of a boto3 Session consumed by the loader."""

    def client(self, service_name: str, *, endpoint_url: str, config: object) -> object:
        """Build a service client."""


class Boto3SessionFactory(Protocol):
    """Factory protocol for boto3.Session."""

    def __call__(
        self,
        *,
        aws_access_key_id: str,
        aws_secret_access_key: str,
    ) -> Boto3Session:
        """Create a boto3 session."""


class Boto3Module(Protocol):
    """Subset of boto3 consumed by the loader."""

    Session: Boto3SessionFactory


class BotocoreConfigFactory(Protocol):
    """Factory protocol for botocore.config.Config."""

    def __call__(
        self,
        *,
        signature_version: str,
        retries: dict[str, object],
        connect_timeout: int,
        read_timeout: int,
    ) -> object:
        """Create botocore client config."""


class BotocoreConfigModule(Protocol):
    """Subset of botocore.config consumed by the loader."""

    Config: BotocoreConfigFactory


class ArrowTable(Protocol):
    """Subset of Arrow Table behavior consumed by the loader."""

    @property
    def num_rows(self) -> int:
        """Return the number of table rows."""

    @property
    def column_names(self) -> list[str]:
        """Return table column names."""

    def column(self, name: str) -> object:
        """Return a table column."""

    def append_column(self, name: str, column: object) -> ArrowTable:
        """Return a table with one extra column."""


class PyArrowModule(Protocol):
    """Subset of pyarrow consumed by the loader."""

    ArrowInvalid: type[Exception]
    ArrowIOError: type[Exception]

    def array(self, values: object, *, type: object) -> object:
        """Build an Arrow array."""

    def date32(self) -> object:
        """Return Arrow date32 type."""

    def timestamp(self, unit: str, *, tz: str) -> object:
        """Return Arrow timestamp type."""


class PyArrowCsvModule(Protocol):
    """Subset of pyarrow.csv consumed by the loader."""

    def read_csv(self, source: object) -> ArrowTable:
        """Read CSV bytes into an Arrow table."""


class PyArrowParquetModule(Protocol):
    """Subset of pyarrow.parquet consumed by the loader."""

    def write_table(self, table: ArrowTable, where: Path, *, compression: str) -> None:
        """Write an Arrow table to Parquet."""


# ---------------------------------------------------------------------------
# Functional core (pure)
# ---------------------------------------------------------------------------


def trading_dates(start: date, end: date) -> Iterator[date]:
    """
    Yield Mon–Fri dates inclusive in [start, end]. US market holidays are not
    excluded; missing dates simply 404 on S3 and are logged as failures.
    Skipping weekends avoids ~28% of pointless 404s per year.
    """
    cur = start
    one_day = timedelta(days=1)
    while cur <= end:
        if cur.weekday() < 5:  # 0=Mon ... 4=Fri
            yield cur
        cur += one_day


def s3_key_for(d: date) -> str:
    return f"{DATASET_PREFIX}/{d.year}/{d.month:02d}/{d.isoformat()}.csv.gz"


def parquet_path_for(output_root: Path, d: date) -> Path:
    return output_root / f"year={d.year}" / f"month={d.month:02d}" / f"{d.isoformat()}.parquet"


def partition_identity_for(d: date) -> str:
    payload = {
        "dataset_prefix": DATASET_PREFIX,
        "source_id": SOURCE_ID,
        "s3_key": s3_key_for(d),
        "trade_date": d.isoformat(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"massive.partition.v1:b2-256:{blake2b_hex(encoded)}"


def _load_pyarrow_modules() -> tuple[PyArrowModule, PyArrowCsvModule]:
    try:
        import pyarrow as pa  # type: ignore[import-untyped]
        import pyarrow.csv as pacsv  # type: ignore[import-untyped]
    except ImportError as e:
        raise MissingDependencyError(
            "Massive flat-file conversion requires optional dependency 'pyarrow'. "
            "Install project data-source dependencies before running this loader."
        ) from e
    return cast(PyArrowModule, pa), cast(PyArrowCsvModule, pacsv)


def _load_pyarrow_parquet_module() -> PyArrowParquetModule:
    try:
        import pyarrow.parquet as pq  # type: ignore[import-untyped]
    except ImportError as e:
        raise MissingDependencyError(
            "Massive flat-file Parquet writes require optional dependency 'pyarrow'. "
            "Install project data-source dependencies before running this loader."
        ) from e
    return cast(PyArrowParquetModule, pq)


def _utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _with_modular_market_data_columns(
    table: ArrowTable,
    *,
    trade_date: date,
    knowledge_time: datetime,
) -> ArrowTable:
    pa, _ = _load_pyarrow_modules()
    column_names = set(table.column_names)
    row_count = table.num_rows
    enriched = table

    if "symbol" not in column_names and "ticker" in column_names:
        enriched = enriched.append_column("symbol", enriched.column("ticker"))
        column_names.add("symbol")
    if "date" not in column_names:
        enriched = enriched.append_column(
            "date",
            pa.array([trade_date] * row_count, type=pa.date32()),
        )
        column_names.add("date")
    if "valid_time" not in column_names:
        enriched = enriched.append_column(
            "valid_time",
            pa.array([trade_date] * row_count, type=pa.date32()),
        )
        column_names.add("valid_time")
    if "knowledge_time" not in column_names:
        enriched = enriched.append_column(
            "knowledge_time",
            pa.array(
                [_utc_datetime(knowledge_time)] * row_count,
                type=pa.timestamp("us", tz="UTC"),
            ),
        )

    return enriched


def csv_gz_bytes_to_parquet_table(
    raw_gz: bytes,
    *,
    trade_date: date,
    knowledge_time: datetime,
) -> ArrowTable:
    """
    Pure: decompress gzip CSV bytes into an Arrow Table.

    The Polygon day_aggs_v1 schema (as of 2025) is:
        ticker, volume, open, close, high, low, window_start, transactions
    `window_start` is nanoseconds since epoch.

    We let pyarrow.csv infer types; the file is small enough per day that
    inference cost is negligible.
    """
    try:
        decompressed = gzip.decompress(raw_gz)
    except (OSError, EOFError) as e:
        raise ParquetConversionError(f"gzip decompression failed: {e}") from e

    pa, pacsv = _load_pyarrow_modules()
    try:
        table = pacsv.read_csv(io.BytesIO(decompressed))
    except (pa.ArrowInvalid, pa.ArrowIOError) as e:
        raise ParquetConversionError(f"CSV parse failed: {e}") from e

    return _with_modular_market_data_columns(
        cast(ArrowTable, table),
        trade_date=trade_date,
        knowledge_time=knowledge_time,
    )


def blake2b_hex(data: bytes) -> str:
    return hashlib.blake2b(data, digest_size=32).hexdigest()


def atomic_write_json(path: Path, payload: object) -> None:
    """Atomically write JSON without importing heavier artifact-registry package state."""
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


def dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {
        "python": platform.python_version(),
    }
    for package_name in ("boto3", "botocore", "pyarrow"):
        try:
            versions[package_name] = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            versions[package_name] = None
    return versions


# ---------------------------------------------------------------------------
# Imperative shell
# ---------------------------------------------------------------------------


def _load_boto3_modules() -> tuple[Boto3Module, BotocoreConfigModule]:
    try:
        import boto3  # type: ignore[import-untyped]
        import botocore.config  # type: ignore[import-untyped]
    except ImportError as e:
        raise MissingDependencyError(
            "Massive flat-file downloads require optional dependency 'boto3'. "
            "Install project data-source dependencies before running this loader."
        ) from e
    return cast(Boto3Module, boto3), cast(BotocoreConfigModule, botocore.config)


def _s3_download_error_types() -> tuple[type[Exception], ...]:
    try:
        from botocore.exceptions import BotoCoreError, ClientError  # type: ignore[import-untyped]
    except ImportError:
        return ()
    return (BotoCoreError, ClientError)


def _client_error_code(error: Exception) -> str:
    response = getattr(error, "response", None)
    if not isinstance(response, dict):
        return "Unknown"
    error_section = response.get("Error")
    if not isinstance(error_section, dict):
        return "Unknown"
    code = error_section.get("Code")
    return str(code) if code else "Unknown"


def build_s3_client() -> S3Client:
    access_key = os.environ.get("MASSIVE_S3_KEY")
    secret_key = os.environ.get("MASSIVE_S3_SECRET")
    if not access_key or not secret_key:
        raise CredentialsMissingError(
            "MASSIVE_S3_KEY and MASSIVE_S3_SECRET must be set in the environment."
        )

    boto3, botocore_config = _load_boto3_modules()
    session = boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
    return cast(
        S3Client,
        session.client(
            "s3",
            endpoint_url=MASSIVE_ENDPOINT,
            config=botocore_config.Config(
                signature_version="s3v4",
                retries={"max_attempts": 5, "mode": "adaptive"},
                connect_timeout=15,
                read_timeout=120,
            ),
        ),
    )


def ensure_runtime_dependencies() -> None:
    """Validate optional runtime dependencies before network retrieval begins."""
    _load_boto3_modules()
    _load_pyarrow_modules()
    _load_pyarrow_parquet_module()


def download_and_convert_one(
    s3: S3Client,
    d: date,
    output_root: Path,
    overwrite: bool,
) -> FileResult:
    """
    Process a single trading date:
      1. Skip if Parquet already exists and overwrite=False.
      2. GET the gzipped CSV from S3.
      3. Convert to Parquet (snappy).
      4. Write atomically (.tmp -> rename).
    """
    started = time.perf_counter()
    knowledge_time = datetime.now(UTC)
    key = s3_key_for(d)
    out_path = parquet_path_for(output_root, d)
    partition_identity = partition_identity_for(d)

    if out_path.exists() and not overwrite:
        logger.debug(
            "skip_existing", extra={"trade_date": d.isoformat(), "parquet_path": str(out_path)}
        )
        return FileResult(
            trade_date=d.isoformat(),
            s3_key=key,
            parquet_path=str(out_path),
            status="skipped_existing",
            rows=0,
            bytes_compressed=0,
            bytes_parquet=out_path.stat().st_size,
            s3_etag="",
            content_blake2b="",
            error=None,
            duration_seconds=time.perf_counter() - started,
            partition_identity=partition_identity,
            knowledge_time_utc=None,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    try:
        response = s3.get_object(Bucket=MASSIVE_BUCKET, Key=key)
        raw_gz: bytes = response["Body"].read()
        etag: str = response.get("ETag", "").strip('"')
    except _s3_download_error_types() as e:
        code = _client_error_code(e)
        msg = f"S3 ClientError {code} for {key}: {e}"
        if code in S3_MISSING_OBJECT_CODES:
            logger.info("object_missing", extra={"trade_date": d.isoformat(), "error": msg})
            return FileResult(
                trade_date=d.isoformat(),
                s3_key=key,
                parquet_path=str(out_path),
                status="missing",
                rows=0,
                bytes_compressed=0,
                bytes_parquet=0,
                s3_etag="",
                content_blake2b="",
                error=msg,
                duration_seconds=time.perf_counter() - started,
                partition_identity=partition_identity,
                knowledge_time_utc=None,
            )
        logger.warning("download_failed", extra={"trade_date": d.isoformat(), "error": msg})
        return FileResult(
            trade_date=d.isoformat(),
            s3_key=key,
            parquet_path=str(out_path),
            status="failed",
            rows=0,
            bytes_compressed=0,
            bytes_parquet=0,
            s3_etag="",
            content_blake2b="",
            error=msg,
            duration_seconds=time.perf_counter() - started,
            partition_identity=partition_identity,
            knowledge_time_utc=None,
        )

    content_hash = blake2b_hex(raw_gz)

    try:
        table = csv_gz_bytes_to_parquet_table(
            raw_gz,
            trade_date=d,
            knowledge_time=knowledge_time,
        )
        pq = _load_pyarrow_parquet_module()
        pq.write_table(table, tmp_path, compression="snappy")
        tmp_path.replace(out_path)
    except ParquetConversionError as e:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        msg = f"conversion failed: {e}"
        logger.warning("convert_failed", extra={"trade_date": d.isoformat(), "error": msg})
        return FileResult(
            trade_date=d.isoformat(),
            s3_key=key,
            parquet_path=str(out_path),
            status="failed",
            rows=0,
            bytes_compressed=len(raw_gz),
            bytes_parquet=0,
            s3_etag=etag,
            content_blake2b=content_hash,
            error=msg,
            duration_seconds=time.perf_counter() - started,
            partition_identity=partition_identity,
            knowledge_time_utc=knowledge_time.isoformat(),
        )

    bytes_parquet = out_path.stat().st_size
    rows = table.num_rows

    logger.info(
        "downloaded",
        extra={
            "trade_date": d.isoformat(),
            "rows": rows,
            "bytes_compressed": len(raw_gz),
            "bytes_parquet": bytes_parquet,
            "etag": etag,
        },
    )

    return FileResult(
        trade_date=d.isoformat(),
        s3_key=key,
        parquet_path=str(out_path),
        status="downloaded",
        rows=rows,
        bytes_compressed=len(raw_gz),
        bytes_parquet=bytes_parquet,
        s3_etag=etag,
        content_blake2b=content_hash,
        error=None,
        duration_seconds=time.perf_counter() - started,
        partition_identity=partition_identity,
        knowledge_time_utc=knowledge_time.isoformat(),
    )


def run(config: LoaderConfig) -> list[FileResult]:
    """Drive the parallel download with a thread pool."""
    ensure_runtime_dependencies()
    s3 = build_s3_client()
    dates = list(trading_dates(config.start_date, config.end_date))

    logger.info(
        "loader_start",
        extra={
            "start_date": config.start_date.isoformat(),
            "end_date": config.end_date.isoformat(),
            "trading_dates": len(dates),
            "workers": config.workers,
            "output_root": str(config.output_root),
            "overwrite": config.overwrite,
        },
    )

    results: list[FileResult] = []
    with ThreadPoolExecutor(max_workers=config.workers) as pool:
        futures = {
            pool.submit(download_and_convert_one, s3, d, config.output_root, config.overwrite): d
            for d in dates
        }
        for fut in as_completed(futures):
            results.append(fut.result())

    results.sort(key=lambda r: r.trade_date)
    return results


def write_manifest(config: LoaderConfig, results: list[FileResult]) -> Path:
    """Emit a JSON manifest with per-file lineage for reproducibility."""
    generated_at_utc = datetime.now(UTC)
    manifest_path = (
        config.output_root
        / "_manifests"
        / f"manifest_{config.start_date.isoformat()}_{config.end_date.isoformat()}_"
        f"{generated_at_utc.strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    by_status: dict[str, int] = {}
    for r in results:
        by_status[r.status] = by_status.get(r.status, 0) + 1

    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "loader_version": LOADER_VERSION,
        "source_id": SOURCE_ID,
        "role": "raw_market_data_ingestion",
        "price_basis": "raw_unadjusted",
        "adjusted_ohlcv_source": "corporate_action_adjuster_only",
        "determinism_tier": DETERMINISM_TIER,
        "dataset": DATASET_PREFIX,
        "endpoint": MASSIVE_ENDPOINT,
        "bucket": MASSIVE_BUCKET,
        "generated_at_utc": generated_at_utc.isoformat(),
        "seed_lineage": "none: external vendor retrieval uses no stochastic process",
        "pit_contract": {
            "pit_compliant": True,
            "valid_time_column": "valid_time",
            "knowledge_time_column": "knowledge_time",
            "valid_time_semantics": "daily_bar_trade_date",
            "knowledge_time_semantics": "utc_vendor_retrieval_time",
            "strategy_access": "DataView.as_of(T)_required",
            "raw_dataset_direct_access_allowed": False,
        },
        "pit_boundary": {
            "start_date": config.start_date.isoformat(),
            "end_date": config.end_date.isoformat(),
            "timezone": "UTC",
        },
        "execution_plan": {
            "plan_version": EXECUTION_PLAN_VERSION,
            "cli_module": "pysrc.pipeline.stages.market_data.sources.massive_day_aggs_loader",
            "cli_command_template": (
                "python -m pysrc.pipeline.stages.market_data.sources.massive_day_aggs_loader "
                "--start YYYY-MM-DD --end YYYY-MM-DD --output-root PATH"
            ),
            "partition_axis": "trade_date",
            "storage_layout": "year=YYYY/month=MM/YYYY-MM-DD.parquet",
            "parallel_workers": config.workers,
        },
        "registry_versions": {
            "source_registry": "standalone_raw_ingestion_cli.v1",
            "source_id": SOURCE_ID,
            "loader_version": LOADER_VERSION,
        },
        "dependency_versions": dependency_versions(),
        "artifact_lineage": {
            "content_hash_algorithm": "blake2b-256",
            "partition_identity_algorithm": "massive.partition.v1:b2-256",
            "s3_etag_recorded": True,
        },
        "config": {
            "start_date": config.start_date.isoformat(),
            "end_date": config.end_date.isoformat(),
            "output_root": str(config.output_root),
            "workers": config.workers,
            "overwrite": config.overwrite,
            "verbose": config.verbose,
        },
        "summary": {
            "files_total": len(results),
            "by_status": by_status,
            "rows_total": sum(r.rows for r in results),
            "bytes_compressed_total": sum(r.bytes_compressed for r in results),
            "bytes_parquet_total": sum(r.bytes_parquet for r in results),
        },
        "files": [dataclasses.asdict(r) for r in results],
    }

    atomic_write_json(manifest_path, payload)
    logger.info("manifest_written", extra={"manifest_path": str(manifest_path)})
    return manifest_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    configure_logger({"console": True, "level": level})
    logging.getLogger().setLevel(level)


def parse_args(argv: list[str]) -> LoaderConfig:
    p = argparse.ArgumentParser(
        description="Download Massive US Stocks SIP day aggregates and convert to Parquet.",
    )
    p.add_argument("--year", type=int, help="Convenience flag: download Jan 1 – Dec 31 of YEAR.")
    p.add_argument("--start", type=str, help="Start date YYYY-MM-DD (overrides --year).")
    p.add_argument("--end", type=str, help="End date YYYY-MM-DD (overrides --year).")
    p.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
        help=f"Output root directory (default: {DEFAULT_OUTPUT_ROOT}).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Parallel download workers (default: {DEFAULT_WORKERS}).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download and overwrite existing Parquet files.",
    )
    p.add_argument("--verbose", action="store_true", help="Enable DEBUG logging.")

    args = p.parse_args(argv)

    if args.start and args.end:
        start = date.fromisoformat(args.start)
        end = date.fromisoformat(args.end)
    elif args.year:
        start = date(args.year, 1, 1)
        end = date(args.year, 12, 31)
    else:
        p.error("Provide --year, or both --start and --end.")

    if end < start:
        p.error("--end must be on or after --start.")
    if args.workers < 1:
        p.error("--workers must be >= 1.")

    return LoaderConfig(
        start_date=start,
        end_date=end,
        output_root=Path(args.output_root),
        workers=args.workers,
        overwrite=args.overwrite,
        verbose=bool(args.verbose),
    )


def main(argv: list[str]) -> int:
    # parse_args is called before logging is configured so its errors go to stderr cleanly.
    config = parse_args(argv)
    configure_logging(verbose=config.verbose)

    try:
        results = run(config)
    except (CredentialsMissingError, MissingDependencyError) as e:
        logger.error("credentials_missing", extra={"error": str(e)})
        return 2

    write_manifest(config, results)

    failed = sum(1 for r in results if r.status == "failed")
    downloaded = sum(1 for r in results if r.status == "downloaded")
    skipped = sum(1 for r in results if r.status == "skipped_existing")
    missing = sum(1 for r in results if r.status == "missing")
    logger.info(
        "loader_complete",
        extra={"downloaded": downloaded, "skipped": skipped, "missing": missing, "failed": failed},
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

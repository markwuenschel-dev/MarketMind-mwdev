"""Build one-Parquet-per-day adjusted Massive OHLCV panels."""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Final

import pandas as pd  # type: ignore[import-untyped]

from pysrc.ops.mm_logkit import configure_logger, get_logger
from pysrc.pipeline.stages.market_data.corporate_actions import build_adjusted_ohlcv_panel
from pysrc.pipeline.stages.market_data.sources.massive_day_aggs_loader import (
    atomic_write_json,
    blake2b_hex,
    dependency_versions,
)

DEFAULT_RAW_ROOT: Final[Path] = Path("./data/massive/us_stocks_sip/day_aggs_v1")
DEFAULT_CORPORATE_ACTIONS_ROOT: Final[Path] = Path(
    "./data/massive/us_stocks_sip/corporate_actions_v1"
)
DEFAULT_OUTPUT_ROOT: Final[Path] = Path("./data/massive/us_stocks_sip/adjusted_day_panel_v1")
MANIFEST_SCHEMA_VERSION: Final[str] = "massive_adjusted_day_panel_manifest.v1"
BUILDER_VERSION: Final[str] = "1.0.0"
SOURCE_ID: Final[str] = "massive.us_stocks_sip.adjusted_day_panel_v1"

LOG = get_logger(__name__)


@dataclass(frozen=True)
class AdjustedPanelBuildConfig:
    """Configuration for local adjusted daily panel materialization."""

    start_date: date
    end_date: date
    raw_root: Path
    corporate_actions_root: Path
    output_root: Path
    splits_path: Path | None
    dividends_path: Path | None
    overwrite: bool
    verbose: bool


@dataclass(frozen=True)
class AdjustedPanelFileResult:
    """One adjusted daily output result."""

    trade_date: str
    path: str
    rows: int
    content_blake2b: str
    status: str
    duration_seconds: float


class AdjustedPanelBuildError(ValueError):
    """Raised when adjusted panel inputs are missing or invalid."""


def build_adjusted_daily_panels(config: AdjustedPanelBuildConfig) -> list[AdjustedPanelFileResult]:
    raw_paths = _raw_paths(config)
    if not raw_paths:
        raise AdjustedPanelBuildError(
            "No raw Massive day-aggs Parquet files found for requested range."
        )

    LOG.info(
        "adjusted_day_panel_start",
        extra={
            "start_date": config.start_date.isoformat(),
            "end_date": config.end_date.isoformat(),
            "raw_files": len(raw_paths),
            "output_root": str(config.output_root),
        },
    )
    raw_panel = _read_raw_panel(raw_paths)
    splits = _read_actions(_resolve_action_path(config, "splits"))
    dividends = _read_actions(_resolve_action_path(config, "dividends"))
    enriched = _attach_action_event_columns(
        build_adjusted_ohlcv_panel(raw_panel, splits=splits, dividends=dividends),
        splits=splits,
        dividends=dividends,
    )

    results: list[AdjustedPanelFileResult] = []
    for trade_date, day_frame in enriched.groupby("_panel_date", sort=True):
        day = pd.Timestamp(trade_date).date()
        out_path = _panel_path_for(config.output_root, day)
        if out_path.exists() and not config.overwrite:
            results.append(
                AdjustedPanelFileResult(
                    trade_date=day.isoformat(),
                    path=str(out_path),
                    rows=len(day_frame),
                    content_blake2b=blake2b_hex(out_path.read_bytes()),
                    status="skipped_existing",
                    duration_seconds=0.0,
                )
            )
            continue
        started = time.perf_counter()
        _atomic_write_parquet(day_frame.drop(columns=["_panel_date"]), out_path)
        results.append(
            AdjustedPanelFileResult(
                trade_date=day.isoformat(),
                path=str(out_path),
                rows=len(day_frame),
                content_blake2b=blake2b_hex(out_path.read_bytes()),
                status="written",
                duration_seconds=time.perf_counter() - started,
            )
        )

    write_manifest(config, results)
    LOG.info("adjusted_day_panel_complete", extra={"files": len(results)})
    return results


def write_manifest(
    config: AdjustedPanelBuildConfig, results: Sequence[AdjustedPanelFileResult]
) -> Path:
    manifest_path = (
        config.output_root
        / "_manifests"
        / f"manifest_{config.start_date.isoformat()}_{config.end_date.isoformat()}.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source_id": SOURCE_ID,
        "builder_version": BUILDER_VERSION,
        "role": "raw_plus_corporate_action_adjusted_daily_panel",
        "start_date": config.start_date.isoformat(),
        "end_date": config.end_date.isoformat(),
        "raw_root": str(config.raw_root),
        "corporate_actions_root": str(config.corporate_actions_root),
        "splits_path": str(_resolve_action_path(config, "splits")),
        "dividends_path": str(_resolve_action_path(config, "dividends")),
        "output_root": str(config.output_root),
        "price_usage_policy": {
            "return_signals": "adj_close",
            "forward_labels": "adj_close",
            "price_filters": "raw_close",
            "liquidity_filters": "raw_volume",
            "tradability_checks": "raw_open/raw_high/raw_low/raw_close/raw_volume",
        },
        "bar_quality_policy": {
            "repair_mode": "not_emitted",
            "bar_quality_mode": "none",
            "bar_quality_detector_version": None,
            "bar_quality_flags": [],
        },
        "dependency_versions": dependency_versions(),
        "files": [dataclasses.asdict(result) for result in results],
    }
    atomic_write_json(manifest_path, payload)
    return manifest_path


def parse_args(argv: list[str]) -> AdjustedPanelBuildConfig:
    parser = argparse.ArgumentParser(
        description="Build daily Massive OHLCV panels with corporate-action adjustment columns."
    )
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD.")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD.")
    parser.add_argument("--raw-root", default=str(DEFAULT_RAW_ROOT), help="Raw day_aggs_v1 root.")
    parser.add_argument(
        "--corporate-actions-root",
        default=str(DEFAULT_CORPORATE_ACTIONS_ROOT),
        help="Corporate actions root from massive_corporate_actions_loader.",
    )
    parser.add_argument("--splits-path", default=None, help="Explicit splits Parquet path.")
    parser.add_argument("--dividends-path", default=None, help="Explicit dividends Parquet path.")
    parser.add_argument(
        "--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Adjusted panel output root."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing daily panel files."
    )
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging.")
    args = parser.parse_args(argv)

    start = date.fromisoformat(str(args.start))
    end = date.fromisoformat(str(args.end))
    if end < start:
        parser.error("--end must be on or after --start.")

    return AdjustedPanelBuildConfig(
        start_date=start,
        end_date=end,
        raw_root=Path(args.raw_root),
        corporate_actions_root=Path(args.corporate_actions_root),
        output_root=Path(args.output_root),
        splits_path=Path(args.splits_path) if args.splits_path else None,
        dividends_path=Path(args.dividends_path) if args.dividends_path else None,
        overwrite=bool(args.overwrite),
        verbose=bool(args.verbose),
    )


def configure_logging(verbose: bool) -> None:
    configure_logger({"console": True, "level": "DEBUG" if verbose else "INFO"})


def main(argv: list[str]) -> int:
    config = parse_args(argv)
    configure_logging(config.verbose)
    try:
        build_adjusted_daily_panels(config)
    except AdjustedPanelBuildError as exc:
        LOG.error("adjusted_day_panel_failed", extra={"error": str(exc)})
        return 1
    return 0


def _raw_paths(config: AdjustedPanelBuildConfig) -> list[Path]:
    paths: list[Path] = []
    cur = config.start_date
    while cur <= config.end_date:
        path = (
            config.raw_root
            / f"year={cur.year}"
            / f"month={cur.month:02d}"
            / f"{cur.isoformat()}.parquet"
        )
        if path.exists():
            paths.append(path)
        cur += timedelta(days=1)
    return paths


def _read_raw_panel(paths: Sequence[Path]) -> pd.DataFrame:
    frames = [pd.read_parquet(path) for path in paths]
    return pd.concat(frames, ignore_index=True)


def _read_actions(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _resolve_action_path(config: AdjustedPanelBuildConfig, action: str) -> Path:
    explicit = config.splits_path if action == "splits" else config.dividends_path
    if explicit is not None:
        return explicit
    exact = config.corporate_actions_root / (
        f"{action}_{config.start_date.isoformat()}_{config.end_date.isoformat()}.parquet"
    )
    if exact.exists():
        return exact
    candidates = sorted(config.corporate_actions_root.glob(f"{action}_*.parquet"))
    if candidates:
        return candidates[-1]
    return exact


def _attach_action_event_columns(
    panel: pd.DataFrame,
    *,
    splits: pd.DataFrame,
    dividends: pd.DataFrame,
) -> pd.DataFrame:
    out = panel.copy()
    out["_panel_date"] = pd.to_datetime(out["date"]).dt.date
    out = _left_join_action(
        out,
        actions=splits,
        action_prefix="split",
        event_date_column="execution_date",
        columns=(
            "historical_adjustment_factor",
            "split_from",
            "split_to",
            "adjustment_type",
            "id",
        ),
    )
    out = _left_join_action(
        out,
        actions=dividends,
        action_prefix="dividend",
        event_date_column="ex_dividend_date",
        columns=(
            "historical_adjustment_factor",
            "cash_amount",
            "distribution_type",
            "id",
        ),
    )
    return out


def _left_join_action(
    panel: pd.DataFrame,
    *,
    actions: pd.DataFrame,
    action_prefix: str,
    event_date_column: str,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    event_output_column = f"{action_prefix}_{event_date_column}"
    prefixed_columns = [f"{action_prefix}_{column}" for column in columns]
    if actions.empty:
        out = panel.copy()
        out[event_output_column] = pd.NA
        for column in prefixed_columns:
            out[column] = pd.NA
        return out

    normalized = actions.copy()
    if "symbol" not in normalized.columns and "ticker" in normalized.columns:
        normalized["symbol"] = normalized["ticker"].astype(str)
    normalized["_panel_date"] = pd.to_datetime(
        normalized[event_date_column], errors="coerce"
    ).dt.date
    keep_columns = [
        "symbol",
        "_panel_date",
        event_date_column,
        *[column for column in columns if column in normalized.columns],
    ]
    normalized = normalized[keep_columns].dropna(subset=["symbol", "_panel_date"])
    rename = {event_date_column: event_output_column}
    rename.update(
        {column: f"{action_prefix}_{column}" for column in columns if column in normalized.columns}
    )
    normalized = normalized.rename(columns=rename)
    return panel.merge(normalized, on=["symbol", "_panel_date"], how="left", sort=False)


def _panel_path_for(output_root: Path, d: date) -> Path:
    return output_root / f"year={d.year}" / f"month={d.month:02d}" / f"{d.isoformat()}.parquet"


def _atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(tmp_path, index=False)
    tmp_path.replace(path)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

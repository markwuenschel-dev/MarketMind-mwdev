"""Massive corporate-actions loader for splits and dividends."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final, Literal, cast
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen

import pandas as pd  # type: ignore[import-untyped]

from pysrc.ops.mm_logkit import configure_logger, get_logger
from pysrc.pipeline.stages.market_data.sources.massive_day_aggs_loader import (
    atomic_write_json,
    blake2b_hex,
    dependency_versions,
)

MASSIVE_API_BASE: Final[str] = "https://api.massive.com"
DEFAULT_OUTPUT_ROOT: Final[Path] = Path("./data/massive/us_stocks_sip/corporate_actions_v1")
LOADER_VERSION: Final[str] = "1.0.0"
MANIFEST_SCHEMA_VERSION: Final[str] = "massive_corporate_actions_manifest.v1"
SOURCE_ID: Final[str] = "massive.us_stocks_sip.corporate_actions_v1"
ActionKind = Literal["splits", "dividends"]

LOG = get_logger(__name__)


class CorporateActionsLoaderError(Exception):
    """Base corporate-actions loader error."""


class CorporateActionsCredentialsError(CorporateActionsLoaderError):
    """Raised when the Massive REST API key is absent."""


class CorporateActionsFetchError(CorporateActionsLoaderError):
    """Raised when the Massive REST API response is invalid."""


@dataclass(frozen=True)
class CorporateActionsConfig:
    """Configuration for Massive corporate-actions retrieval."""

    start_date: date
    end_date: date
    output_root: Path
    actions: tuple[ActionKind, ...]
    tickers: tuple[str, ...]
    api_key_env: str
    limit: int
    verbose: bool


@dataclass(frozen=True)
class CorporateActionsFileResult:
    """Per-corporate-action output record."""

    action: ActionKind
    path: str
    rows: int
    content_blake2b: str
    status: str
    duration_seconds: float


def normalize_split_records(records: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    return _normalize_records(
        records,
        columns=(
            "ticker",
            "execution_date",
            "historical_adjustment_factor",
            "split_from",
            "split_to",
            "adjustment_type",
            "id",
        ),
        date_columns=("execution_date",),
        numeric_columns=("historical_adjustment_factor", "split_from", "split_to"),
    )


def normalize_dividend_records(records: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    return _normalize_records(
        records,
        columns=(
            "ticker",
            "ex_dividend_date",
            "historical_adjustment_factor",
            "cash_amount",
            "distribution_type",
            "id",
        ),
        date_columns=("ex_dividend_date",),
        numeric_columns=("historical_adjustment_factor", "cash_amount"),
    )


def fetch_corporate_actions(config: CorporateActionsConfig) -> dict[ActionKind, pd.DataFrame]:
    api_key = os.environ.get(config.api_key_env)
    if not api_key:
        raise CorporateActionsCredentialsError(
            f"{config.api_key_env} must be set in the environment."
        )

    output: dict[ActionKind, pd.DataFrame] = {}
    for action in config.actions:
        records: list[Mapping[str, object]] = []
        ticker_values: tuple[str | None, ...] = config.tickers or (None,)
        for ticker in ticker_values:
            records.extend(
                _fetch_action_records(action, config=config, api_key=api_key, ticker=ticker)
            )
        if action == "splits":
            output[action] = normalize_split_records(records)
        else:
            output[action] = normalize_dividend_records(records)
    return output


def write_corporate_actions(
    config: CorporateActionsConfig,
    frames: Mapping[ActionKind, pd.DataFrame],
) -> list[CorporateActionsFileResult]:
    results: list[CorporateActionsFileResult] = []
    for action, frame in frames.items():
        started = time.perf_counter()
        path = (
            config.output_root
            / f"{action}_{config.start_date.isoformat()}_{config.end_date.isoformat()}.parquet"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f".{path.name}.tmp")
        frame.to_parquet(tmp_path, index=False)
        tmp_path.replace(path)
        data = path.read_bytes()
        results.append(
            CorporateActionsFileResult(
                action=action,
                path=str(path),
                rows=len(frame),
                content_blake2b=blake2b_hex(data),
                status="written",
                duration_seconds=time.perf_counter() - started,
            )
        )
    return results


def write_manifest(
    config: CorporateActionsConfig, results: Sequence[CorporateActionsFileResult]
) -> Path:
    generated_at_utc = datetime.now(UTC)
    manifest_path = (
        config.output_root
        / "_manifests"
        / f"manifest_{config.start_date.isoformat()}_{config.end_date.isoformat()}_"
        f"{generated_at_utc.strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source_id": SOURCE_ID,
        "loader_version": LOADER_VERSION,
        "role": "corporate_action_factor_ingestion",
        "generated_at_utc": generated_at_utc.isoformat(),
        "api_base": MASSIVE_API_BASE,
        "actions": list(config.actions),
        "tickers": list(config.tickers),
        "start_date": config.start_date.isoformat(),
        "end_date": config.end_date.isoformat(),
        "factor_policy": {
            "split_factor_column": "historical_adjustment_factor",
            "dividend_factor_column": "historical_adjustment_factor",
            "fallback": "split_from/split_to only when split historical_adjustment_factor is absent",
        },
        "bar_quality_policy": {
            "provider_repair_claimed": False,
            "marketmind_bar_quality_detection_claimed": False,
        },
        "dependency_versions": dependency_versions(),
        "files": [dataclasses.asdict(result) for result in results],
    }
    atomic_write_json(manifest_path, payload)
    return manifest_path


def run(config: CorporateActionsConfig) -> list[CorporateActionsFileResult]:
    LOG.info(
        "corporate_actions_loader_start",
        extra={
            "start_date": config.start_date.isoformat(),
            "end_date": config.end_date.isoformat(),
            "actions": list(config.actions),
            "tickers": list(config.tickers),
        },
    )
    frames = fetch_corporate_actions(config)
    results = write_corporate_actions(config, frames)
    write_manifest(config, results)
    LOG.info("corporate_actions_loader_complete", extra={"files": len(results)})
    return results


def parse_args(argv: list[str]) -> CorporateActionsConfig:
    parser = argparse.ArgumentParser(description="Fetch Massive splits/dividends factor data.")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD.")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Output root.")
    parser.add_argument(
        "--actions",
        default="splits,dividends",
        help="Comma-separated actions: splits,dividends.",
    )
    parser.add_argument(
        "--ticker", action="append", default=[], help="Optional ticker filter; repeatable."
    )
    parser.add_argument(
        "--api-key-env", default="MASSIVE_API_KEY", help="Environment variable with REST API key."
    )
    parser.add_argument("--limit", type=int, default=1000, help="Massive page size.")
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging.")
    args = parser.parse_args(argv)

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    if end < start:
        parser.error("--end must be on or after --start.")
    if args.limit < 1:
        parser.error("--limit must be >= 1.")
    actions = _parse_actions(str(args.actions), parser)
    tickers = tuple(str(ticker).strip().upper() for ticker in args.ticker if str(ticker).strip())
    return CorporateActionsConfig(
        start_date=start,
        end_date=end,
        output_root=Path(args.output_root),
        actions=actions,
        tickers=tickers,
        api_key_env=str(args.api_key_env),
        limit=int(args.limit),
        verbose=bool(args.verbose),
    )


def configure_logging(verbose: bool) -> None:
    level = "DEBUG" if verbose else "INFO"
    configure_logger({"console": True, "level": level})


def main(argv: list[str]) -> int:
    config = parse_args(argv)
    configure_logging(config.verbose)
    try:
        run(config)
    except CorporateActionsCredentialsError as exc:
        LOG.error("corporate_actions_credentials_missing", extra={"error": str(exc)})
        return 2
    except CorporateActionsFetchError as exc:
        LOG.error("corporate_actions_fetch_failed", extra={"error": str(exc)})
        return 1
    return 0


def _parse_actions(raw: str, parser: argparse.ArgumentParser) -> tuple[ActionKind, ...]:
    values = tuple(part.strip().lower() for part in raw.split(",") if part.strip())
    invalid = [value for value in values if value not in {"splits", "dividends"}]
    if invalid:
        parser.error(f"Unsupported actions: {invalid}")
    if not values:
        parser.error("--actions must include at least one value.")
    return tuple(cast(ActionKind, value) for value in values)


def _fetch_action_records(
    action: ActionKind,
    *,
    config: CorporateActionsConfig,
    api_key: str,
    ticker: str | None,
) -> list[Mapping[str, object]]:
    date_field = "execution_date" if action == "splits" else "ex_dividend_date"
    endpoint = f"{MASSIVE_API_BASE}/stocks/v1/{action}"
    params: dict[str, str] = {
        f"{date_field}.gte": config.start_date.isoformat(),
        f"{date_field}.lte": config.end_date.isoformat(),
        "limit": str(config.limit),
        "sort": f"{date_field}.asc",
    }
    if ticker is not None:
        params["ticker"] = ticker

    records: list[Mapping[str, object]] = []
    next_url: str | None = _url_with_api_key(endpoint, params=params, api_key=api_key)
    while next_url is not None:
        payload = _get_json(next_url, api_key=api_key)
        raw_results = payload.get("results", [])
        if not isinstance(raw_results, list):
            raise CorporateActionsFetchError(f"Massive {action} response results must be a list.")
        records.extend(cast(list[Mapping[str, object]], raw_results))
        raw_next = payload.get("next_url")
        next_url = (
            _url_with_api_key(str(raw_next), params={}, api_key=api_key) if raw_next else None
        )
    return records


def _get_json(url: str, *, api_key: str) -> Mapping[str, object]:
    request = Request(
        url, headers={"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    )
    try:
        with urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise CorporateActionsFetchError(f"Massive REST request failed for {url}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CorporateActionsFetchError("Massive REST response must be a JSON object.")
    return cast(Mapping[str, object], payload)


def _url_with_api_key(endpoint: str, *, params: Mapping[str, str], api_key: str) -> str:
    parsed = urlparse(endpoint)
    query = dict(parse_qsl(parsed.query))
    query.update(params)
    query.setdefault("apiKey", api_key)
    return urlunparse(parsed._replace(query=urlencode(query)))


def _normalize_records(
    records: Sequence[Mapping[str, object]],
    *,
    columns: tuple[str, ...],
    date_columns: tuple[str, ...],
    numeric_columns: tuple[str, ...],
) -> pd.DataFrame:
    frame = pd.DataFrame(
        [{column: record.get(column) for column in columns} for record in records], columns=columns
    )
    for column in date_columns:
        frame[column] = pd.to_datetime(frame[column], errors="coerce").dt.date
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "ticker" in frame.columns:
        frame["ticker"] = frame["ticker"].astype("string").str.upper()
    return frame.sort_values([columns[0], date_columns[0]], kind="mergesort").reset_index(drop=True)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

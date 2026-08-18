import asyncio
import os
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from pysrc.pipeline.stages.market_data.exceptions import DataFetchError
from pysrc.pipeline.stages.market_data.sources.contracts import DataSource

try:
    from pysrc.pipeline.core.pipeline_core_registry import register_source
except ImportError:

    def register_source(_name: str):
        def _noop(cls):
            return cls

        return _noop


def _normalize_config(config: str | os.PathLike[str] | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config, (str, os.PathLike, Path)):
        file_path = str(config)
        return {
            "file_path": file_path,
            "format": (os.path.splitext(file_path)[1].lstrip(".") or "csv").lower(),
            "tail": False,
        }
    file_path = str(config["file_path"])
    return {
        "file_path": file_path,
        "format": config.get("format", os.path.splitext(file_path)[1].lstrip(".")).lower(),
        "tail": bool(config.get("tail", False)),
    }


@register_source("file")
class FileSource(DataSource):
    def __init__(self, config: str | os.PathLike[str] | dict[str, Any]):
        normalized = _normalize_config(config)
        super().__init__(normalized)
        self.file_path = str(normalized["file_path"])
        self.format = str(normalized["format"])
        self.tail = bool(normalized["tail"])

        if self.format not in {"csv", "json", "parquet", "ipc"}:
            raise ValueError(f"Unsupported file format: {self.format}")

    def _fallback_symbol(self, requested_symbol: str) -> str:
        if requested_symbol.strip():
            return requested_symbol.strip()
        return Path(self.file_path).stem

    async def get_historical(
        self,
        symbol: str,
        start: str,
        end: str,
        *,
        eager: bool = False,
    ) -> pl.LazyFrame | pl.DataFrame:
        try:
            if self.format == "csv":
                lf = pl.scan_csv(self.file_path, try_parse_dates=True)
            elif self.format == "json":
                lf = pl.scan_ndjson(self.file_path)
            elif self.format == "parquet":
                lf = pl.scan_parquet(self.file_path)
            else:
                lf = pl.scan_ipc(self.file_path)

            col_names = lf.collect_schema().names()
            low = {column.lower(): column for column in col_names}
            tcol = low.get("timestamp") or low.get("date") or low.get("time")
            if not tcol:
                raise DataFetchError("No timestamp or date column found")

            # Parse date/timestamp column: support ISO and US M/D/YYYY (e.g. fixture sample_spysrc.csv).
            # Cast to string first so we can try multiple formats; already-datetime cols become ISO-like.
            exprs: list[pl.Expr] = []
            if tcol == "timestamp":
                exprs.append(pl.col("timestamp").cast(pl.Datetime).alias("timestamp"))
            else:
                ts_str = pl.col(tcol).cast(pl.Utf8)
                exprs.append(
                    pl.coalesce(
                        ts_str.str.to_datetime(format="%m/%d/%Y", strict=False),
                        ts_str.str.to_datetime(format="%Y-%m-%d", strict=False),
                        ts_str.str.to_datetime(format="%Y-%m-%d %H:%M:%S", strict=False),
                    ).alias("timestamp")
                )
            if "symbol" not in col_names:
                exprs.append(pl.lit(self._fallback_symbol(symbol)).alias("symbol"))
            lf = lf.with_columns(exprs)
            col_names = lf.collect_schema().names()

            if "symbol" in col_names and symbol:
                lf = lf.filter(pl.col("symbol") == symbol)

            start_dt = datetime.fromisoformat(str(start)[:19])
            end_dt = datetime.fromisoformat(str(end)[:19])
            lf = lf.filter(pl.col("timestamp").is_between(start_dt, end_dt)).sort("timestamp")
            cols = set(lf.collect_schema().names())

            # Phase I approximation: valid_time == knowledge_time for daily file-sourced data.
            # Only add temporal columns when they are absent; never overwrite existing ones.
            temporal_exprs: list[pl.Expr] = []
            if "valid_time" not in cols:
                temporal_exprs.append(pl.col("timestamp").dt.date().alias("valid_time"))
            if "knowledge_time" not in cols:
                temporal_exprs.append(pl.col("timestamp").dt.date().alias("knowledge_time"))
            if temporal_exprs:
                lf = lf.with_columns(temporal_exprs)

            if eager:
                df = lf.collect()
                if df.is_empty():
                    raise DataFetchError(
                        f"No historical data found in {self.file_path} for {symbol or '*'}"
                    )
                return df
            return lf
        except DataFetchError:
            raise
        except (
            FileNotFoundError,
            OSError,
            ValueError,
            TypeError,
            pl.exceptions.PolarsError,
        ) as exc:
            raise DataFetchError(
                f"Failed to load historical data from {self.file_path}: {exc}"
            ) from exc

    def get_historical_sync(
        self,
        symbol: str,
        start: str,
        end: str,
        *,
        eager: bool = False,
    ) -> pl.LazyFrame | pl.DataFrame:
        return asyncio.run(self.get_historical(symbol, start, end, eager=eager))

    async def get_realtime(
        self, symbol: str, *, interval: float = 60.0
    ) -> AsyncIterator[pl.DataFrame]:
        if not self.tail:
            raise NotImplementedError(
                "Real-time data not supported for file sources without tail=True"
            )
        last_size = os.path.getsize(self.file_path)
        while True:
            await asyncio.sleep(interval)
            size = os.path.getsize(self.file_path)
            if size > last_size and self.format == "csv":
                import io

                with open(self.file_path, "rb") as handle:
                    handle.seek(last_size)
                    chunk = handle.read(size - last_size)
                df = pl.read_csv(io.BytesIO(chunk), try_parse_dates=True, has_header=False)
                low = {column.lower(): column for column in df.columns}
                tcol = low.get("timestamp") or (df.columns[0] if df.columns else None)
                if tcol and tcol != "timestamp":
                    df = df.rename({tcol: "timestamp"})
                df = df.with_columns(pl.col("timestamp").cast(pl.Datetime))
                if "symbol" not in df.columns:
                    df = df.with_columns(pl.lit(self._fallback_symbol(symbol)).alias("symbol"))
                if "symbol" in df.columns and symbol:
                    df = df.filter(pl.col("symbol") == symbol)
                if not df.is_empty():
                    yield df
                last_size = size

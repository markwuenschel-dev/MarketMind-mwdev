"""CLI for preprocessing OHLCV data."""

import argparse
import json
from pathlib import Path
from typing import Any

import click

from pysrc.preprocessor.core import build_features, load_ohlcv


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess OHLCV data")
    parser.add_argument("input", type=Path, help="Input CSV path")
    parser.add_argument("output", type=Path, help="Output CSV path")
    parser.add_argument("--sma", type=int, nargs="+", default=[20, 50], help="SMA windows")
    parser.add_argument("--rsi", type=int, default=14, help="RSI window")
    args: Any = parser.parse_args()

    # Load
    df = load_ohlcv(args.input)
    click.echo(f"Loaded {df.height} rows from {args.input}")

    # Transform — single graph pass, no intermediate DataFrames
    df = build_features(df, sma_windows=args.sma, rsi_window=args.rsi)

    # Save
    df.write_csv(args.output)
    click.echo(f"Wrote {df.height} rows with {len(df.columns)} columns to {args.output}")

    # Emit preprocessing_report.json alongside output — required by gate.py
    preprocessing_report = {
        "schema_version": "1.0.0",
        "steps": [{"name": "build_features", "sma_windows": args.sma, "rsi_window": args.rsi}],
        "timings": {},  # populate with wall-clock times if available
        "warnings": [],
    }
    report_path = args.output.parent / "preprocessing_report.json"
    report_path.write_text(json.dumps(preprocessing_report, indent=2))


if __name__ == "__main__":
    main()

"""CLI for running backtests and producing Appendix C-compliant run bundles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pysrc.ops.mm_logkit import get_logger
from pysrc.pipeline.orchestrator import OrchestratorConfig, run_orchestration

LOG = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run SMA crossover backtest")
    parser.add_argument("input", type=Path, help="Input CSV (OHLCV data)")
    parser.add_argument("--output-dir", type=Path, default=Path("bundles/latest"))
    parser.add_argument("--fast-sma", type=int, default=5)
    parser.add_argument("--slow-sma", type=int, default=10)
    args = parser.parse_args()

    exit_code, payload = run_orchestration(
        OrchestratorConfig(
            input_path=args.input,
            fast_sma=args.fast_sma,
            slow_sma=args.slow_sma,
            bundle_dir=args.output_dir,
        )
    )
    LOG.info(
        "cli_backtest_complete",
        bundle_path=payload.get("bundle_path"),
        success=payload.get("success"),
        validation_status=payload.get("validation", {}).get("status"),
    )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

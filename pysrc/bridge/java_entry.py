from __future__ import annotations

"""Java/UI-facing entry adapter for the Python pipeline.

This module is the external subprocess contract surface used by the Java
desktop UI. It is responsible for:

- Parsing CLI arguments from Java or the shell.
- Setting up logging / warning suppression for subprocess use.
- Delegating to the canonical Python orchestrator in pysrc.pipeline.orchestrator.
- Emitting JSON to stdout and mapping results to process exit codes.

Core orchestration logic lives in pysrc.pipeline.orchestrator.run_orchestration.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from pysrc.pipeline.orchestrator import OrchestratorConfig, run_orchestration


def _configure_process_silence() -> None:
    """Best-effort suppression of noisy logging for bridge subprocess runs."""
    # Capture stdout/stderr during import-time logging configuration.
    import io
    import warnings

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()

    try:
        warnings.filterwarnings("ignore")
        logging.disable(logging.CRITICAL)

        try:
            import structlog

            structlog.configure(
                wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL + 1),
            )
        except Exception:
            # structlog is optional; failure here should not break the bridge.
            pass
    finally:
        # Restore stdout/stderr so bridge output contract is preserved.
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run full backtest pipeline")
    parser.add_argument("input", type=Path, help="Input CSV path")
    parser.add_argument("--fast-sma", type=int, default=5)
    parser.add_argument("--slow-sma", type=int, default=10)
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=None,
        help="Optional bundle directory (defaults to bundles/<timestamp>)",
    )
    return parser


def _print_json(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload))
    sys.stdout.flush()


def main() -> int:
    """Entry point for Java/UI subprocesses."""
    _configure_process_silence()

    parser = _build_parser()
    args = parser.parse_args()

    try:
        config = OrchestratorConfig(
            input_path=args.input,
            fast_sma=args.fast_sma,
            slow_sma=args.slow_sma,
            bundle_dir=args.bundle_dir,
        )
        exit_code, payload = run_orchestration(config)
        _print_json(payload)
        return exit_code
    except Exception as exc:  # noqa: BLE001
        error_payload: dict[str, Any] = {
            "success": False,
            "error": str(exc),
        }
        _print_json(error_payload)
        # Preserve existing contract where unexpected errors map to exit code 2.
        return 2


if __name__ == "__main__":
    sys.exit(main())

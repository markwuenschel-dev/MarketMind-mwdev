#!/usr/bin/env python3
"""MLN-06 / MLN-07 static audits for CI (governed RG-09 bundles and pilot config)."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path

from pysrc.meta.rg09_threshold_catalog import RG09_CONFIG_THRESHOLD_SPECS

MLN06_TRIPLE: tuple[str, ...] = (
    "task_manifest.json",
    "meta_validity_report.json",
    "execution_assumptions.json",
)


def _die(message: str) -> None:
    print(f"mln_governance_audit: {message}", file=sys.stderr)
    raise SystemExit(1)


def audit_governed_bundle_dir(bundle_dir: Path) -> None:
    missing = [name for name in MLN06_TRIPLE if not (bundle_dir / name).is_file()]
    if missing:
        _die(f"bundle {bundle_dir} missing MLN-06 files: {', '.join(missing)}")


def audit_pilot_config(config_path: Path) -> None:
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        _die(f"{config_path} root must be an object")
    for field_name, spec in RG09_CONFIG_THRESHOLD_SPECS.items():
        if not spec.gate_critical:
            continue
        if field_name not in raw:
            _die(f"{config_path} missing gate-critical field {field_name!r}")
        value = raw[field_name]
        if not isinstance(value, Mapping):
            _die(
                f"{config_path} field {field_name!r} must be a threshold object with threshold_id "
                f"(gate-critical); got {type(value).__name__}"
            )
        tid = value.get("threshold_id")
        if tid is None or not str(tid).strip():
            _die(f"{config_path} field {field_name!r} missing non-empty threshold_id")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_bundle = sub.add_parser("bundle-dir", help="Require MLN-06 triple in a directory")
    p_bundle.add_argument("path", type=Path)

    p_cfg = sub.add_parser("pilot-config", help="Require gate-critical threshold_id objects")
    p_cfg.add_argument("path", type=Path)

    args = parser.parse_args(argv)
    if args.command == "bundle-dir":
        if not args.path.is_dir():
            _die(f"not a directory: {args.path}")
        audit_governed_bundle_dir(args.path)
    elif args.command == "pilot-config":
        if not args.path.is_file():
            _die(f"not a file: {args.path}")
        audit_pilot_config(args.path)
    else:  # pragma: no cover
        _die("unknown command")


if __name__ == "__main__":
    main()

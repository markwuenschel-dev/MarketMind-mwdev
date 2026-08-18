#!/usr/bin/env bash
set -euo pipefail

# Usage: devtools/scan_helpers.sh [TESTS_DIR] [SRC_DIR]
TESTS_DIR="${1:-tests/python}"
SRC_DIR="${2:-py}"

# Add or remove helper names here
HELPERS=(
  "_get_or_register_gauge"
  # "another_helper"
  # "missing_module_func"
)

echo "== Tooling =="
if command -v pylint >/dev/null 2>&1; then
  echo "pylint: $(pylint --version | head -n1)"
  HAVE_PYLINT=1
else
  echo "pylint: not found (static analysis will be skipped)"
  HAVE_PYLINT=0
fi
echo

echo "== 1) Static scan for undefined names/import errors =="
if [[ "$HAVE_PYLINT" -eq 1 ]]; then
  # E0602 undefined-variable, E0401 import-error
  pylint -j 0 \
    --disable=all \
    --enable=E0602,E0401 \
    "$TESTS_DIR" "$SRC_DIR" || true
else
  echo "(pylint unavailable) Skipping; you can: pip install pylint"
fi
echo

echo "== 2) Where are helpers used/defined? =="
for h in "${HELPERS[@]}"; do
  echo "-- Calls of ${h}(...):"
  grep -Rns -- "${h}\(" "$SRC_DIR" "$TESTS_DIR" || echo "   (none)"
  echo "-- Definitions of def ${h}(...):"
  grep -Rns -- "def[[:space:]]\+${h}\b" "$SRC_DIR" "$TESTS_DIR" || echo "   (none)"
  echo
done

echo "== 3) Order check in metrics.py (definition before use) =="
MP="${SRC_DIR}/pipeline/stages/cleaning/core/metrics.py"
if [[ -f "$MP" ]]; then
  DEF_LINE="$(grep -nE '^def[[:space:]]+_get_or_register_gauge\b' "$MP" | cut -d: -f1 | head -n1 || true)"
  USE_LINE="$(grep -n "_get_or_register_gauge('streaming_cleaner_latency'" "$MP" | cut -d: -f1 | head -n1 || true)"
  if [[ -n "${DEF_LINE:-}" && -n "${USE_LINE:-}" ]]; then
    if (( DEF_LINE < USE_LINE )); then
      echo "OK: helper defined at line ${DEF_LINE}, used first at line ${USE_LINE}"
    else
      echo "WARN: helper is used at line ${USE_LINE} BEFORE it is defined at line ${DEF_LINE}"
    fi
  else
    echo "INFO: could not find helper definition or first use in ${MP}"
  fi
else
  echo "INFO: ${MP} not found (skipping)"
fi
echo

echo "== 4) Import smoke test =="
python - <<'PY'
import importlib, sys
mods = [
  "py.pipeline.stages.cleaning.core.metrics",
]
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"OK import {m}; helper present: {hasattr(mod, '_get_or_register_gauge')}")
    except Exception as e:
        print(f"ERR importing {m}: {e.__class__.__name__}: {e}")
        sys.exit(1)
PY

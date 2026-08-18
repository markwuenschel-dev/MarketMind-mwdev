#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

TEST_PATH="${TEST_PATH:-tests/python}"
MARKERS="${MARKERS:-}"
PYTEST_ARGS="${PYTEST_ARGS:-}"
PYTEST_BASETEMP="${PYTEST_BASETEMP:-${TMPDIR:-/tmp}/marketmind-pytest-${USER:-user}-$$}"
VERBOSE="${VERBOSE:-0}"
WORKERS="${WORKERS:-auto}"
DIST="${DIST:-worksteal}"
BYTECODE="${BYTECODE:-1}"
CLEAN="${CLEAN:-0}"
LASTFAIL="${LASTFAIL:-0}"
DRY_RUN="${DRY_RUN:-0}"
PLUGIN_AUTOLOAD="${PLUGIN_AUTOLOAD:-1}"
PYTEST_BIN="${PYTEST_BIN:-pytest}"

if [[ "${BYTECODE}" != "0" ]]; then
  export PYTHONDONTWRITEBYTECODE=1
fi

if [[ "${PLUGIN_AUTOLOAD}" == "0" ]]; then
  export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
fi

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<EOF
Usage: ./tests/run_tests.sh [pytest selectors/flags...]

Thin pytest launcher. Repo policy lives in pyproject.toml.

Environment variables:
  TEST_PATH         target path (default: ${TEST_PATH})
  MARKERS           pytest -m expression, e.g. "not net"
  PYTEST_ARGS       extra raw pytest args
  PYTEST_BASETEMP   pytest temporary root (default: ${PYTEST_BASETEMP})
  VERBOSE           0=quiet, 1=-v, 2=-vv
  WORKERS           xdist workers; "auto" by default
  DIST              xdist distribution strategy (default: ${DIST})
  BYTECODE          1 => set PYTHONDONTWRITEBYTECODE=1
  CLEAN             1 => remove common test/cache artifacts before run
  LASTFAIL          1 => add --last-failed --last-failed-no-failures=all
  DRY_RUN           1 => print command and exit
  PLUGIN_AUTOLOAD   0 => disable pytest plugin autoload
  PYTEST_BIN        pytest binary (default: ${PYTEST_BIN})

Examples:
  ./tests/run_tests.sh
  MARKERS="not net" ./tests/run_tests.sh
  MARKERS="smoke and not net" ./tests/run_tests.sh tests/python/unit/
  PYTEST_ARGS="--no-cov -x" ./tests/run_tests.sh tests/python/unit/backtest/
EOF
  exit 0
fi

if ! command -v "${PYTEST_BIN}" >/dev/null 2>&1; then
  echo "Error: ${PYTEST_BIN} not found."
  exit 1
fi

have_xdist=0
if python3 - <<'PY' >/dev/null 2>&1
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("xdist") else 1)
PY
then
  have_xdist=1
fi

if [[ "${WORKERS}" == "auto" ]]; then
  cores="$(python3 - <<'PY' 2>/dev/null || echo 1
import os
print(os.cpu_count() or 1)
PY
)"
  if [[ "${cores}" =~ ^[0-9]+$ && "${cores}" -ge 4 ]]; then
    WORKERS=$((cores - 1))
  else
    WORKERS=2
  fi
fi

if [[ "${CLEAN}" != "0" ]]; then
  echo "==> Cleaning test artifacts"
  rm -f .coverage coverage.json coverage.xml || true
  rm -rf .pytest_cache .mypy_cache .ruff_cache .hypothesis htmlcov || true
  find "${TEST_PATH}" -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
fi

cmd=("${PYTEST_BIN}")

case "${VERBOSE}" in
  0) ;;
  1) cmd+=(-v) ;;
  2) cmd+=(-vv) ;;
  *) cmd+=(-v) ;;
esac

if [[ "${LASTFAIL}" != "0" ]]; then
  cmd+=(--last-failed --last-failed-no-failures=all)
fi

if [[ -n "${MARKERS}" ]]; then
  cmd+=(-m "${MARKERS}")
fi

if [[ "${have_xdist}" -eq 1 ]]; then
  cmd+=(-n "${WORKERS}" --dist="${DIST}")
fi

cmd+=(--basetemp "${PYTEST_BASETEMP}")
cmd+=("${TEST_PATH}")

if [[ $# -gt 0 ]]; then
  cmd+=("$@")
fi

if [[ -n "${PYTEST_ARGS}" ]]; then
  # shellcheck disable=SC2206
  extra_args=(${PYTEST_ARGS})
  cmd+=("${extra_args[@]}")
fi

echo "==> Repo root: ${REPO_ROOT}"
echo "==> Test path: ${TEST_PATH}"
echo "==> Pytest basetemp: ${PYTEST_BASETEMP}"
echo "==> xdist: $([[ "${have_xdist}" -eq 1 ]] && echo on || echo off)"
[[ -n "${MARKERS}" ]] && echo "==> Markers: ${MARKERS}"

if [[ "${DRY_RUN}" != "0" ]]; then
  printf 'DRY RUN =>'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

"${cmd[@]}"
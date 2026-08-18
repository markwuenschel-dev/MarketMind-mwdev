"""Structural guard for docs/DECISIONS.md — the single ADR+PDR home.

Checks *form*, not content, so it does not go stale as decisions evolve: the
file exists, every ADR/PDR entry carries a valid Status, and IDs are unique
within each sequence. See AGENTS.md §5.2 and the DECISIONS.md header.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
DECISIONS = ROOT / "docs" / "DECISIONS.md"

# Entry heading, e.g. "## ADR-002 · Demolish-in-place keep-list".
_ENTRY = re.compile(r"^## (ADR|PDR)-(\d+) ·", re.MULTILINE)
# Lifecycle rule: only these two statuses ever live in the file.
_VALID_STATUS = {"Accepted", "Proposed"}


def _blocks() -> list[tuple[str, int, str]]:
    """Return (kind, number, body) for each decision entry, body = heading..next."""
    text = DECISIONS.read_text(encoding="utf-8")
    matches = list(_ENTRY.finditer(text))
    out: list[tuple[str, int, str]] = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out.append((m.group(1), int(m.group(2)), text[m.start() : end]))
    return out


@pytest.mark.determinism("d1")
def test_decisions_file_exists() -> None:
    assert DECISIONS.is_file(), "docs/DECISIONS.md must exist as the single ADR+PDR home"


@pytest.mark.determinism("d1")
def test_has_at_least_one_entry() -> None:
    assert _blocks(), "DECISIONS.md must contain at least one ADR/PDR entry"


@pytest.mark.determinism("d1")
def test_every_entry_has_a_valid_status() -> None:
    for kind, num, body in _blocks():
        m = re.search(r"^Status:\s*(\w+)", body, re.MULTILINE)
        assert m is not None, f"{kind}-{num:03d} is missing a 'Status:' line"
        assert m.group(1) in _VALID_STATUS, (
            f"{kind}-{num:03d} has Status '{m.group(1)}'; only {sorted(_VALID_STATUS)} "
            "may live in DECISIONS.md (superseded/retired entries are deleted)"
        )


@pytest.mark.determinism("d1")
def test_ids_unique_within_each_sequence() -> None:
    seen: dict[str, set[int]] = {"ADR": set(), "PDR": set()}
    for kind, num, _ in _blocks():
        assert num not in seen[kind], f"duplicate {kind}-{num:03d} in DECISIONS.md"
        seen[kind].add(num)

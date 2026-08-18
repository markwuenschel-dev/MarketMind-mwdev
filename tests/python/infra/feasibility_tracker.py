# tests/python/infra/feasibility_tracker.py
# In-memory tracker for suspected infeasible branches, with optional persistence.

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


class FeasibilityTracker:
    def __init__(self, max_attempts: int = 5, cache_file: Path | None = None):
        self.max_attempts = max_attempts
        self.cache_file = cache_file
        self._attempts: dict[tuple[str, int], int] = defaultdict(int)
        if self.cache_file and self.cache_file.exists():
            self.load()

    def should_attempt(self, file: str, lineno: int) -> bool:
        return self._attempts[(file, lineno)] < self.max_attempts

    def record_attempt(self, file: str, lineno: int, covered: bool) -> None:
        key = (file, lineno)
        if covered:
            self._attempts[key] = 0
        else:
            self._attempts[key] += 1

    def suspected_infeasible(self) -> list[tuple[str, int]]:
        return [k for k, v in self._attempts.items() if v >= self.max_attempts]

    def save(self) -> None:
        if not self.cache_file:
            return
        data = {f"{k[0]}:{k[1]}": v for k, v in self._attempts.items()}
        self.cache_file.write_text(json.dumps(data, indent=2))

    def load(self) -> None:
        if not self.cache_file or not self.cache_file.exists():
            return
        raw = json.loads(self.cache_file.read_text())
        for key_str, count in raw.items():
            file, line = key_str.rsplit(":", 1)
            self._attempts[(file, int(line))] = count

    def _group_by_file(self) -> dict[str, int]:
        by_file: dict[str, int] = defaultdict(int)
        for (file, _), count in self._attempts.items():
            if count >= self.max_attempts:
                by_file[file] += 1
        return dict(by_file)

    def get_report(self) -> dict[str, Any]:
        total_branches = len(self._attempts)
        total_attempts = sum(self._attempts.values())
        suspected = self.suspected_infeasible()
        suspected_count = len(suspected)
        suspected_pct = (suspected_count / total_branches * 100.0) if total_branches else 0.0

        return {
            "total_branches_attempted": total_branches,
            "total_attempts": total_attempts,
            "suspected_infeasible_count": suspected_count,
            "suspected_infeasible_pct": suspected_pct,
            "by_file": self._group_by_file(),
            "suspected_infeasible": suspected,
        }

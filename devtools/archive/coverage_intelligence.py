# devtools/coverage_intelligence.py
# High-level integration of coverage export, branch typology, and feasibility tracking.

from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List

import json

from devtools.coverage_export import export_file_coverage
from devtools.branch_typology import classify_from_export
from tests.python.infra.feasibility_tracker import FeasibilityTracker


class CoverageIntelligence:
    # Orchestrates coverage export, typology classification, and feasibility tracking.

    def __init__(self, root: Path, cache_file: Path | None = None):
        self.root = root
        self.tracker = FeasibilityTracker(max_attempts=5, cache_file=cache_file)

    def analyze(
        self,
        coverage_data_file: Path,
        target_files: List[Path],
    ) -> Dict[str, Any]:
        # coverage_data_file is usually ".coverage" from a pytest run.
        coverage_report = export_file_coverage(coverage_data_file, target_files)

        export_path = self.root / "coverage_export.json"
        export_path.write_text(json.dumps(coverage_report, indent=2))

        typology = classify_from_export(export_path, self.root)

        for filename, branches in typology.items():
            missing_lines = set(coverage_report[filename]["missing_lines"])
            for b in branches:
                covered = b["lineno"] not in missing_lines
                self.tracker.record_attempt(filename, b["lineno"], covered)

        feasibility_report = self.tracker.get_report()

        recommendations = self._recommend(coverage_report, typology, feasibility_report)

        full = {
            "coverage": coverage_report,
            "typology": typology,
            "feasibility": feasibility_report,
            "recommendations": recommendations,
        }

        summary_path = self.root / "coverage_intel_summary.json"
        summary_path.write_text(json.dumps(full, indent=2))
        return full

    def _recommend(
        self,
        coverage_report: Dict[str, Any],
        typology: Dict[str, List[Dict[str, Any]]],
        feasibility_report: Dict[str, Any],
    ) -> List[str]:
        recs: List[str] = []

        by_file_counts: Dict[str, Dict[str, int]] = {}
        for filename, branches in typology.items():
            counts: Dict[str, int] = defaultdict(int)
            missing_lines = set(coverage_report[filename]["missing_lines"])
            for b in branches:
                if b["lineno"] in missing_lines:
                    counts[b["kind"]] += 1
            by_file_counts[filename] = counts

        for filename, counts in by_file_counts.items():
            numeric = counts.get("DATA_NUMERIC", 0)
            env = counts.get("ENVIRONMENTAL", 0)
            defensive = counts.get("DEFENSIVE", 0)

            if numeric > 0:
                recs.append(
                    f"{filename}: {numeric} uncovered numeric branches → prioritize Hypothesis/property tests."
                )
            if env > 0:
                recs.append(
                    f"{filename}: {env} uncovered environmental branches → add stateful mocks or integration tests."
                )
            if defensive > 10:
                recs.append(
                    f"{filename}: {defensive} defensive branches → consider refactoring or marking infeasible."
                )

        suspected = feasibility_report.get("suspected_infeasible", [])
        for file, lineno in suspected:
            recs.append(
                f"{file}:{lineno} reached max_attempts → treat as infeasible or adjust SUT."
            )

        return recs

    # devtools/coverage_intelligence.py - ADD THIS METHOD AFTER _recommend()

    def print_report(self, report: Dict[str, Any]) -> None:
        """Pretty-print the coverage intelligence report to console"""
        print("\n" + "=" * 80)
        print("COVERAGE INTELLIGENCE REPORT")
        print("=" * 80 + "\n")

        # Overall stats
        feasibility = report["feasibility"]
        print(f"Total branches analyzed: {feasibility['total_branches_attempted']}")
        print(f"Suspected infeasible: {feasibility['suspected_infeasible_count']} "
              f"({feasibility['suspected_infeasible_pct']:.1f}%)\n")

        # Per-file breakdown
        print("Per-File Coverage:")
        print("-" * 80)

        for filename, cov_data in report["coverage"].items():
            short_name = filename.split("/")[-1]  # Just filename, not full path
            print(f"\n{short_name}:")
            print(f"  Line Coverage:   {cov_data['line_pct']:.1f}%")
            print(f"  Branch Coverage: {cov_data['branch_pct']:.1f}%")

            # Typology breakdown
            type_counts = defaultdict(int)
            for branch in report["typology"].get(filename, []):
                if branch["lineno"] in cov_data["missing_lines"]:
                    type_counts[branch["kind"]] += 1

            if type_counts:
                print("  Uncovered by type:")
                for kind, count in sorted(type_counts.items()):
                    print(f"    - {kind}: {count}")

        # Recommendations
        if report["recommendations"]:
            print("\n" + "=" * 80)
            print("ACTIONABLE RECOMMENDATIONS")
            print("=" * 80)
            for i, rec in enumerate(report["recommendations"], 1):
                print(f"{i}. {rec}")

        # Suspected infeasibles
        suspected = feasibility.get("suspected_infeasible", [])
        if suspected:
            print("\n" + "=" * 80)
            print("SUSPECTED INFEASIBLE BRANCHES (hit max attempts)")
            print("=" * 80)
            for file, lineno in suspected[:10]:  # Show first 10
                short_file = file.split("/")[-1]
                print(f"  {short_file}:{lineno}")
            if len(suspected) > 10:
                print(f"  ... and {len(suspected) - 10} more")

        print("\n" + "=" * 80 + "\n")



if __name__ == "__main__":
    root = Path(".").resolve()
    intel = CoverageIntelligence(root, cache_file=root / ".feasibility_cache.json")
    cov_file = root / ".coverage"
    targets = [
        Path("tests/python/test_adaptive_strategies/migrated_strategies/harness.py"),
    ]
    report = intel.analyze(cov_file, targets)
    print("Wrote coverage_intel_summary.json")



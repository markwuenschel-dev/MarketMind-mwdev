from __future__ import annotations

import json
from pathlib import Path

from pysrc.meta.reptile_trainer_benchmark import run_bounded_benchmark


def test_committed_benchmark_artifact_schema() -> None:
    path = Path("artifacts/phase_ii/mlc3/reptile_trainer_benchmark.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == "mlc3.benchmark.v1"
    assert "mean_inner_loop_gain_query_ic" in data
    assert data["determinism_inner_adapt_replay_max_abs_delta"] == 0.0
    assert data["crisis_floor_met"] is True
    live = run_bounded_benchmark(seed=int(data["seed"]))
    assert live["schema_version"] == data["schema_version"]
    assert live["seed"] == data["seed"]
    assert live["determinism_inner_adapt_replay_max_abs_delta"] == 0.0


def test_committed_d0_replay_artifact() -> None:
    path = Path("artifacts/phase_ii/mlc3/reptile_trainer_d0_replay_report.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == "mlc3.d0_replay.v1"
    assert data["overall_max_abs_delta"] == 0.0
    assert len(data["per_task_max_abs_delta"]) == int(data["n_tasks"])

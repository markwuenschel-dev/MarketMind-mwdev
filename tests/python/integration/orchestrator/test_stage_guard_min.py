import time

import pytest

from pysrc.pipeline import orchestrator as m


def _base_cfg():
    return {
        "pipeline": {
            "cleaning": {"combos": {"default": {"steps": [], "order": {}}}, "use": "default"}
        }
    }


def test_stage_guard_success():
    orch = m.DataPrepOrchestrator(_base_cfg())
    out = orch._stage_with_guard("ok", lambda: "success", timeout_s=None)
    assert out == "success"
    assert any(r.get("name") == "ok" for r in orch._metrics.get("stages", []))


def test_stage_guard_retry_then_success():
    orch = m.DataPrepOrchestrator(
        {
            "error_handling": {
                "retry_policy": {
                    "max_attempts": 2,
                    "initial_backoff_seconds": 0,
                    "max_backoff_seconds": 0,
                }
            },
            **_base_cfg(),
        }
    )
    calls = {"n": 0}

    def flappy():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("first")
        return "ok"

    out = orch._stage_with_guard("retry", flappy, timeout_s=None)
    assert out == "ok"


def test_stage_guard_retries_exhausted():
    orch = m.DataPrepOrchestrator(
        {
            "error_handling": {
                "retry_policy": {
                    "max_attempts": 2,
                    "initial_backoff_seconds": 0,
                    "max_backoff_seconds": 0,
                }
            },
            **_base_cfg(),
        }
    )

    def always_fail():
        raise ValueError("boom")

    with pytest.raises(m.DataPrepError):
        orch._stage_with_guard("fail", always_fail, timeout_s=None)
    assert any(
        r.get("name") == "fail" and r.get("status") == "failed"
        for r in orch._metrics.get("stages", [])
    )


def test_stage_guard_timeout_path():
    orch = m.DataPrepOrchestrator(_base_cfg())
    with pytest.raises(Exception):
        orch._stage_with_guard("slow", lambda: time.sleep(0.25), timeout_s=0.05)


@pytest.mark.parametrize(("val", "expected_min"), [("auto", 1), ("3", 3), (4, 3)])
def test_resolve_workers_param(val, expected_min):
    if not hasattr(m.DataPrepOrchestrator, "_resolve_workers"):
        pytest.skip("no _resolve_workers")
    orch = m.DataPrepOrchestrator(_base_cfg())
    got = orch._resolve_workers(val)
    assert int(got) >= expected_min

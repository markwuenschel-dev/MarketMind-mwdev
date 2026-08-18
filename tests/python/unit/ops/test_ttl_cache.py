from __future__ import annotations

import pytest

from pysrc.ops.caching import ttl_cache


@pytest.mark.determinism("d1")
def test_ttl_cache_accepts_unhashable_config_args(deterministic_seed: int) -> None:
    _ = deterministic_seed
    calls = 0

    @ttl_cache(ttl=60)
    def build(configs: list[dict[str, object]]) -> int:
        nonlocal calls
        calls += 1
        return calls

    configs = [{"name": "join", "on": ["date", "symbol"]}]

    assert build(configs) == 1
    assert build(configs) == 1
    assert calls == 1

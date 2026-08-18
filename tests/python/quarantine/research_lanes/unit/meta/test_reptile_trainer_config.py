from __future__ import annotations

import pytest

from pysrc.core.errors import DataPreconditionError
from pysrc.meta.reptile_trainer_config import ReptileTrainerConfig


def test_defaults_k5_reptile() -> None:
    c = ReptileTrainerConfig()
    assert c.K == 5
    assert c.algorithm == "reptile"
    assert c.crisis_floor_pct == 0.10


def test_frozen_dataclass() -> None:
    c = ReptileTrainerConfig()
    with pytest.raises(AttributeError):
        c.K = 3  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"K": 0}, "K must be"),
        ({"outer_lr": 0.0}, "outer_lr"),
        ({"outer_lr": float("nan")}, "outer_lr"),
        ({"crisis_floor_pct": 0.0}, "crisis_floor_pct"),
        ({"crisis_floor_pct": 1.5}, "crisis_floor_pct"),
        ({"task_failure_abort_threshold": 0}, "task_failure_abort_threshold"),
    ],
)
def test_validation_errors(kwargs: dict, match: str) -> None:
    with pytest.raises(DataPreconditionError, match=match):
        ReptileTrainerConfig(**kwargs)


def test_algorithm_literal_accepted() -> None:
    ReptileTrainerConfig(algorithm="anil")
    ReptileTrainerConfig(algorithm="fomaml")


def test_task_pool_bucket_minimums_must_cover_all_buckets() -> None:
    with pytest.raises(DataPreconditionError, match="task_pool_bucket_minimums"):
        ReptileTrainerConfig(task_pool_bucket_minimums={"bull": 50})


def test_task_pool_bucket_minimums_negative_rejected() -> None:
    from pysrc.meta_learning.regime_vocabulary import REGIME_CLASS_ORDER

    bad = dict.fromkeys(REGIME_CLASS_ORDER, 1)
    bad["crisis"] = -1
    with pytest.raises(DataPreconditionError, match="non-negative"):
        ReptileTrainerConfig(task_pool_bucket_minimums=bad)

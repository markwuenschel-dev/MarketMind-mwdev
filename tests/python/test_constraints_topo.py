import pytest

from pysrc.pipeline.core.pipeline_core_builder import topo_order
from tests.python.infra.matrix import matrix


def test_conflict_detection():
    steps = ["A", "B", "C"]
    order = {"before": {"B": ["A"], "A": ["B"]}}  # impossible - creates cycle A->B->A
    with pytest.raises(ValueError, match="Cyclic step constraints detected"):
        topo_order(steps, order)


@matrix(policy=["fail_fast", "last_wins"])
@pytest.mark.abstract
def test_constraints_topo(policy):
    steps = ["A", "B", "C"]
    order = {"before": {"A": ["B"]}}
    topo = topo_order(steps, order)
    plan = {"steps": steps, "order": order, "policy": policy, "topo": topo}
    assert plan["topo"]
    assert set(plan["topo"]) == set(steps)

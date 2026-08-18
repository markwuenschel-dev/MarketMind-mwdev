import numpy as np
import pytest

from pysrc.preprocessor.graph.backends.polars import get
from tests.python.infra.matrix import matrix


@matrix(
    win=[5, 20],
    ddof=[0, 1],
    backend=["polars", "pandas"],
    ids={"backend": {"polars": "pl", "pandas": "pd"}},
)
@pytest.mark.dynamic
def test_zscore_invariants_matrix(win, ddof, backend):
    try:
        fn = get(backend, "scaling.zscore")
    except Exception:
        pytest.skip("zscore op not registered for backend")
    s = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)
    out = fn(s, window=win, ddof=ddof)
    if np.nanstd(s) > 0:
        m, std = np.nanmean(out), np.nanstd(out)
        assert abs(m) < 1e-6
        assert 0.99 < std < 1.01

"""Panel sequence window materialization tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pysrc.pipeline.panel.sequence_data import build_sequence_windows


@pytest.mark.determinism("d1")
def test_build_sequence_windows_point_in_time_order(deterministic_seed: int) -> None:
    del deterministic_seed
    frame = pd.DataFrame(
        {
            "date": [
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",
            ],
            "instrument": ["A", "A", "A", "B", "B", "B"],
            "feat": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            "target": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )
    xs, ys, meta = build_sequence_windows(
        frame,
        ["feat"],
        "target",
        sequence_length=2,
    )
    assert xs.shape == (2, 2, 1)
    assert ys.tolist() == [0.3, 0.6]
    assert meta["date"].tolist() == ["2020-01-03", "2020-01-03"]
    assert np.isfinite(xs).all()

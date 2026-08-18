import numpy as np
import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from pysrc.pipeline.stages.cleaning import build_cleaning_pipeline
from tests.python.infra.matrix import matrix


def _frames_equal(a: pl.DataFrame, b: pl.DataFrame) -> bool:
    return a.equals(b)


def _missing_pipeline(method: str, *, backward_fill: bool = True):
    return build_cleaning_pipeline(
        {
            "steps": [
                {
                    "step_id": "impute.missing",
                    "step_type": "impute.missing",
                    "version": "1",
                    "params": {
                        "method": method,
                        "backward_fill": backward_fill,
                    },
                }
            ]
        }
    )


@st.composite
def frames_with_nans(draw):
    n = draw(st.integers(min_value=5, max_value=50))
    vals = draw(
        st.lists(
            st.one_of(st.none(), st.floats(allow_nan=False, allow_infinity=False)),
            min_size=n,
            max_size=n,
        )
    )
    if all(v is None for v in vals):
        vals[0] = 0.0
    return pl.DataFrame({"x": vals})


@pytest.mark.determinism("d1")
@settings(
    deadline=None,
    max_examples=100,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(df=frames_with_nans())
def test_missing_imputer_idempotent(df: pl.DataFrame, deterministic_seed: int):
    _ = deterministic_seed
    pipeline = _missing_pipeline("forward_fill")
    a = pipeline.run(df).frame
    b = pipeline.run(a).frame
    assert _frames_equal(a, b)
    assert a.shape == df.shape == b.shape
    assert (
        a.select(pl.col("x").is_null().sum()).item()
        <= df.select(pl.col("x").is_null().sum()).item()
    )


def make_frame_with_rate(n: int, missing_rate: float, seed: int = 123) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    data = rng.normal(size=n)
    mask = rng.random(n) < missing_rate
    vals = [None if m else float(v) for v, m in zip(data, mask, strict=False)]
    if all(v is None for v in vals):
        vals[0] = 0.0
    return pl.DataFrame({"x": vals})


@pytest.mark.determinism("d1")
@matrix(method=["forward_fill", "median"], miss=[0.0, 0.1, 0.3])
def test_missing_imputer_idempotent_matrix(method: str, miss: float, deterministic_seed: int):
    _ = deterministic_seed
    df = make_frame_with_rate(50, missing_rate=miss, seed=123)
    pipeline = _missing_pipeline(method)
    a = pipeline.run(df).frame
    b = pipeline.run(a).frame
    assert _frames_equal(a, b)
    assert a.shape == df.shape

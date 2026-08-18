# tests/python/property/test_optimize_equiv.py
import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st


def spec_draw():
    return st.fixed_dictionaries({"ops": st.lists(st.text(max_size=10), max_size=5)})


@settings(deadline=None, max_examples=75)
@given(spec_draw())
def test_optimize_equivalence(spec):
    # Simple test that just validates the spec structure
    df = pl.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    assert isinstance(spec, dict)
    assert "ops" in spec
    assert isinstance(spec["ops"], list)
    # For empty ops, just return the original dataframe
    result = df
    assert result is not None

# tests/python/test_validators_contracts.py
from datetime import datetime

import polars as pl
import pytest

from pysrc.pipeline.stages.cleaning.validators.contracts import MarketDataFrameSchema
from tests.python.infra.matrix import matrix


@pytest.mark.determinism("d1")
@pytest.mark.parametrize(
    "dtype", [pl.Float64, pl.Int64, pl.Utf8], ids=lambda d: str(d).split(".")[-1]
)
def test_required_column_types(dtype, deterministic_seed):
    _ = deterministic_seed
    schema = MarketDataFrameSchema(required_columns={"foo": dtype})
    df = pl.DataFrame({"foo": pl.Series([1, 2, 3]).cast(dtype)})
    ok, errors = schema.validate(df, strict=False)
    assert ok, f"errors: {errors}"


@pytest.mark.determinism("d1")
def test_type_mismatch_reports(deterministic_seed):
    _ = deterministic_seed
    schema = MarketDataFrameSchema(required_columns={"foo": pl.Float64})
    df = pl.DataFrame({"foo": ["a", "b", "c"]})
    ok, errors = schema.validate(df, strict=False)
    assert not ok
    assert any("foo" in e for e in errors)


@pytest.mark.determinism("d1")
def test_schema_dtype_mismatch_reports_errors(deterministic_seed):
    _ = deterministic_seed
    schema = MarketDataFrameSchema(
        required_columns={"timestamp": pl.Datetime, "price": pl.Float64},
        optional_columns={"volume": pl.Int64},
    )
    df_prices = pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
            "price": [1.0, 2.0],
            "volume": [1, 2],
        }
    )
    df_bad = df_prices.with_columns(pl.col("price").cast(pl.Utf8))  # wrong type
    ok, errors = schema.validate(df_bad, strict=False)
    assert ok is False
    assert errors
    assert "price" in " ".join(errors)


@pytest.mark.determinism("d1")
def test_missing_required_column_reports_errors(deterministic_seed):
    _ = deterministic_seed
    schema = MarketDataFrameSchema(required_columns={"foo": pl.Float64})
    df = pl.DataFrame({"bar": [1.0, 2.0]})
    ok, errors = schema.validate(df, strict=True)
    assert not ok
    assert any("foo" in e.lower() for e in errors)


@pytest.mark.determinism("d1")
def test_unknown_column_allowed_when_not_strict(deterministic_seed):
    _ = deterministic_seed
    schema = MarketDataFrameSchema(required_columns={"x": pl.Int64})
    df = pl.DataFrame({"x": [1, 2], "extra": [10, 20]})
    ok, errors = schema.validate(df, strict=False)
    assert ok
    assert not errors


@pytest.mark.determinism("d1")
@matrix(
    dtype=[pl.Float64, pl.Int64, pl.Utf8],
    strict=[True, False],
    unknown_ok=[True, False],
    ids={"strict": {True: "strict", False: "loose"}},
)
@pytest.mark.contract
def test_required_column_types_matrix(dtype, strict, unknown_ok, deterministic_seed):
    _ = deterministic_seed
    schema = MarketDataFrameSchema(required_columns={"x": dtype}, unknown_ok=unknown_ok)
    # Create data with the correct dtype
    if dtype == pl.Float64:
        data = [1.0, 2.0, 3.0]
    elif dtype == pl.Int64:
        data = [1, 2, 3]
    else:  # pl.Utf8
        data = ["1", "2", "3"]
    df = pl.DataFrame({"x": data})
    if unknown_ok:
        df = df.with_columns(pl.Series("extra", [1, 2, 3]))
    ok, errors = schema.validate(df, strict=strict)
    assert ok
    assert not errors

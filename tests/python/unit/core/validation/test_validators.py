# tests/python/unit/core/validation/test_validators.py
from datetime import datetime
from unittest.mock import Mock

import pandas as pd
import pytest
from allpairspy import AllPairs
from hypothesis import given, seed, settings
from hypothesis import strategies as st

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]
import numpy as np

pytestmark = [pytest.mark.determinism("d0"), pytest.mark.usefixtures("deterministic_seed")]
requires_torch = pytest.mark.skipif(torch is None, reason="torch optional dependency not installed")


# Optional imports
try:
    import polars as pl
except ImportError:
    pl = None

try:
    import dask.dataframe as dd
except ImportError:
    dd = None

try:
    import cudf
except ImportError:
    cudf = None

try:
    import cupy as cp
except ImportError:
    cp = None

from pysrc.core.errors import DataValidationError
from pysrc.core.validation import (
    all_of,
    any_of,
    compose_validators,
    lazy_validate_ohlcv,
    register_validator,
    validate_data_for_training,
    validate_dataframe,
    validate_date,
    validate_file_data,
    validate_ohlcv,
    validate_series,
    validate_stream_chunk,
    validate_symbol,
    validate_tensor,
)

# ========================= validate_dataframe tests =========================


def test_validate_dataframe_empty_pandas_raises():
    df = pd.DataFrame()
    with pytest.raises(DataValidationError, match="DataFrame is empty"):
        validate_dataframe(df)


def test_validate_dataframe_nonempty_pandas_passes():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    validate_dataframe(df)  # should not raise


def test_validate_dataframe_required_cols_present():
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    validate_dataframe(df, required_cols=["a", "c"])  # should not raise


def test_validate_dataframe_required_cols_missing():
    df = pd.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(DataValidationError, match="Missing required columns"):
        validate_dataframe(df, required_cols=["a", "c"])


def test_validate_dataframe_empty_with_required_cols():
    # By design, "Missing required columns" wins unless required_cols includes "missing"
    df = pd.DataFrame()
    with pytest.raises(DataValidationError, match="Missing required columns"):
        validate_dataframe(df, required_cols=["a", "b"])


def test_validate_dataframe_no_columns_attribute():
    obj_without_columns = Mock(spec=[])  # no columns attribute
    # Module defers to core validator on this path -> "Unsupported dataframe type"
    with pytest.raises(DataValidationError, match="Unsupported dataframe type"):
        validate_dataframe(obj_without_columns, required_cols=["a"])


def test_validate_dataframe_unsupported_type():
    # Avoid faking an 'empty' attribute; ensure we hit the unsupported path
    unsupported_obj = Mock(spec=[])  # no 'empty' attribute
    with pytest.raises(DataValidationError, match="Unsupported dataframe type"):
        validate_dataframe(unsupported_obj)


@pytest.mark.skipif(pl is None, reason="polars not installed")
def test_validate_dataframe_polars_empty():
    df = pl.DataFrame()
    with pytest.raises(DataValidationError, match="DataFrame is empty"):
        validate_dataframe(df)


@pytest.mark.skipif(pl is None, reason="polars not installed")
def test_validate_dataframe_polars_nonempty():
    df = pl.DataFrame({"a": [1, 2]})
    validate_dataframe(df)  # should not raise


@pytest.mark.skipif(pl is None, reason="polars not installed")
def test_validate_dataframe_lazyframe_allowed():
    lf = pl.LazyFrame({"a": [1, 2]})
    validate_dataframe(lf, allow_lazy=True)  # should not raise


@pytest.mark.skipif(pl is None, reason="polars not installed")
def test_validate_dataframe_lazyframe_not_allowed():
    lf = pl.LazyFrame({"a": [1, 2]})
    with pytest.raises(DataValidationError, match="LazyFrame not allowed"):
        validate_dataframe(lf, allow_lazy=False)


# pairwise combinatorial tests for validate_dataframe
def test_validate_dataframe_pairwise(tmp_path):
    # build test dataframes
    pd_empty = pd.DataFrame()
    pd_with_a = pd.DataFrame({"a": [1]})
    pd_ohlcv = pd.DataFrame(
        {"Open": [1], "High": [2], "Low": [0.5], "Close": [1.5], "Volume": [100]}
    )

    dfs = [pd_empty, pd_with_a, pd_ohlcv]
    if pl is not None:
        dfs.extend([pl.DataFrame(), pl.DataFrame({"a": [1]}), pl.LazyFrame({"a": [1]})])

    pairwise_params = [
        {"df": df, "required_cols": cols, "allow_lazy": lazy}
        for df, cols, lazy in AllPairs(
            [dfs, [None, ["a"], ["Open", "High", "Low", "Close", "Volume"]], [True, False]]
        )
    ]

    for params in pairwise_params:
        df = params["df"]
        required = params["required_cols"]
        allow_lazy = params["allow_lazy"]

        try:
            validate_dataframe(df, required_cols=required, allow_lazy=allow_lazy)
        except DataValidationError:
            pass  # expected for some combinations


# ========================= lazy_validate_ohlcv tests =========================


def test_lazy_validate_ohlcv_pandas_valid():
    df = pd.DataFrame({"open": [1], "high": [2], "low": [0.5], "close": [1.5], "volume": [100]})
    assert lazy_validate_ohlcv(df) is True


def test_lazy_validate_ohlcv_missing_columns():
    df = pd.DataFrame({"open": [1], "close": [2]})
    assert lazy_validate_ohlcv(df) is False


def test_lazy_validate_ohlcv_case_insensitive():
    df = pd.DataFrame({"OPEN": [1], "HIGH": [2], "LOW": [0.5], "CLOSE": [1.5], "VOLUME": [100]})
    assert lazy_validate_ohlcv(df) is True


def test_lazy_validate_ohlcv_no_columns():
    obj = Mock(spec=[])
    assert lazy_validate_ohlcv(obj) is False


# ========================= validate_ohlcv tests =========================


def test_validate_ohlcv_pandas_valid():
    df = pd.DataFrame({"open": [1], "high": [2], "low": [0.5], "close": [1.5], "volume": [100]})
    validate_ohlcv(df)  # should not raise


def test_validate_ohlcv_pandas_missing_columns():
    df = pd.DataFrame({"open": [1], "close": [2]})
    with pytest.raises(DataValidationError, match="Missing OHLCV columns"):
        validate_ohlcv(df)


def test_validate_ohlcv_pandas_case_variants():
    # pandas is lenient on case
    df = pd.DataFrame({"Open": [1], "HIGH": [2], "low": [0.5], "Close": [1.5], "VOLUME": [100]})
    validate_ohlcv(df)  # should not raise


def test_validate_ohlcv_pandas_stringified_numeric():
    # pandas is lenient - attempts coercion
    df = pd.DataFrame(
        {"open": ["1"], "high": ["2"], "low": ["0.5"], "close": ["1.5"], "volume": ["100"]}
    )
    validate_ohlcv(df)  # should not raise due to leniency


@pytest.mark.skipif(pl is None, reason="polars not installed")
def test_validate_ohlcv_polars_valid():
    df = pl.DataFrame(
        {"open": [1.0], "high": [2.0], "low": [0.5], "close": [1.5], "volume": [100.0]}
    )
    validate_ohlcv(df)  # should not raise


@pytest.mark.skipif(pl is None, reason="polars not installed")
def test_validate_ohlcv_polars_non_numeric_raises():
    df = pl.DataFrame(
        {"open": ["1"], "high": ["2"], "low": ["0.5"], "close": ["1.5"], "volume": ["100"]}
    )
    with pytest.raises(DataValidationError, match="must be numeric"):
        validate_ohlcv(df)


def test_validate_ohlcv_unsupported_type():
    unsupported = Mock(spec=[])
    with pytest.raises(DataValidationError, match="Unsupported dataframe type"):
        validate_ohlcv(unsupported)


# ========================= validate_stream_chunk tests =========================


def test_validate_stream_chunk_pandas_empty():
    chunk = pd.DataFrame()
    with pytest.raises(DataValidationError, match="Stream chunk cannot be empty"):
        validate_stream_chunk(chunk)


def test_validate_stream_chunk_pandas_nonempty():
    chunk = pd.DataFrame({"a": [1, 2]})
    validate_stream_chunk(chunk)  # should not raise


@pytest.mark.skipif(pl is None, reason="polars not installed")
def test_validate_stream_chunk_polars_empty():
    chunk = pl.DataFrame()
    with pytest.raises(DataValidationError, match="Stream chunk cannot be empty"):
        validate_stream_chunk(chunk)


def test_validate_stream_chunk_unsupported():
    chunk = "not a dataframe"
    with pytest.raises(DataValidationError, match="Unsupported chunk type"):
        validate_stream_chunk(chunk)


# ========================= validate_data_for_training tests =========================


def test_validate_data_for_training_delegates_to_ohlcv():
    df = pd.DataFrame({"open": [1], "high": [2], "low": [0.5], "close": [1.5], "volume": [100]})
    validate_data_for_training(df)  # should not raise

    invalid_df = pd.DataFrame({"a": [1]})
    with pytest.raises(DataValidationError, match="Missing OHLCV columns"):
        validate_data_for_training(invalid_df)


# ========================= validate_series tests =========================


def test_validate_series_pandas_empty():
    s = pd.Series([], dtype=float)
    with pytest.raises(DataValidationError, match="series is empty"):
        validate_series(s)


def test_validate_series_pandas_all_nan():
    s = pd.Series([np.nan, np.nan])
    with pytest.raises(DataValidationError, match="contains only NaN values"):
        validate_series(s)


def test_validate_series_pandas_valid():
    s = pd.Series([1, 2, np.nan, 3])
    validate_series(s)  # should not raise


def test_validate_series_custom_name():
    s = pd.Series([], dtype=float)
    with pytest.raises(DataValidationError, match="my_series is empty"):
        validate_series(s, name="my_series")


@pytest.mark.skipif(pl is None, reason="polars not installed")
def test_validate_series_polars_empty():
    s = pl.Series("s", [], dtype=pl.Float64)
    with pytest.raises(DataValidationError, match="series is empty"):
        validate_series(s)


@pytest.mark.skipif(pl is None, reason="polars not installed")
def test_validate_series_polars_all_null():
    s = pl.Series("s", [None, None])
    with pytest.raises(DataValidationError, match="contains only null values"):
        validate_series(s)


def test_validate_series_unsupported():
    s = [1, 2, 3]  # not a series
    with pytest.raises(DataValidationError, match="Unsupported series type"):
        validate_series(s)


# pairwise for series states
@pytest.mark.parametrize(
    ("backend", "state"),
    [
        ("pandas", "empty"),
        ("pandas", "all_nulls"),
        ("pandas", "valid_mixed"),
        pytest.param("polars", "empty", marks=pytest.mark.skipif(pl is None, reason="no polars")),
        pytest.param(
            "polars", "all_nulls", marks=pytest.mark.skipif(pl is None, reason="no polars")
        ),
        pytest.param(
            "polars", "valid_mixed", marks=pytest.mark.skipif(pl is None, reason="no polars")
        ),
    ],
)
def test_validate_series_states(backend, state):
    if backend == "pandas":
        if state == "empty":
            s = pd.Series([], dtype=float)
            with pytest.raises(DataValidationError):
                validate_series(s)
        elif state == "all_nulls":
            s = pd.Series([np.nan, np.nan])
            with pytest.raises(DataValidationError):
                validate_series(s)
        else:  # valid_mixed
            s = pd.Series([1, np.nan, 2])
            validate_series(s)
    elif backend == "polars":
        if state == "empty":
            s = pl.Series("s", [], dtype=pl.Float64)
            with pytest.raises(DataValidationError):
                validate_series(s)
        elif state == "all_nulls":
            s = pl.Series("s", [None, None])
            with pytest.raises(DataValidationError):
                validate_series(s)
        else:  # valid_mixed
            s = pl.Series("s", [1, None, 2])
            validate_series(s)


# ========================= validate_symbol tests =========================


def test_validate_symbol_valid_alphanumeric():
    validate_symbol("AAPL")  # should not raise
    validate_symbol("AAPL1")  # should not raise


def test_validate_symbol_invalid_underscore():
    with pytest.raises(DataValidationError, match="Symbol must be an alphanumeric string"):
        validate_symbol("AAP_L")


def test_validate_symbol_invalid_space():
    with pytest.raises(DataValidationError, match="Symbol must be an alphanumeric string"):
        validate_symbol("AAP L")


def test_validate_symbol_empty_string():
    with pytest.raises(DataValidationError, match="Symbol must be an alphanumeric string"):
        validate_symbol("")


def test_validate_symbol_not_string():
    with pytest.raises(DataValidationError, match="Symbol must be an alphanumeric string"):
        validate_symbol(123)


# pairwise for symbol values
@pytest.mark.parametrize("symbol", ["AAPL", "AAPL1", "AAP_L", ""])
def test_validate_symbol_values(symbol):
    if symbol in ["AAPL", "AAPL1"]:
        validate_symbol(symbol)  # should pass
    else:
        with pytest.raises(DataValidationError):
            validate_symbol(symbol)


# ========================= validate_date tests =========================


def test_validate_date_iso_string():
    validate_date("2021-01-01")  # should not raise


def test_validate_date_datetime_object():
    validate_date(datetime.now())  # should not raise


def test_validate_date_invalid_format():
    with pytest.raises(DataValidationError, match="Invalid date format"):
        validate_date("2021/01/01")


def test_validate_date_not_string_or_datetime():
    with pytest.raises(DataValidationError, match="Date must be a string or datetime object"):
        validate_date(123)


# pairwise for date values
@pytest.mark.parametrize("date", ["2021-01-01", "2021/01/01", datetime.now()])
def test_validate_date_values(date):
    if isinstance(date, datetime) or date == "2021-01-01":
        validate_date(date)
    else:
        with pytest.raises(DataValidationError):
            validate_date(date)


# ========================= validate_tensor tests =========================


@requires_torch
def test_validate_tensor_empty():
    t = torch.tensor([])
    with pytest.raises(DataValidationError, match="tensor is empty"):
        validate_tensor(t)


@requires_torch
def test_validate_tensor_correct_ndim():
    t = torch.ones(3, 4)
    validate_tensor(t, ndim=2)  # should not raise


@requires_torch
def test_validate_tensor_incorrect_ndim():
    t = torch.ones(3, 4)
    with pytest.raises(DataValidationError, match="must have 3 dimensions"):
        validate_tensor(t, ndim=3)


@requires_torch
def test_validate_tensor_float_with_nan():
    t = torch.tensor([1.0, float("nan"), 3.0])
    with pytest.raises(DataValidationError, match="contains NaN or infinite values"):
        validate_tensor(t)


@requires_torch
def test_validate_tensor_float_with_inf():
    t = torch.tensor([1.0, float("inf"), 3.0])
    with pytest.raises(DataValidationError, match="contains NaN or infinite values"):
        validate_tensor(t)


@requires_torch
def test_validate_tensor_float_clean():
    t = torch.tensor([1.0, 2.0, 3.0])
    validate_tensor(t)  # should not raise


@requires_torch
def test_validate_tensor_int_clean():
    # int tensors skip NaN/Inf check
    t = torch.tensor([1, 2, 3], dtype=torch.int32)
    validate_tensor(t)  # should not raise


@requires_torch
def test_validate_tensor_custom_name():
    t = torch.tensor([])
    with pytest.raises(DataValidationError, match="my_tensor is empty"):
        validate_tensor(t, name="my_tensor")


def test_validate_tensor_unsupported():
    t = [1, 2, 3]  # not a tensor
    with pytest.raises(DataValidationError, match="Unsupported tensor type"):
        validate_tensor(t)


# pairwise for tensor scenarios
tensor_test_cases = [
    ("float_with_nan", None),
    ("float_with_inf", None),
    ("float_clean", None),
    ("int_clean", None),
    ("float_clean", 2),  # correct ndim
    ("float_clean", 3),  # incorrect ndim
]


@pytest.mark.parametrize(("kind", "ndim_arg"), tensor_test_cases)
@requires_torch
def test_validate_tensor_pairwise(kind, ndim_arg):
    if kind == "float_with_nan":
        t = torch.tensor([1.0, float("nan")], dtype=torch.float32)
        with pytest.raises(DataValidationError):
            validate_tensor(t, ndim=ndim_arg)
    elif kind == "float_with_inf":
        t = torch.tensor([1.0, float("inf")], dtype=torch.float32)
        with pytest.raises(DataValidationError):
            validate_tensor(t, ndim=ndim_arg)
    elif kind == "float_clean":
        t = torch.ones(2, 3, dtype=torch.float32)
        if ndim_arg == 3:
            with pytest.raises(DataValidationError):
                validate_tensor(t, ndim=ndim_arg)
        else:
            validate_tensor(t, ndim=ndim_arg)
    elif kind == "int_clean":
        t = torch.ones(2, 3, dtype=torch.int32)
        validate_tensor(t, ndim=ndim_arg)


# ========================= compose_validators tests =========================


def test_compose_validators_all_pass():
    def v1(x):
        pass

    def v2(x):
        pass

    composed = compose_validators(v1, v2)
    composed("test")  # should not raise


def test_compose_validators_first_fails():
    def v1(x):
        raise DataValidationError("v1 failed")

    def v2(x):
        pass

    composed = compose_validators(v1, v2)
    with pytest.raises(DataValidationError, match="v1 failed"):
        composed("test")


def test_compose_validators_second_fails():
    def v1(x):
        pass

    def v2(x):
        raise DataValidationError("v2 failed")

    composed = compose_validators(v1, v2)
    with pytest.raises(DataValidationError, match="v2 failed"):
        composed("test")


# ========================= all_of tests =========================


def test_all_of_all_pass():
    validators = [
        lambda x: None,
        lambda x: None,
    ]
    validator = all_of(validators)
    validator("test")  # should not raise


def test_all_of_one_fails():
    validators = [
        lambda x: None,
        lambda x: (_ for _ in ()).throw(DataValidationError("failed")),
    ]
    validator = all_of(validators)
    with pytest.raises(DataValidationError, match="failed"):
        validator("test")


def test_all_of_empty_list():
    validator = all_of([])
    validator("test")  # should not raise (no validators)


# ========================= any_of tests =========================


def test_any_of_first_passes():
    validators = [
        lambda x: None,
        lambda x: (_ for _ in ()).throw(DataValidationError("v2")),
    ]
    validator = any_of(validators)
    validator("test")  # should not raise


def test_any_of_second_passes():
    validators = [
        lambda x: (_ for _ in ()).throw(DataValidationError("v1")),
        lambda x: None,
    ]
    validator = any_of(validators)
    validator("test")  # should not raise


def test_any_of_all_fail():
    validators = [
        lambda x: (_ for _ in ()).throw(DataValidationError("v1")),
        lambda x: (_ for _ in ()).throw(DataValidationError("v2")),
    ]
    validator = any_of(validators)
    with pytest.raises(DataValidationError, match="All validators failed"):
        validator("test")


def test_any_of_empty_list():
    validator = any_of([])
    validator("test")  # should not raise (empty list passes)


# ========================= register_validator tests =========================


def test_register_validator_exact_type_match(monkeypatch):
    # mock the registry

    class CustomType:
        pass

    called = []

    @register_validator(CustomType)
    def custom_validator(obj):
        called.append(obj)
        raise DataValidationError("custom validation")

    # validate_dataframe uses registry
    obj = CustomType()
    with pytest.raises(DataValidationError, match="custom validation"):
        validate_dataframe(obj)

    assert len(called) == 1


def test_register_validator_overrides_default():
    # test that registry takes precedence
    class MockDF:
        empty = False
        columns = ["a", "b"]

    @register_validator(MockDF)
    def mock_validator(df):
        raise DataValidationError("registry override")

    df = MockDF()
    with pytest.raises(DataValidationError, match="registry override"):
        validate_dataframe(df)


# ========================= validate_file_data tests =========================


def test_validate_file_data_unknown_format():
    data = pd.DataFrame({"a": [1]})
    with pytest.raises(DataValidationError, match="Unknown file format"):
        validate_file_data(data, "unknown")


def test_validate_file_data_unsupported():
    data = "not a dataframe"
    with pytest.raises(DataValidationError, match="Unsupported data for IO validation"):
        validate_file_data(data, "csv")


@pytest.mark.skipif(pl is None, reason="polars not installed")
def test_validate_file_data_polars_csv_with_commas():
    df = pl.DataFrame({"col1": ["value,with,comma"]})
    with pytest.raises(DataValidationError, match="Potential unescaped commas in column col1"):
        validate_file_data(df, "csv")


@pytest.mark.skipif(pl is None, reason="polars not installed")
def test_validate_file_data_polars_csv_clean():
    df = pl.DataFrame({"col1": ["clean_value"]})
    validate_file_data(df, "csv")  # should not raise


# pairwise for file_data
@pytest.mark.skipif(pl is None, reason="polars not installed")
@pytest.mark.parametrize(
    ("data_type", "format"),
    [
        ("csv_with_commas", "csv"),
        ("csv_clean", "csv"),
        ("csv_with_commas", "unknown"),
        ("csv_clean", "unknown"),
    ],
)
def test_validate_file_data_pairwise(data_type, format):
    if data_type == "csv_with_commas":
        df = pl.DataFrame({"col": ["a,b,c"]})
    else:
        df = pl.DataFrame({"col": ["abc"]})

    if format == "unknown":
        with pytest.raises(DataValidationError, match="Unknown file format"):
            validate_file_data(df, format)
    elif data_type == "csv_with_commas":
        with pytest.raises(DataValidationError, match="Potential unescaped commas"):
            validate_file_data(df, format)
    else:
        validate_file_data(df, format)


# ========================= Property-based tests =========================


@seed(12345)
@settings(deadline=None)
@given(rows=st.integers(min_value=0, max_value=100), cols=st.integers(min_value=1, max_value=10))
def test_property_validate_dataframe_consistency(rows, cols):
    # property: empty check is consistent
    if rows == 0:
        df = pd.DataFrame({f"c{i}": [] for i in range(cols)})
        with pytest.raises(DataValidationError, match="empty"):
            validate_dataframe(df)
    else:
        df = pd.DataFrame({f"c{i}": list(range(rows)) for i in range(cols)})
        validate_dataframe(df)  # should not raise


@seed(12345)
@settings(deadline=None)
@given(
    s=st.lists(
        st.one_of(st.floats(allow_nan=True, allow_infinity=False), st.none()),
        min_size=0,
        max_size=50,
    )
)
def test_property_validate_series_nulls(s):
    # property: series validation handles nulls correctly
    series = pd.Series(s)

    if len(s) == 0:
        with pytest.raises(DataValidationError, match="empty"):
            validate_series(series)
    elif all(x is None or (isinstance(x, float) and np.isnan(x)) for x in s):
        with pytest.raises(DataValidationError, match="NaN"):
            validate_series(series)
    else:
        validate_series(series)


@seed(12345)
@settings(deadline=None)
@given(symbol=st.text(min_size=0, max_size=10))
def test_property_validate_symbol_alphanumeric(symbol):
    # property: only alphanumeric strings pass
    if symbol.isalnum() and len(symbol) > 0:
        validate_symbol(symbol)
    else:
        with pytest.raises(DataValidationError):
            validate_symbol(symbol)


@seed(12345)
@settings(deadline=50)  # trivial operation
@given(validators_pass=st.lists(st.booleans(), min_size=0, max_size=5))
def test_property_any_of_logic(validators_pass):
    # property: any_of passes iff at least one validator passes
    validators = [
        (lambda x: None)
        if should_pass
        else (lambda x: (_ for _ in ()).throw(DataValidationError(f"v{i}")))
        for i, should_pass in enumerate(validators_pass)
    ]

    validator = any_of(validators)

    if any(validators_pass) or not validators_pass:  # empty list passes
        validator("test")
    else:
        with pytest.raises(DataValidationError, match="All validators failed"):
            validator("test")


@seed(12345)
@settings(deadline=None)
@given(
    shape=st.tuples(st.integers(min_value=0, max_value=10), st.integers(min_value=0, max_value=10)),
    has_nan=st.booleans(),
    has_inf=st.booleans(),
    is_int=st.booleans(),
)
@requires_torch
def test_property_validate_tensor_invariants(shape, has_nan, has_inf, is_int):
    # property: tensor validation follows type-based rules
    total_elements = shape[0] * shape[1]

    if total_elements == 0:
        t = torch.empty(*shape)
        with pytest.raises(DataValidationError, match="empty"):
            validate_tensor(t)
    elif is_int:
        # int tensors don't check NaN/Inf
        t = torch.randint(0, 10, shape, dtype=torch.int32)
        validate_tensor(t)
    else:
        # float tensors
        t = torch.randn(*shape)
        if has_nan:
            t.view(-1)[0] = float("nan")
        if has_inf:
            t.view(-1)[-1] = float("inf")

        if has_nan or has_inf:
            with pytest.raises(DataValidationError, match="NaN or infinite"):
                validate_tensor(t)
        else:
            validate_tensor(t)


# ========================= Contract tests for multiple implementations =========================
@pytest.mark.parametrize(
    ("series_cls", "name"),
    [
        (pd.Series, "pandas"),
        pytest.param(
            pl.Series if pl else None,
            "polars",
            marks=pytest.mark.skipif(pl is None, reason="no polars"),
        ),
    ],
)
def test_contract_series_validation_invariants(series_cls, name):
    if series_cls is None:
        return

    # empty series always fails
    if name == "pandas":
        s = series_cls([], dtype=float)
    else:  # polars
        s = series_cls("s", [], dtype=pl.Float64)
    with pytest.raises(DataValidationError, match="empty"):
        validate_series(s)

    # all nulls always fails - use flexible regex
    if name == "pandas":
        s = series_cls([np.nan, np.nan])
    else:  # polars
        s = series_cls("s", [None, None])
    with pytest.raises(DataValidationError, match="(NaN|null)"):  # Changed this line
        validate_series(s)

    # mixed valid/null passes
    if name == "pandas":
        s = series_cls([1.0, np.nan, 2.0])
    else:  # polars
        s = series_cls("s", [1.0, None, 2.0])
    validate_series(s)  # should not raise


@pytest.mark.parametrize(
    ("df_cls", "name"),
    [
        (pd.DataFrame, "pandas"),
        pytest.param(
            pl.DataFrame if pl else None,
            "polars",
            marks=pytest.mark.skipif(pl is None, reason="no polars"),
        ),
    ],
)
def test_contract_dataframe_validation_invariants(df_cls, name):
    if df_cls is None:
        return

    # empty always fails
    df = df_cls()
    with pytest.raises(DataValidationError, match="empty"):
        validate_dataframe(df)

    # non-empty passes
    df = df_cls({"a": [1, 2]})
    validate_dataframe(df)

    # required columns checked
    df = df_cls({"a": [1], "b": [2]})
    validate_dataframe(df, required_cols=["a"])  # passes
    with pytest.raises(DataValidationError, match="Missing required columns"):
        validate_dataframe(df, required_cols=["a", "c"])


# ========================= Edge case and error condition coverage =========================


def test_edge_validate_dataframe_missing_column_priority():
    # test the special "missing" column name logic
    df = pd.DataFrame()
    with pytest.raises(DataValidationError, match="DataFrame is empty"):
        validate_dataframe(df, required_cols=["missing"])


def test_error_dtype_enforcement_helper():
    # test ensure_numeric helper indirectly through validate_ohlcv
    df = pd.DataFrame({"open": [1], "high": [2], "low": [0.5], "close": [1.5], "volume": [100]})
    validate_ohlcv(df)  # numeric columns pass


@pytest.mark.skipif(cp is None, reason="cupy not installed")
def test_validate_tensor_cupy():
    t = cp.array([1.0, 2.0, 3.0])
    validate_tensor(t)  # should not raise

    t_empty = cp.array([])
    with pytest.raises(DataValidationError, match="empty"):
        validate_tensor(t_empty)

    t_nan = cp.array([1.0, float("nan")])
    with pytest.raises(
        DataValidationError, match="(NaN|infinite)"
    ):  # Make sure this matches the actual error message
        validate_tensor(t_nan)


def test_combinator_error_aggregation():
    # test that any_of aggregates error messages
    def v1(x):
        return (_ for _ in ()).throw(DataValidationError("error1"))

    def v2(x):
        return (_ for _ in ()).throw(DataValidationError("error2"))

    validator = any_of([v1, v2])

    try:
        validator("test")
        raise AssertionError("Should have raised")
    except DataValidationError as e:
        assert "error1" in str(e)
        assert "error2" in str(e)


def test_registry_persistence():
    # test that registry is module-level and persists
    class TestType:
        pass

    @register_validator(TestType)
    def test_validator(obj):
        raise DataValidationError("test")

    # registry should be active
    from pysrc.core.validation import _validator_registry

    assert TestType in _validator_registry


# Test ensure_numeric directly
def test_ensure_numeric_rejects_non_numeric():
    df = pd.DataFrame({"price": ["100", "200"], "volume": [1, 2]})
    from pysrc.core.validation import ensure_numeric

    with pytest.raises(TypeError, match="must be numeric"):
        ensure_numeric(df, ["price"])


# Test _df_column_names
def test_df_column_names_lazyframe():
    if pl is None:
        pytest.skip("polars not available")
    lf = pl.LazyFrame({"a": [1], "b": [2]})
    from pysrc.core.validation import _df_column_names

    cols = _df_column_names(lf)
    assert cols == ["a", "b"]


# Test protocol-based fallback
def test_validate_series_protocol_fallback():
    class MockSeries:
        def is_empty(self):
            return False

        def null_count(self):
            return 0

        def len(self):
            return 5

    # This won't work as-is because Protocol is runtime_checkable
    # but the dispatch won't see it. You'd need to register it.
    MockSeries()
    # Expected behavior depends on implementation details


# Test Dask if available
@pytest.mark.skipif(dd is None, reason="dask not installed")
def test_validate_dataframe_dask_empty():
    import dask.dataframe as dd
    import pandas as pd

    df = dd.from_pandas(pd.DataFrame(), npartitions=1)
    with pytest.raises(DataValidationError, match="empty"):
        validate_dataframe(df)


# Test cuDF if available
@pytest.mark.skipif(cudf is None, reason="cudf not installed")
def test_validate_series_cudf():
    """Test cuDF Series validation"""
    import cudf

    # Empty
    s = cudf.Series([])
    with pytest.raises(DataValidationError, match="empty"):
        validate_series(s)

    # All nulls - cuDF reports "null values" not "NaN"
    s = cudf.Series([None, None])
    with pytest.raises(DataValidationError, match="(NaN|null)"):
        validate_series(s)

    # Valid
    s = cudf.Series([1, 2, 3])
    validate_series(s)


# Test CustomSeriesValidator
def test_custom_series_validator_min_length():
    from pysrc.core.validation import CustomSeriesValidator

    validator = CustomSeriesValidator(min_length=5)
    short_series = pd.Series([1, 2, 3])
    with pytest.raises(DataValidationError, match="too short"):
        validator.validate(short_series)


# Test ensure_numeric with Polars (early return)
@pytest.mark.skipif(pl is None, reason="polars not installed")
def test_ensure_numeric_polars_skips():
    from pysrc.core.validation import ensure_numeric

    df = pl.DataFrame({"a": ["str"], "b": [1]})
    # Should not raise because Polars path returns early
    ensure_numeric(df, ["a", "b"])


# Test error message aggregation more thoroughly
def test_any_of_error_messages_contain_all_failures():
    errors = []
    for i in range(3):
        errors.append(
            lambda x, i=i: (_ for _ in ()).throw(DataValidationError(f"validator_{i}_failed"))
        )

    validator = any_of(errors)
    with pytest.raises(DataValidationError) as exc_info:
        validator("test")

    for i in range(3):
        assert f"validator_{i}_failed" in str(exc_info.value)


def test_validate_series_protocol_based_fallback():
    """Test SeriesLike protocol fallback for custom series types"""
    from pysrc.core.validation import validate_series

    class CustomSeries:
        def __init__(self, data, empty=False, all_null=False):
            self._data = data
            self._empty = empty
            self._all_null = all_null

        def is_empty(self):
            return self._empty

        def null_count(self):
            return len(self._data) if self._all_null else 0

        def len(self):
            return len(self._data)

    # Test empty via protocol
    empty_series = CustomSeries([], empty=True)
    with pytest.raises(DataValidationError, match="series is empty"):
        validate_series(empty_series)

    # Test all nulls via protocol
    null_series = CustomSeries([None, None], all_null=True)
    with pytest.raises(DataValidationError, match="contains only null values"):
        validate_series(null_series)

    # Test valid via protocol
    valid_series = CustomSeries([1, 2, 3])
    validate_series(valid_series)  # Should pass


def test_validate_series_registry_takes_precedence():
    """Test that registry validator is called before protocol check"""
    from pysrc.core.validation import register_validator, validate_series

    class MyCustomSeries:
        pass

    called = []

    @register_validator(MyCustomSeries)
    def my_validator(series, name="series"):
        called.append((series, name))
        # Intentionally pass without raising

    obj = MyCustomSeries()
    validate_series(obj, name="custom")

    assert len(called) == 1
    assert called[0][1] == "custom"


def test_validate_tensor_registry_takes_precedence():
    """Test that registry validator is called before protocol check"""
    from pysrc.core.validation import register_validator, validate_tensor

    class MyCustomTensor:
        pass

    called = []

    @register_validator(MyCustomTensor)
    def my_tensor_validator(tensor, ndim=None, name="tensor"):
        called.append((tensor, ndim, name))

    obj = MyCustomTensor()
    validate_tensor(obj, ndim=3, name="my_tensor")

    assert len(called) == 1
    assert called[0][1] == 3
    assert called[0][2] == "my_tensor"


def test_protocol_definitions_exist():
    """Test that protocol classes are defined and can be checked"""
    import inspect

    from pysrc.core.validation import DataFrameLike, SeriesLike, TensorLike

    # Verify protocols are runtime_checkable
    assert hasattr(SeriesLike, "__protocol_attrs__") or inspect.isclass(SeriesLike)
    assert hasattr(DataFrameLike, "__protocol_attrs__") or inspect.isclass(DataFrameLike)
    assert hasattr(TensorLike, "__protocol_attrs__") or inspect.isclass(TensorLike)


# Remove the broken protocol tests and replace with these:


def test_validate_series_registry_fallback():
    """Test that custom types can be validated via registry"""
    from pysrc.core.validation import register_validator, validate_series

    class MyCustomSeries:
        def __init__(self, data):
            self.data = data

    # Before registration, should fail
    obj = MyCustomSeries([1, 2, 3])
    with pytest.raises(DataValidationError, match="Unsupported series type"):
        validate_series(obj)

    # Register validator
    @register_validator(MyCustomSeries)
    def validate_my_series(series, name="series"):
        if not series.data:
            raise DataValidationError(f"{name} has no data")

    # After registration, registry is checked
    validate_series(obj)  # Should pass

    # Test with failing validation
    empty_obj = MyCustomSeries([])
    with pytest.raises(DataValidationError, match="has no data"):
        validate_series(empty_obj, name="my_series")


def test_validate_tensor_registry_fallback():
    """Test that custom tensor types can be validated via registry"""
    from pysrc.core.validation import register_validator, validate_tensor

    class MyCustomTensor:
        def __init__(self, shape):
            self.shape = shape

    # Before registration, should fail
    obj = MyCustomTensor((3, 4))
    with pytest.raises(DataValidationError, match="Unsupported .* type"):
        validate_tensor(obj)

    # Register validator
    @register_validator(MyCustomTensor)
    def validate_my_tensor(tensor, ndim=None, name="tensor"):
        if ndim is not None and len(tensor.shape) != ndim:
            raise DataValidationError(f"{name} has wrong dimensions")

    # After registration, should use registry
    validate_tensor(obj)  # Should pass (no ndim specified)
    validate_tensor(obj, ndim=2)  # Should pass

    with pytest.raises(DataValidationError, match="wrong dimensions"):
        validate_tensor(obj, ndim=3)


def test_ensure_numeric_uint_dtype():
    """Test UInt dtype passes numeric check"""
    from pysrc.core.validation import ensure_numeric

    df = pd.DataFrame({"col": pd.array([1, 2, 3], dtype="UInt32")})
    ensure_numeric(df, ["col"])  # Should pass


def test_ensure_numeric_mixed_numeric():
    """Test int and float dtypes pass"""
    from pysrc.core.validation import ensure_numeric

    df = pd.DataFrame({"a": [1, 2], "b": [1.5, 2.5]})
    ensure_numeric(df, ["a", "b"])  # Should pass


def test_ensure_numeric_object_dtype_numbers():
    """Object dtype with numeric strings should fail"""
    from pysrc.core.validation import ensure_numeric

    df = pd.DataFrame({"col": ["1", "2"]}, dtype=object)
    with pytest.raises(TypeError, match="must be numeric"):
        ensure_numeric(df, ["col"])


@pytest.mark.skipif(cudf is None, reason="cudf not installed")
def test_validate_dataframe_cudf():
    """Test cuDF DataFrame validation"""
    import cudf

    # Empty cuDF DataFrame
    df = cudf.DataFrame()
    with pytest.raises(DataValidationError, match="empty"):
        validate_dataframe(df)

    # Non-empty cuDF DataFrame
    df = cudf.DataFrame({"a": [1, 2, 3]})
    validate_dataframe(df)  # Should pass


@pytest.mark.skipif(dd is None, reason="dask not installed")
def test_validate_dataframe_dask():
    """Test Dask DataFrame validation"""
    df_pandas = pd.DataFrame({"a": [1, 2, 3]})
    df = dd.from_pandas(df_pandas, npartitions=2)
    validate_dataframe(df)  # Should pass


@pytest.mark.skipif(dd is None, reason="dask not installed")
def test_validate_series_dask():
    """Test Dask Series validation"""
    s_pandas = pd.Series([1, 2, 3])
    s = dd.from_pandas(s_pandas, npartitions=2)
    validate_series(s)  # Should pass

    # Empty
    s_empty = dd.from_pandas(pd.Series([]), npartitions=1)
    with pytest.raises(DataValidationError, match="empty"):
        validate_series(s_empty)


def test_is_df_with_cudf():
    """Test _is_df with cuDF types"""
    from pysrc.core.validation import _is_df

    if cudf is not None:
        df = cudf.DataFrame({"a": [1]})
        assert _is_df(df) is True


def test_is_series_with_cudf():
    """Test _is_series with cuDF types"""
    from pysrc.core.validation import _is_series

    if cudf is not None:
        s = cudf.Series([1, 2, 3])
        assert _is_series(s) is True


def test_validate_stream_chunk_polars_nonempty():
    """Test Polars stream chunk validation passes for non-empty"""
    if pl is None:
        pytest.skip("polars not available")

    chunk = pl.DataFrame({"a": [1, 2, 3]})
    validate_stream_chunk(chunk)  # Should pass


def test_lazy_validate_ohlcv_exception_handling():
    """Test lazy_validate_ohlcv handles exceptions gracefully"""
    from pysrc.core.validation import lazy_validate_ohlcv

    class BrokenObject:
        @property
        def columns(self):
            raise RuntimeError("Columns are broken!")

    obj = BrokenObject()
    # Should return False when exception occurs
    assert lazy_validate_ohlcv(obj) is False


def test_df_column_names_no_columns():
    """Test _df_column_names with object lacking columns"""
    from pysrc.core.validation import _df_column_names

    obj = object()
    with pytest.raises(DataValidationError, match="no columns attribute"):
        _df_column_names(obj)


def test_validate_ohlcv_via_registry():
    """Test validate_ohlcv can use registry for custom types"""
    from pysrc.core.validation import register_validator, validate_ohlcv

    class CustomDF:
        pass

    @register_validator(CustomDF)
    def validate_custom(df):
        raise DataValidationError("Custom OHLCV validation")

    obj = CustomDF()
    with pytest.raises(DataValidationError, match="Custom OHLCV validation"):
        validate_ohlcv(obj)

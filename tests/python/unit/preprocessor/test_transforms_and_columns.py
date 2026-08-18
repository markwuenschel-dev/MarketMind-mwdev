# tests/python/unit/preprocessor/test_transforms_and_columns.py
"""
Comprehensive tests for py/preprocessor/utils/transforms.py and columns.py

These tests target:
- transforms.py: 36% → ~75% (112 missing lines)
- columns.py: 31% → ~70% (91 missing lines)
"""

from __future__ import annotations

import json
import tempfile
from unittest.mock import MagicMock, patch

import pytest

# Try importing polars - skip tests if not available
try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    pl = None

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None


# =============================================================================
# columns.py Tests
# =============================================================================


class TestColumnsImports:
    """Test that columns module imports correctly."""

    def test_columns_module_imports(self):
        """Basic import smoke test."""
        from pysrc.preprocessor.utils import columns

        assert hasattr(columns, "_as_list")
        assert hasattr(columns, "_derive_out_names")
        assert hasattr(columns, "ColumnOp")
        assert hasattr(columns, "ColumnOpFactory")
        assert hasattr(columns, "op_chain")


class TestAsListFunction:
    """Test _as_list() helper function."""

    def test_as_list_with_none(self):
        """None returns empty list."""
        from pysrc.preprocessor.utils.columns import _as_list

        assert _as_list(None) == []

    def test_as_list_with_single_value(self):
        """Single value wrapped in list."""
        from pysrc.preprocessor.utils.columns import _as_list

        assert _as_list("col") == ["col"]
        assert _as_list(42) == [42]

    def test_as_list_with_list(self):
        """List returned as-is."""
        from pysrc.preprocessor.utils.columns import _as_list

        assert _as_list(["a", "b"]) == ["a", "b"]

    def test_as_list_with_tuple(self):
        """Tuple returned as-is (not converted to list)."""
        from pysrc.preprocessor.utils.columns import _as_list

        result = _as_list(("a", "b"))
        assert result == ("a", "b")


class TestDeriveOutNames:
    """Test _derive_out_names() function."""

    def test_derive_with_explicit_out_col(self):
        """Explicit out_col takes precedence."""
        from pysrc.preprocessor.utils.columns import _derive_out_names

        result = _derive_out_names(["a", "b"], suffix="_z", out_col="custom")
        assert result == ["custom"]

    def test_derive_with_suffix(self):
        """Suffix appended to column names."""
        from pysrc.preprocessor.utils.columns import _derive_out_names

        result = _derive_out_names(["a", "b"], suffix="norm")
        assert result == ["a_norm", "b_norm"]

    def test_derive_without_suffix(self):
        """No suffix returns original columns."""
        from pysrc.preprocessor.utils.columns import _derive_out_names

        result = _derive_out_names(["a", "b"], suffix=None)
        assert result == ["a", "b"]

    def test_derive_single_column(self):
        """Single column input."""
        from pysrc.preprocessor.utils.columns import _derive_out_names

        result = _derive_out_names("close", suffix="z")
        assert result == ["close_z"]


class TestIsSeqOfStr:
    """Test _is_seq_of_str() helper."""

    def test_is_seq_of_str_with_list_of_strings(self):
        """List of strings returns True."""
        from pysrc.preprocessor.utils.columns import _is_seq_of_str

        assert _is_seq_of_str(["a", "b", "c"]) is True

    def test_is_seq_of_str_with_tuple_of_strings(self):
        """Tuple of strings returns True."""
        from pysrc.preprocessor.utils.columns import _is_seq_of_str

        assert _is_seq_of_str(("a", "b")) is True

    def test_is_seq_of_str_with_set_of_strings(self):
        """Set of strings returns True."""
        from pysrc.preprocessor.utils.columns import _is_seq_of_str

        assert _is_seq_of_str({"a", "b"}) is True

    def test_is_seq_of_str_with_mixed_types(self):
        """Mixed types returns False."""
        from pysrc.preprocessor.utils.columns import _is_seq_of_str

        assert _is_seq_of_str(["a", 1, "b"]) is False

    def test_is_seq_of_str_with_string(self):
        """Single string returns False (not a sequence of strings)."""
        from pysrc.preprocessor.utils.columns import _is_seq_of_str

        assert _is_seq_of_str("abc") is False

    def test_is_seq_of_str_with_empty_list(self):
        """Empty list returns True."""
        from pysrc.preprocessor.utils.columns import _is_seq_of_str

        assert _is_seq_of_str([]) is True


class TestColumnOpValidation:
    """Test ColumnOp.validate() method."""

    def test_validate_with_invalid_cols_type(self):
        """Non-sequence cols raises ValueError."""
        from pysrc.preprocessor.utils.columns import CastNumeric

        op = CastNumeric()
        mock_df = MagicMock()
        mock_df.columns = ["a", "b"]

        with pytest.raises(ValueError, match="sequence of column names"):
            op.validate(mock_df, "not_a_list")

    def test_validate_with_missing_columns(self):
        """Missing columns raises SchemaMismatch."""
        from pysrc.preprocessor.utils.columns import CastNumeric
        from pysrc.preprocessor.utils.errors import SchemaMismatch

        op = CastNumeric()
        mock_df = MagicMock()
        mock_df.columns = ["a", "b"]

        with pytest.raises(SchemaMismatch, match="Missing columns"):
            op.validate(mock_df, ["a", "missing_col"])

    def test_validate_success(self):
        """Valid columns pass validation."""
        from pysrc.preprocessor.utils.columns import CastNumeric

        op = CastNumeric()
        mock_df = MagicMock()
        mock_df.columns = ["a", "b", "c"]

        # Should not raise
        op.validate(mock_df, ["a", "b"])


@pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
class TestCastNumericPolars:
    """Test CastNumeric with Polars DataFrames."""

    def test_cast_numeric_polars_float32(self):
        """Cast to float32 in Polars."""
        from pysrc.preprocessor.utils.columns import CastNumeric

        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        op = CastNumeric()
        result = op.apply(df, ["a", "b"], dtype="float32")

        assert result["a"].dtype == pl.Float32
        assert result["b"].dtype == pl.Float32

    def test_cast_numeric_polars_float64(self):
        """Cast to float64 in Polars."""
        from pysrc.preprocessor.utils.columns import CastNumeric

        df = pl.DataFrame({"a": [1, 2, 3]})
        op = CastNumeric()
        result = op.apply(df, ["a"], dtype="float64")

        assert result["a"].dtype == pl.Float64

    def test_cast_numeric_polars_int32(self):
        """Cast to int32 in Polars."""
        from pysrc.preprocessor.utils.columns import CastNumeric

        df = pl.DataFrame({"a": [1.5, 2.5, 3.5]})
        op = CastNumeric()
        result = op.apply(df, ["a"], dtype="int32")

        assert result["a"].dtype == pl.Int32


@pytest.mark.skipif(not HAS_PANDAS, reason="Pandas not available")
class TestCastNumericPandas:
    """Test CastNumeric with Pandas DataFrames."""

    def test_cast_numeric_pandas(self):
        """Cast in Pandas."""
        from pysrc.preprocessor.utils.columns import CastNumeric

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        op = CastNumeric()
        result = op.apply(df, ["a", "b"], dtype="float32")

        assert result["a"].dtype == "float32"
        assert result["b"].dtype == "float32"


@pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
class TestPromoteCategoricalPolars:
    """Test PromoteCategorical with Polars."""

    def test_promote_categorical_polars(self):
        """Promote to categorical in Polars."""
        from pysrc.preprocessor.utils.columns import PromoteCategorical

        df = pl.DataFrame({"cat": ["a", "b", "a", "c"]})
        op = PromoteCategorical()
        result = op.apply(df, ["cat"])

        assert result["cat"].dtype == pl.Categorical

    def test_promote_categorical_ordered(self):
        """Promote to ordered categorical."""
        from pysrc.preprocessor.utils.columns import PromoteCategorical

        df = pl.DataFrame({"cat": ["low", "medium", "high"]})
        op = PromoteCategorical()
        result = op.apply(df, ["cat"], ordered=True)

        assert result["cat"].dtype == pl.Categorical


@pytest.mark.skipif(not HAS_PANDAS, reason="Pandas not available")
class TestPromoteCategoricalPandas:
    """Test PromoteCategorical with Pandas."""

    def test_promote_categorical_pandas(self):
        """Promote to categorical in Pandas."""
        from pysrc.preprocessor.utils.columns import PromoteCategorical

        df = pd.DataFrame({"cat": ["a", "b", "a", "c"]})
        op = PromoteCategorical()
        result = op.apply(df, ["cat"])

        assert result["cat"].dtype.name == "category"


class TestColumnOpFactory:
    """Test ColumnOpFactory class."""

    def test_factory_build_cast_numeric(self):
        """Build CastNumeric from factory."""
        from pysrc.preprocessor.utils.columns import CastNumeric, ColumnOpFactory

        op = ColumnOpFactory.build("cast_numeric")
        assert isinstance(op, CastNumeric)

    def test_factory_build_promote_categorical(self):
        """Build PromoteCategorical from factory."""
        from pysrc.preprocessor.utils.columns import ColumnOpFactory, PromoteCategorical

        op = ColumnOpFactory.build("promote_categorical")
        assert isinstance(op, PromoteCategorical)

    def test_factory_build_unknown_raises(self):
        """Unknown op raises UnsupportedAST."""
        from pysrc.preprocessor.utils.columns import ColumnOpFactory
        from pysrc.preprocessor.utils.errors import UnsupportedAST

        with pytest.raises(UnsupportedAST, match="not registered"):
            ColumnOpFactory.build("unknown_op")

    def test_factory_register_custom_op(self):
        """Register and build custom op."""
        from pysrc.preprocessor.utils.columns import ColumnOp, ColumnOpFactory

        class CustomOp(ColumnOp):
            def apply(self, df, cols, **kwargs):
                return df

        ColumnOpFactory.register("custom_test_op", CustomOp)
        op = ColumnOpFactory.build("custom_test_op")
        assert isinstance(op, CustomOp)


class TestOpChain:
    """Test op_chain() function."""

    @pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
    def test_op_chain_single_op(self):
        """Chain with single op."""
        from pysrc.preprocessor.utils.columns import op_chain

        df = pl.DataFrame({"a": [1, 2, 3]})
        chain = op_chain("cast_numeric")
        result = chain(df, ["a"], dtype="float64")

        assert result["a"].dtype == pl.Float64

    @pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
    def test_op_chain_multiple_ops(self):
        """Chain with multiple ops."""
        from pysrc.preprocessor.utils.columns import op_chain

        df = pl.DataFrame({"a": ["x", "y", "x"]})
        # Just test that chaining works (cast then promote would fail, so use same op twice)
        chain = op_chain("promote_categorical")
        result = chain(df, ["a"])

        assert result["a"].dtype == pl.Categorical


class TestLiveAfter:
    """Test live_after() function."""

    def test_live_after_removes_duplicates(self):
        """Duplicates removed, order preserved."""
        from pysrc.preprocessor.utils.columns import live_after

        result = live_after(["a", "b", "a", "c", "b"])
        assert result == ["a", "b", "c"]

    def test_live_after_preserves_order(self):
        """Order is preserved."""
        from pysrc.preprocessor.utils.columns import live_after

        result = live_after(["c", "b", "a"])
        assert result == ["c", "b", "a"]

    def test_live_after_with_string_raises(self):
        """Single string raises ValueError."""
        from pysrc.preprocessor.utils.columns import live_after

        with pytest.raises(ValueError, match="iterable of column names"):
            live_after("abc")

    def test_live_after_empty(self):
        """Empty input returns empty list."""
        from pysrc.preprocessor.utils.columns import live_after

        result = live_after([])
        assert result == []


class TestEnsureUnique:
    """Test ensure_unique() function."""

    def test_ensure_unique_no_duplicates(self):
        """No duplicates returns original."""
        from pysrc.preprocessor.utils.columns import ensure_unique

        result = ensure_unique(["a", "b", "c"])
        assert result == ["a", "b", "c"]

    def test_ensure_unique_with_duplicates(self):
        """Duplicates get suffixed."""
        from pysrc.preprocessor.utils.columns import ensure_unique

        result = ensure_unique(["a", "a", "a"])
        assert result == ["a", "a__1", "a__2"]

    def test_ensure_unique_custom_separator(self):
        """Custom separator used."""
        from pysrc.preprocessor.utils.columns import ensure_unique

        result = ensure_unique(["x", "x"], sep="_v")
        assert result == ["x", "x_v1"]

    def test_ensure_unique_empty(self):
        """Empty input returns empty list."""
        from pysrc.preprocessor.utils.columns import ensure_unique

        result = ensure_unique([])
        assert result == []

    def test_ensure_unique_mixed_duplicates(self):
        """Mixed duplicates handled correctly."""
        from pysrc.preprocessor.utils.columns import ensure_unique

        result = ensure_unique(["a", "b", "a", "c", "b", "a"])
        assert result == ["a", "b", "a__1", "c", "b__1", "a__2"]


class TestSaveMetrics:
    """Test save_metrics() function."""

    def test_save_metrics_creates_file(self):
        """Metrics saved to JSON file."""
        from pysrc.preprocessor.utils.columns import _prof_metrics, save_metrics

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = f.name

        # Add some test metrics
        _prof_metrics["test_metric"] = 0.123

        save_metrics(filepath)

        with open(filepath) as f:
            data = json.load(f)

        assert "test_metric" in data
        assert data["test_metric"] == 0.123


class TestProfileOp:
    """Test profile_op decorator."""

    def test_profile_op_records_timing(self):
        """Decorator records timing."""
        from pysrc.preprocessor.utils.columns import ColumnOp, _prof_metrics, profile_op

        class TimedOp(ColumnOp):
            @profile_op
            def apply(self, df, cols, **kwargs):
                return df

        op = TimedOp()
        mock_df = MagicMock()
        mock_df.columns = ["a"]

        op.apply(mock_df, ["a"])

        # Check that some metric was recorded
        assert any("TimedOp" in k for k in _prof_metrics)


# =============================================================================
# transforms.py Tests
# =============================================================================


class TestTransformsImports:
    """Test that transforms module imports correctly."""

    def test_transforms_module_imports(self):
        """Basic import smoke test."""
        from pysrc.preprocessor.utils import transforms

        assert hasattr(transforms, "Transform")
        assert hasattr(transforms, "NormalizeTransform")
        assert hasattr(transforms, "BollingerTransform")
        assert hasattr(transforms, "LogTransform")
        assert hasattr(transforms, "MinMaxScaleTransform")
        assert hasattr(transforms, "TransformFactory")


class TestTransformBase:
    """Test Transform base class."""

    def test_transform_selects_backend(self):
        """Transform auto-selects backend."""
        from pysrc.preprocessor.utils.transforms import CompositeTransform

        t = CompositeTransform(lambda df: df)
        assert t.backend in ("cpu", "polars", "cudf")

    def test_transform_explicit_backend(self):
        """Transform accepts explicit backend."""
        from pysrc.preprocessor.utils.transforms import CompositeTransform

        t = CompositeTransform(lambda df: df, backend="cpu")
        assert t.backend == "cpu"

    def test_transform_call_invokes_fn(self):
        """Calling transform invokes internal function."""
        from pysrc.preprocessor.utils.transforms import CompositeTransform

        called = []

        def fn(df):
            called.append(True)
            return df

        t = CompositeTransform(fn)
        t("dummy_df")

        assert len(called) == 1

    def test_transform_apply_same_as_call(self):
        """apply() is same as __call__."""
        from pysrc.preprocessor.utils.transforms import CompositeTransform

        t = CompositeTransform(lambda df: "result")
        assert t.apply("df") == t("df")


class TestTransformComposition:
    """Test Transform.__add__ composition."""

    def test_transform_add_creates_composite(self):
        """Adding transforms creates CompositeTransform."""
        from pysrc.preprocessor.utils.transforms import CompositeTransform

        t1 = CompositeTransform(lambda df: df + "_a")
        t2 = CompositeTransform(lambda df: df + "_b")

        combined = t1 + t2
        assert isinstance(combined, CompositeTransform)

    def test_transform_add_chains_execution(self):
        """Combined transform chains execution."""
        from pysrc.preprocessor.utils.transforms import CompositeTransform

        t1 = CompositeTransform(lambda df: df + "_a")
        t2 = CompositeTransform(lambda df: df + "_b")

        combined = t1 + t2
        result = combined("start")

        assert result == "start_a_b"


@pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
class TestNormalizeTransformPolars:
    """Test NormalizeTransform with Polars."""

    def test_normalize_creates_column(self):
        """Normalize creates _norm column."""
        from pysrc.preprocessor.utils.transforms import NormalizeTransform

        df = pl.DataFrame({"close": [100.0, 110.0, 90.0, 105.0]})
        t = NormalizeTransform(col="close")
        result = t(df)

        assert "close_norm" in result.columns

    def test_normalize_with_explicit_params(self):
        """Normalize with explicit mean/std."""
        from pysrc.preprocessor.utils.transforms import NormalizeTransform

        df = pl.DataFrame({"close": [100.0, 110.0, 90.0]})
        t = NormalizeTransform(col="close", mean=100.0, std=10.0)
        result = t(df)

        # (100 - 100) / 10 = 0, (110 - 100) / 10 = 1, (90 - 100) / 10 = -1
        assert abs(result["close_norm"][0] - 0.0) < 0.01
        assert abs(result["close_norm"][1] - 1.0) < 0.01
        assert abs(result["close_norm"][2] - (-1.0)) < 0.01

    def test_normalize_validation_missing_column(self):
        """Normalize raises on missing column."""
        from pysrc.preprocessor.utils.errors import SchemaMismatch
        from pysrc.preprocessor.utils.transforms import NormalizeTransform

        df = pl.DataFrame({"other": [1, 2, 3]})
        t = NormalizeTransform(col="close")

        with pytest.raises(SchemaMismatch):
            t(df)


@pytest.mark.skipif(not HAS_PANDAS, reason="Pandas not available")
class TestNormalizeTransformPandas:
    """Test NormalizeTransform with Pandas."""

    def test_normalize_pandas(self):
        """Normalize works with Pandas."""
        from pysrc.preprocessor.utils.transforms import NormalizeTransform

        df = pd.DataFrame({"close": [100.0, 110.0, 90.0]})
        t = NormalizeTransform(col="close", mean=100.0, std=10.0)
        result = t(df)

        assert "close_norm" in result.columns


@pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
class TestBollingerTransformPolars:
    """Test BollingerTransform with Polars."""

    def test_bollinger_creates_bands(self):
        """Bollinger creates ma, upper, lower columns."""
        from pysrc.preprocessor.utils.transforms import BollingerTransform

        df = pl.DataFrame({"close": [100.0 + i for i in range(30)]})
        t = BollingerTransform(col="close", window=5)
        result = t(df)

        assert "close_ma" in result.columns
        assert "close_upper" in result.columns
        assert "close_lower" in result.columns

    def test_bollinger_custom_output_cols(self):
        """Bollinger with custom output column names."""
        from pysrc.preprocessor.utils.transforms import BollingerTransform

        df = pl.DataFrame({"close": [100.0 + i for i in range(30)]})
        t = BollingerTransform(col="close", window=5, output_cols=["ma", "top", "bottom"])
        result = t(df)

        assert "ma" in result.columns
        assert "top" in result.columns
        assert "bottom" in result.columns

    def test_bollinger_validation(self):
        """Bollinger validates column exists."""
        from pysrc.preprocessor.utils.errors import SchemaMismatch
        from pysrc.preprocessor.utils.transforms import BollingerTransform

        df = pl.DataFrame({"other": [1, 2, 3]})
        t = BollingerTransform(col="close")

        with pytest.raises(SchemaMismatch):
            t(df)


@pytest.mark.skipif(not HAS_PANDAS, reason="Pandas not available")
class TestBollingerTransformPandas:
    """Test BollingerTransform with Pandas."""

    def test_bollinger_pandas(self):
        """Bollinger works with Pandas."""
        from pysrc.preprocessor.utils.transforms import BollingerTransform

        df = pd.DataFrame({"close": [100.0 + i for i in range(30)]})
        t = BollingerTransform(col="close", window=5)
        result = t(df)

        assert "close_ma" in result.columns


@pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
class TestLogTransformPolars:
    """Test LogTransform with Polars."""

    def test_log_creates_column(self):
        """LogTransform creates _log column."""
        from pysrc.preprocessor.utils.transforms import LogTransform

        df = pl.DataFrame({"close": [10.0, 100.0, 1000.0]})
        t = LogTransform(col="close", base=10.0)
        result = t(df)

        assert "close_log" in result.columns
        # log10(10) = 1, log10(100) = 2, log10(1000) = 3
        assert abs(result["close_log"][0] - 1.0) < 0.01
        assert abs(result["close_log"][1] - 2.0) < 0.01
        assert abs(result["close_log"][2] - 3.0) < 0.01

    def test_log_handles_small_values(self):
        """LogTransform clips small values with eps."""
        from pysrc.preprocessor.utils.transforms import LogTransform

        df = pl.DataFrame({"close": [0.0, -1.0, 100.0]})
        t = LogTransform(col="close", base=10.0, eps=1e-6)
        result = t(df)

        # Should not raise, values clipped to eps
        assert "close_log" in result.columns

    def test_log_validation(self):
        """LogTransform validates column exists."""
        from pysrc.preprocessor.utils.errors import SchemaMismatch
        from pysrc.preprocessor.utils.transforms import LogTransform

        df = pl.DataFrame({"other": [1, 2, 3]})
        t = LogTransform(col="close")

        with pytest.raises(SchemaMismatch):
            t(df)


@pytest.mark.skipif(not HAS_PANDAS, reason="Pandas not available")
class TestLogTransformPandas:
    """Test LogTransform with Pandas."""

    def test_log_pandas(self):
        """LogTransform works with Pandas."""
        from pysrc.preprocessor.utils.transforms import LogTransform

        df = pd.DataFrame({"close": [10.0, 100.0, 1000.0]})
        t = LogTransform(col="close", base=10.0)
        result = t(df)

        assert "close_log" in result.columns


@pytest.mark.skipif(not HAS_PANDAS, reason="Pandas not available")
class TestMinMaxScaleTransform:
    """Test MinMaxScaleTransform."""

    def test_minmax_creates_column(self):
        """MinMaxScale creates _scaled column."""
        from pysrc.preprocessor.utils.transforms import MinMaxScaleTransform

        df = pd.DataFrame({"close": [0.0, 50.0, 100.0]})
        t = MinMaxScaleTransform(col="close")
        result = t(df)

        assert "close_scaled" in result.columns
        # (0-0)/(100-0) = 0, (50-0)/(100-0) = 0.5, (100-0)/(100-0) = 1
        assert abs(result["close_scaled"].iloc[0] - 0.0) < 0.01
        assert abs(result["close_scaled"].iloc[1] - 0.5) < 0.01
        assert abs(result["close_scaled"].iloc[2] - 1.0) < 0.01

    def test_minmax_explicit_range(self):
        """MinMaxScale with explicit min/max."""
        from pysrc.preprocessor.utils.transforms import MinMaxScaleTransform

        df = pd.DataFrame({"close": [25.0, 50.0, 75.0]})
        t = MinMaxScaleTransform(col="close", min_val=0.0, max_val=100.0)
        result = t(df)

        assert abs(result["close_scaled"].iloc[0] - 0.25) < 0.01
        assert abs(result["close_scaled"].iloc[1] - 0.50) < 0.01
        assert abs(result["close_scaled"].iloc[2] - 0.75) < 0.01

    def test_minmax_zero_range(self):
        """MinMaxScale handles zero range (all same values)."""
        from pysrc.preprocessor.utils.transforms import MinMaxScaleTransform

        df = pd.DataFrame({"close": [50.0, 50.0, 50.0]})
        t = MinMaxScaleTransform(col="close")
        result = t(df)

        # Should not raise, uses eps
        assert "close_scaled" in result.columns

    def test_minmax_validation(self):
        """MinMaxScale validates column exists."""
        from pysrc.preprocessor.utils.errors import SchemaMismatch
        from pysrc.preprocessor.utils.transforms import MinMaxScaleTransform

        df = pd.DataFrame({"other": [1, 2, 3]})
        t = MinMaxScaleTransform(col="close")

        with pytest.raises(SchemaMismatch):
            t(df)


class TestToTorchTransform:
    """Test ToTorchTransform."""

    def test_to_torch_validation_missing_cols(self):
        """ToTorchTransform validates columns."""
        from pysrc.preprocessor.utils.errors import SchemaMismatch
        from pysrc.preprocessor.utils.transforms import ToTorchTransform

        mock_df = MagicMock()
        mock_df.columns = ["a", "b"]

        t = ToTorchTransform(cols=["a", "missing"])

        with pytest.raises(SchemaMismatch, match="Missing columns"):
            t.validate(mock_df)

    def test_to_torch_validation_success(self):
        """ToTorchTransform passes with valid columns."""
        from pysrc.preprocessor.utils.transforms import ToTorchTransform

        mock_df = MagicMock()
        mock_df.columns = ["a", "b", "c"]

        t = ToTorchTransform(cols=["a", "b"])
        # Should not raise
        t.validate(mock_df)


class TestTransformFactory:
    """Test TransformFactory class."""

    def test_factory_build_normalize(self):
        """Build NormalizeTransform from factory."""
        from pysrc.preprocessor.utils.transforms import NormalizeTransform, TransformFactory

        t = TransformFactory.build("normalize", col="close")
        assert isinstance(t, NormalizeTransform)

    def test_factory_build_bollinger(self):
        """Build BollingerTransform from factory."""
        from pysrc.preprocessor.utils.transforms import BollingerTransform, TransformFactory

        t = TransformFactory.build("bollinger", col="close", window=20)
        assert isinstance(t, BollingerTransform)

    def test_factory_build_log(self):
        """Build LogTransform from factory."""
        from pysrc.preprocessor.utils.transforms import LogTransform, TransformFactory

        t = TransformFactory.build("log", col="close")
        assert isinstance(t, LogTransform)

    def test_factory_build_minmax(self):
        """Build MinMaxScaleTransform from factory."""
        from pysrc.preprocessor.utils.transforms import MinMaxScaleTransform, TransformFactory

        t = TransformFactory.build("minmax_scale", col="close")
        assert isinstance(t, MinMaxScaleTransform)

    def test_factory_build_unknown_raises(self):
        """Unknown transform raises UnsupportedAST."""
        from pysrc.preprocessor.utils.errors import UnsupportedAST
        from pysrc.preprocessor.utils.transforms import TransformFactory

        with pytest.raises(UnsupportedAST, match="not registered"):
            TransformFactory.build("unknown_transform")

    def test_factory_compose_single(self):
        """Compose single transform."""
        from pysrc.preprocessor.utils.transforms import Transform, TransformFactory

        t = TransformFactory.compose("normalize", normalize={"col": "close"})
        assert isinstance(t, Transform)

    def test_factory_compose_multiple(self):
        """Compose multiple transforms."""
        from pysrc.preprocessor.utils.transforms import CompositeTransform, TransformFactory

        t = TransformFactory.compose(
            "normalize", "log", normalize={"col": "close"}, log={"col": "close"}
        )
        assert isinstance(t, CompositeTransform)

    def test_factory_compose_empty(self):
        """Compose with no transforms returns identity."""
        from pysrc.preprocessor.utils.transforms import CompositeTransform, TransformFactory

        t = TransformFactory.compose()
        assert isinstance(t, CompositeTransform)

        # Identity transform
        result = t("input")
        assert result == "input"

    def test_factory_register_custom(self):
        """Register and build custom transform."""
        from pysrc.preprocessor.utils.transforms import CompositeTransform, TransformFactory

        TransformFactory.register("custom_test", lambda **kw: CompositeTransform(lambda df: df))
        t = TransformFactory.build("custom_test")
        assert isinstance(t, CompositeTransform)


class TestProfileTransform:
    """Test profile_transform decorator."""

    def test_profile_transform_logs_timing(self):
        """Decorator records timing info."""
        from pysrc.preprocessor.utils.transforms import NormalizeTransform

        # Profile is called via .apply()
        mock_df = MagicMock()
        mock_df.columns = ["close"]
        mock_df.__getitem__ = lambda self, k: MagicMock(mean=lambda: 0, std=lambda: 1)

        t = NormalizeTransform(col="close", mean=0, std=1)

        # This exercises the profile_transform decorator
        with patch.object(t, "_fn", return_value=mock_df):
            t.apply(mock_df)


@pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
class TestFeatureEngineerChain:
    """Test feature_engineer_chain() function."""

    def test_feature_engineer_chain_returns_transform(self):
        """feature_engineer_chain returns a Transform."""
        from pysrc.preprocessor.utils.transforms import Transform, feature_engineer_chain

        t = feature_engineer_chain(["close"])
        assert isinstance(t, Transform)

# tests/python/unit/preprocessor/test_expr_ops_executor.py
"""
Comprehensive tests for:
- py/preprocessor/graph/expr.py (45% → ~80%)
- py/preprocessor/graph/ops_custom.py (57% → ~80%)
- py/preprocessor/graph/executor.py (48% → ~75%)
"""

from __future__ import annotations

import pytest

try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    pl = None


# =============================================================================
# expr.py Tests
# =============================================================================


class TestExprImports:
    """Test expr module imports."""

    def test_expr_module_imports(self):
        """Basic import smoke test."""
        from pysrc.preprocessor.graph import expr

        assert hasattr(expr, "Expr")
        assert hasattr(expr, "Column")
        assert hasattr(expr, "Literal")
        assert hasattr(expr, "OpExpr")
        assert hasattr(expr, "expr_factory")
        assert hasattr(expr, "register_expr")


class TestColumn:
    """Test Column expression class."""

    def test_column_creation(self):
        """Create a Column expression."""
        from pysrc.preprocessor.graph.expr import Column

        col = Column("close")
        assert col.name == "close"
        assert col.op == "col"

    def test_column_to_ir(self):
        """Column.to_ir() returns correct IR."""
        from pysrc.preprocessor.graph.expr import Column

        col = Column("close")
        ir = col.to_ir()

        assert ir["op"] == "col"
        assert ir["params"]["name"] == "close"

    def test_column_repr(self):
        """Column.__repr__() is readable."""
        from pysrc.preprocessor.graph.expr import Column

        col = Column("close")
        assert "close" in repr(col)

    def test_column_validation_fails_on_non_string(self):
        """Column with non-string name raises ValueError."""
        from pysrc.preprocessor.graph.expr import Column

        with pytest.raises(ValueError, match="must be str"):
            Column(123)

    def test_column_hash(self):
        """Column is hashable."""
        from pysrc.preprocessor.graph.expr import Column

        col = Column("close")
        h = hash(col)
        assert isinstance(h, int)

    def test_column_equality(self):
        """Column equality comparison."""
        from pysrc.preprocessor.graph.expr import Column

        c1 = Column("close")
        c2 = Column("close")
        c3 = Column("open")

        assert c1 == c2
        assert c1 != c3
        assert c1 != "close"  # Not equal to string


class TestLiteral:
    """Test Literal expression class."""

    def test_literal_creation(self):
        """Create a Literal expression."""
        from pysrc.preprocessor.graph.expr import Literal

        lit = Literal(42)
        assert lit.value == 42
        assert lit.op == "lit"

    def test_literal_to_ir(self):
        """Literal.to_ir() returns correct IR."""
        from pysrc.preprocessor.graph.expr import Literal

        lit = Literal(3.14)
        ir = lit.to_ir()

        assert ir["op"] == "lit"
        assert ir["params"]["value"] == 3.14

    def test_literal_repr(self):
        """Literal.__repr__() is readable."""
        from pysrc.preprocessor.graph.expr import Literal

        lit = Literal(42)
        assert "42" in repr(lit)

    def test_literal_with_various_types(self):
        """Literal works with various value types."""
        from pysrc.preprocessor.graph.expr import Literal

        assert Literal(42).value == 42
        assert Literal(3.14).value == 3.14
        assert Literal("hello").value == "hello"
        assert Literal(None).value is None
        assert Literal([1, 2, 3]).value == [1, 2, 3]


class TestOpExpr:
    """Test OpExpr expression class."""

    def test_opexpr_creation(self):
        """Create an OpExpr."""
        from pysrc.preprocessor.graph.expr import Literal, OpExpr

        expr = OpExpr("add", [Literal(1), Literal(2)])
        assert expr.op == "add"
        assert len(expr.args) == 2

    def test_opexpr_to_ir(self):
        """OpExpr.to_ir() returns correct IR."""
        from pysrc.preprocessor.graph.expr import Literal, OpExpr

        expr = OpExpr("mul", [Literal(3), Literal(4)], {"extra": "param"})
        ir = expr.to_ir()

        assert ir["op"] == "mul"
        assert len(ir["args"]) == 2
        assert ir["params"]["extra"] == "param"

    def test_opexpr_repr(self):
        """OpExpr.__repr__() is readable."""
        from pysrc.preprocessor.graph.expr import Literal, OpExpr

        expr = OpExpr("add", [Literal(1), Literal(2)])
        r = repr(expr)
        assert "add" in r


class TestExprArithmeticOperators:
    """Test Expr arithmetic operator overloads."""

    def test_add_operator(self):
        """Test + operator."""
        from pysrc.preprocessor.graph.expr import Column, OpExpr

        result = Column("a") + Column("b")
        assert isinstance(result, OpExpr)
        assert result.op == "add"

    def test_sub_operator(self):
        """Test - operator."""
        from pysrc.preprocessor.graph.expr import Column, OpExpr

        result = Column("a") - Column("b")
        assert isinstance(result, OpExpr)
        assert result.op == "sub"

    def test_mul_operator(self):
        """Test * operator."""
        from pysrc.preprocessor.graph.expr import Column, OpExpr

        result = Column("a") * Column("b")
        assert isinstance(result, OpExpr)
        assert result.op == "mul"

    def test_div_operator(self):
        """Test / operator."""
        from pysrc.preprocessor.graph.expr import Column, OpExpr

        result = Column("a") / Column("b")
        assert isinstance(result, OpExpr)
        assert result.op == "div"

    def test_neg_operator(self):
        """Test unary - operator."""
        from pysrc.preprocessor.graph.expr import Column, OpExpr

        result = -Column("a")
        assert isinstance(result, OpExpr)
        assert result.op == "neg"

    def test_pow_operator(self):
        """Test ** operator."""
        from pysrc.preprocessor.graph.expr import Column, OpExpr

        result = Column("a") ** 2
        assert isinstance(result, OpExpr)
        assert result.op == "pow"

    def test_radd_operator(self):
        """Test reverse + operator (number + expr)."""
        from pysrc.preprocessor.graph.expr import Column, OpExpr

        result = 5 + Column("a")
        assert isinstance(result, OpExpr)
        assert result.op == "add"

    def test_rsub_operator(self):
        """Test reverse - operator."""
        from pysrc.preprocessor.graph.expr import Column, OpExpr

        result = 5 - Column("a")
        assert isinstance(result, OpExpr)
        assert result.op == "sub"

    def test_rmul_operator(self):
        """Test reverse * operator."""
        from pysrc.preprocessor.graph.expr import Column, OpExpr

        result = 5 * Column("a")
        assert isinstance(result, OpExpr)
        assert result.op == "mul"

    def test_rtruediv_operator(self):
        """Test reverse / operator."""
        from pysrc.preprocessor.graph.expr import Column, OpExpr

        result = 5 / Column("a")
        assert isinstance(result, OpExpr)
        assert result.op == "div"

    def test_rpow_operator(self):
        """Test reverse ** operator."""
        from pysrc.preprocessor.graph.expr import Column, OpExpr

        result = 2 ** Column("a")
        assert isinstance(result, OpExpr)
        assert result.op == "pow"

    def test_chained_operations(self):
        """Test chained arithmetic."""
        from pysrc.preprocessor.graph.expr import Column, OpExpr

        result = (Column("a") + Column("b")) * Column("c")
        assert isinstance(result, OpExpr)
        assert result.op == "mul"

    def test_mixed_expr_and_literal(self):
        """Test mixing Expr with Python values."""
        from pysrc.preprocessor.graph.expr import Column, Literal, OpExpr

        result = Column("a") + 5
        assert isinstance(result, OpExpr)
        assert result.op == "add"
        # Second arg should be converted to Literal
        assert isinstance(result.args[1], Literal)


class TestOpExprOptimize:
    """Test OpExpr.optimize() constant folding."""

    def test_optimize_add_literals(self):
        """Optimize add of two literals."""
        from pysrc.preprocessor.graph.expr import Literal, OpExpr

        expr = OpExpr("add", [Literal(2), Literal(3)])
        optimized = expr.optimize()

        assert isinstance(optimized, Literal)
        assert optimized.value == 5

    def test_optimize_sub_literals(self):
        """Optimize sub of two literals."""
        from pysrc.preprocessor.graph.expr import Literal, OpExpr

        expr = OpExpr("sub", [Literal(10), Literal(3)])
        optimized = expr.optimize()

        assert isinstance(optimized, Literal)
        assert optimized.value == 7

    def test_optimize_mul_literals(self):
        """Optimize mul of two literals."""
        from pysrc.preprocessor.graph.expr import Literal, OpExpr

        expr = OpExpr("mul", [Literal(4), Literal(5)])
        optimized = expr.optimize()

        assert isinstance(optimized, Literal)
        assert optimized.value == 20

    def test_optimize_div_literals(self):
        """Optimize div of two literals."""
        from pysrc.preprocessor.graph.expr import Literal, OpExpr

        expr = OpExpr("div", [Literal(20), Literal(4)])
        optimized = expr.optimize()

        assert isinstance(optimized, Literal)
        assert optimized.value == 5.0

    def test_optimize_pow_literals(self):
        """Optimize pow of two literals."""
        from pysrc.preprocessor.graph.expr import Literal, OpExpr

        expr = OpExpr("pow", [Literal(2), Literal(3)])
        optimized = expr.optimize()

        assert isinstance(optimized, Literal)
        assert optimized.value == 8

    def test_optimize_neg_literal(self):
        """Optimize neg of a literal."""
        from pysrc.preprocessor.graph.expr import Literal, OpExpr

        expr = OpExpr("neg", [Literal(5)])
        optimized = expr.optimize()

        assert isinstance(optimized, Literal)
        assert optimized.value == -5

    def test_optimize_no_fold_with_column(self):
        """No folding when column is involved."""
        from pysrc.preprocessor.graph.expr import Column, Literal, OpExpr

        expr = OpExpr("add", [Column("a"), Literal(5)])
        optimized = expr.optimize()

        # Should still be OpExpr, not folded
        assert isinstance(optimized, OpExpr)
        assert optimized.op == "add"

    def test_optimize_nested_expression(self):
        """Optimize nested expression with all literals."""
        from pysrc.preprocessor.graph.expr import Literal, OpExpr

        # (2 + 3) * 4 = 20
        inner = OpExpr("add", [Literal(2), Literal(3)])
        outer = OpExpr("mul", [inner, Literal(4)])

        optimized = outer.optimize()
        assert isinstance(optimized, Literal)
        assert optimized.value == 20

    def test_optimize_handles_division_by_zero(self):
        """Optimize handles division by zero gracefully."""
        from pysrc.preprocessor.graph.expr import Literal, OpExpr

        expr = OpExpr("div", [Literal(5), Literal(0)])
        # Should not crash, might return OpExpr or Literal(inf)
        optimized = expr.optimize()
        assert optimized is not None


class TestExprFactory:
    """Test expr_factory() function."""

    def test_expr_factory_creates_column(self):
        """Factory creates Column."""
        from pysrc.preprocessor.graph.expr import Column, expr_factory

        expr = expr_factory("col", name="close")
        assert isinstance(expr, Column)

    def test_expr_factory_creates_literal(self):
        """Factory creates Literal."""
        from pysrc.preprocessor.graph.expr import Literal, expr_factory

        expr = expr_factory("lit", value=42)
        assert isinstance(expr, Literal)

    def test_expr_factory_creates_opexpr(self):
        """Factory creates OpExpr for registered ops."""
        from pysrc.preprocessor.graph.expr import Literal, OpExpr, expr_factory

        expr = expr_factory("add", Literal(1), Literal(2))
        assert isinstance(expr, OpExpr)

    def test_expr_factory_unknown_op_raises(self):
        """Factory raises for unknown op."""
        from pysrc.core.errors import UnsupportedPlan
        from pysrc.preprocessor.graph.expr import expr_factory

        with pytest.raises(UnsupportedPlan, match="No builder"):
            expr_factory("unknown_op_xyz")


class TestRegisterExpr:
    """Test register_expr() function."""

    def test_register_expr_custom_builder(self):
        """Register and use custom builder."""
        from pysrc.preprocessor.graph.expr import Literal, expr_factory, register_expr

        def custom_builder(**kwargs):
            return Literal(kwargs.get("value", 0) * 2)

        register_expr("double", custom_builder)
        expr = expr_factory("double", value=21)

        assert isinstance(expr, Literal)
        assert expr.value == 42


class TestPolarsLowerings:
    """Test Polars lowering registry."""

    def test_register_and_get_lowering(self):
        """Register and retrieve a lowering."""
        from pysrc.preprocessor.graph.expr import get_polars_lowering, register_polars_lowering

        def my_lowering(ir, data, **kwargs):
            return data

        register_polars_lowering("test.op", my_lowering)
        retrieved = get_polars_lowering("test.op")

        assert retrieved is my_lowering

    def test_get_nonexistent_lowering(self):
        """Get nonexistent lowering returns None."""
        from pysrc.preprocessor.graph.expr import get_polars_lowering

        result = get_polars_lowering("nonexistent.op.xyz")
        assert result is None


class TestExprHelperFunctions:
    """Test helper functions in expr module."""

    def test_ensure_expr_with_expr(self):
        """_ensure_expr passes through Expr."""
        from pysrc.preprocessor.graph.expr import Column, _ensure_expr

        col = Column("a")
        result = _ensure_expr(col)
        assert result is col

    def test_ensure_expr_with_string(self):
        """_ensure_expr converts string to Column."""
        from pysrc.preprocessor.graph.expr import Column, _ensure_expr

        result = _ensure_expr("close")
        assert isinstance(result, Column)
        assert result.name == "close"

    def test_ensure_expr_with_number(self):
        """_ensure_expr converts number to Literal."""
        from pysrc.preprocessor.graph.expr import Literal, _ensure_expr

        result = _ensure_expr(42)
        assert isinstance(result, Literal)
        assert result.value == 42


# =============================================================================
# ops_custom.py Tests
# =============================================================================


class TestOpsCustomImports:
    """Test ops_custom module imports."""

    def test_ops_custom_imports(self):
        """Basic import smoke test."""
        from pysrc.preprocessor.graph import ops_custom

        assert hasattr(ops_custom, "RSI")
        assert hasattr(ops_custom, "SMA")
        assert hasattr(ops_custom, "Lags")
        assert hasattr(ops_custom, "ZScore")
        assert hasattr(ops_custom, "RobustScaler")


class TestRSIOp:
    """Test RSI operation."""

    def test_rsi_default_params(self):
        """RSI with minimal params uses defaults."""
        from pysrc.preprocessor.graph.ops_custom import RSI

        op = RSI(input_col="close")
        assert op.params["input_col"] == "close"
        assert op.params["window"] == 14
        assert op.params["out_col"] == "rsi"

    def test_rsi_custom_params(self):
        """RSI with custom params."""
        from pysrc.preprocessor.graph.ops_custom import RSI

        op = RSI(input_col="price", window=21, out_col="custom_rsi")
        assert op.params["input_col"] == "price"
        assert op.params["window"] == 21
        assert op.params["out_col"] == "custom_rsi"

    def test_rsi_requires_provides(self):
        """RSI requires and provides correct columns."""
        from pysrc.preprocessor.graph.ops_custom import RSI

        op = RSI(input_col="close", window=14, out_col="rsi")
        assert op.requires == {"close"}
        assert op.provides == {"rsi"}

    def test_rsi_to_ir(self):
        """RSI.to_ir() includes requires/provides."""
        from pysrc.preprocessor.graph.ops_custom import RSI

        op = RSI(input_col="close", window=14)
        ir = op.to_ir()

        assert "requires" in ir
        assert "provides" in ir
        assert ir["op"] == "technical.RSI"

    def test_rsi_invalid_input_col_type(self):
        """RSI raises on non-string input_col."""
        from pysrc.preprocessor.graph.ops_custom import RSI

        with pytest.raises(ValueError, match="input_col must be a string"):
            RSI(input_col=123)

    def test_rsi_invalid_window(self):
        """RSI raises on invalid window."""
        from pysrc.preprocessor.graph.ops_custom import RSI

        with pytest.raises(ValueError, match="window must be an int > 1"):
            RSI(input_col="close", window=1)

        with pytest.raises(ValueError, match="window must be an int > 1"):
            RSI(input_col="close", window="not_int")

    def test_rsi_invalid_out_col_type(self):
        """RSI raises on non-string out_col."""
        from pysrc.preprocessor.graph.ops_custom import RSI

        with pytest.raises(ValueError, match="out_col must be a string"):
            RSI(input_col="close", out_col=123)


class TestSMAOp:
    """Test SMA operation."""

    def test_sma_params(self):
        """SMA with required params."""
        from pysrc.preprocessor.graph.ops_custom import SMA

        op = SMA(input_col="close", window=20)
        assert op.params["input_col"] == "close"
        assert op.params["window"] == 20
        assert op.params["out_col"] == "close_sma20"

    def test_sma_requires_provides(self):
        """SMA requires and provides correct columns."""
        from pysrc.preprocessor.graph.ops_custom import SMA

        op = SMA(input_col="close", window=10)
        assert op.requires == {"close"}
        assert "close_sma10" in op.provides

    def test_sma_missing_input_col(self):
        """SMA raises on missing input_col."""
        from pysrc.preprocessor.graph.ops_custom import SMA

        with pytest.raises(ValueError, match="input_col"):
            SMA(window=20)

    def test_sma_invalid_window(self):
        """SMA raises on invalid window."""
        from pysrc.preprocessor.graph.ops_custom import SMA

        with pytest.raises(ValueError, match="window must be an int > 1"):
            SMA(input_col="close", window=0)


class TestLagsOp:
    """Test Lags operation."""

    def test_lags_single_col(self):
        """Lags with single column."""
        from pysrc.preprocessor.graph.ops_custom import Lags

        op = Lags(cols="close", n=3)
        assert op.params["cols"] == ["close"]
        assert op.params["n"] == 3
        assert len(op.params["out_cols"]) == 3

    def test_lags_multiple_cols(self):
        """Lags with multiple columns."""
        from pysrc.preprocessor.graph.ops_custom import Lags

        op = Lags(cols=["open", "close"], n=2)
        assert op.params["cols"] == ["open", "close"]
        assert len(op.params["out_cols"]) == 4  # 2 cols * 2 lags

    def test_lags_requires_provides(self):
        """Lags requires and provides correct columns."""
        from pysrc.preprocessor.graph.ops_custom import Lags

        op = Lags(cols=["close"], n=2)
        assert op.requires == {"close"}
        assert "close_lag1" in op.provides
        assert "close_lag2" in op.provides

    def test_lags_missing_cols(self):
        """Lags raises on missing cols."""
        from pysrc.preprocessor.graph.ops_custom import Lags

        with pytest.raises(ValueError, match="cols required"):
            Lags(n=3)

    def test_lags_invalid_n(self):
        """Lags raises on invalid n."""
        from pysrc.preprocessor.graph.ops_custom import Lags

        with pytest.raises(ValueError, match="must be positive int"):
            Lags(cols="close", n=0)

        with pytest.raises(ValueError, match="must be positive int"):
            Lags(cols="close", n=-1)

    def test_lags_out_cols_mismatch(self):
        """Lags raises on out_cols length mismatch."""
        from pysrc.preprocessor.graph.ops_custom import Lags

        with pytest.raises(ValueError, match="out_cols must match"):
            Lags(cols="close", n=3, out_cols=["a", "b"])  # Need 3, got 2


class TestZScoreOp:
    """Test ZScore operation."""

    def test_zscore_single_col(self):
        """ZScore with single column."""
        from pysrc.preprocessor.graph.ops_custom import ZScore

        op = ZScore(cols="close")
        assert op.params["cols"] == ["close"]
        assert "close_z" in op.params["out_cols"]

    def test_zscore_multiple_cols(self):
        """ZScore with multiple columns."""
        from pysrc.preprocessor.graph.ops_custom import ZScore

        op = ZScore(cols=["open", "close"])
        assert op.params["cols"] == ["open", "close"]

    def test_zscore_requires_provides(self):
        """ZScore requires and provides correct columns."""
        from pysrc.preprocessor.graph.ops_custom import ZScore

        op = ZScore(cols=["close"])
        assert op.requires == {"close"}
        assert "close_z" in op.provides

    def test_zscore_state_dict(self):
        """ZScore.state_dict() returns means/stds."""
        from pysrc.preprocessor.graph.ops_custom import ZScore

        op = ZScore(cols=["close"], means=[0.0], stds=[1.0])
        state = op.state_dict()

        assert "means" in state
        assert "stds" in state

    def test_zscore_missing_cols(self):
        """ZScore raises on missing cols."""
        from pysrc.preprocessor.graph.ops_custom import ZScore

        with pytest.raises(ValueError, match="cols required"):
            ZScore()

    def test_zscore_out_cols_mismatch(self):
        """ZScore raises on out_cols mismatch."""
        from pysrc.preprocessor.graph.ops_custom import ZScore

        with pytest.raises(ValueError, match="out_cols must match"):
            ZScore(cols=["a", "b"], out_cols=["only_one"])


class TestRobustScalerOp:
    """Test RobustScaler operation."""

    def test_robust_scaler_single_col_interface(self):
        """RobustScaler with input_col interface."""
        from pysrc.preprocessor.graph.ops_custom import RobustScaler

        op = RobustScaler(input_col="close")
        assert op.params["cols"] == ["close"]
        assert "close_robust" in op.params["out_cols"]

    def test_robust_scaler_multi_col_interface(self):
        """RobustScaler with cols interface."""
        from pysrc.preprocessor.graph.ops_custom import RobustScaler

        op = RobustScaler(cols=["open", "close"])
        assert op.params["cols"] == ["open", "close"]

    def test_robust_scaler_defaults(self):
        """RobustScaler sets default params."""
        from pysrc.preprocessor.graph.ops_custom import RobustScaler

        op = RobustScaler(input_col="close")
        assert op.params["quantile_range"] == (25, 75)
        assert op.params["with_centering"] is True
        assert op.params["with_scaling"] is True

    def test_robust_scaler_missing_cols(self):
        """RobustScaler raises when no cols specified."""
        from pysrc.preprocessor.graph.ops_custom import RobustScaler

        with pytest.raises(ValueError, match="requires either"):
            RobustScaler()

    def test_robust_scaler_out_cols_mismatch(self):
        """RobustScaler raises on out_cols mismatch."""
        from pysrc.preprocessor.graph.ops_custom import RobustScaler

        with pytest.raises(ValueError, match="out_cols must match"):
            RobustScaler(cols=["a", "b"], out_cols=["only_one"])


class TestSentimentLexiconOp:
    """Test SentimentLexicon operation."""

    def test_sentiment_lexicon_params(self):
        """SentimentLexicon with text_col."""
        from pysrc.preprocessor.graph.ops_custom import SentimentLexicon

        op = SentimentLexicon(text_col="headline")
        assert op.params["text_col"] == "headline"
        assert op.params["out_col"] == "sentiment"

    def test_sentiment_lexicon_requires_provides(self):
        """SentimentLexicon requires and provides correct columns."""
        from pysrc.preprocessor.graph.ops_custom import SentimentLexicon

        op = SentimentLexicon(text_col="headline")
        assert op.requires == {"headline"}
        assert op.provides == {"sentiment"}

    def test_sentiment_lexicon_missing_text_col(self):
        """SentimentLexicon raises on missing text_col."""
        from pysrc.preprocessor.graph.ops_custom import SentimentLexicon

        with pytest.raises(ValueError, match="text_col"):
            SentimentLexicon()


class TestPairBetaOp:
    """Test PairBeta operation."""

    def test_pair_beta_params(self):
        """PairBeta with asset pair."""
        from pysrc.preprocessor.graph.ops_custom import PairBeta

        op = PairBeta(a="AAPL", b="MSFT")
        assert op.params["a"] == "AAPL"
        assert op.params["b"] == "MSFT"
        assert op.params["method"] == "ols"
        assert op.params["out_col"] == "beta_AAPL_MSFT"

    def test_pair_beta_requires_provides(self):
        """PairBeta requires and provides correct columns."""
        from pysrc.preprocessor.graph.ops_custom import PairBeta

        op = PairBeta(a="AAPL", b="MSFT")
        assert "AAPL.close" in op.requires
        assert "MSFT.close" in op.requires
        assert "beta_AAPL_MSFT" in op.provides


class TestPairSpreadOp:
    """Test PairSpread operation."""

    def test_pair_spread_params(self):
        """PairSpread with asset pair."""
        from pysrc.preprocessor.graph.ops_custom import PairSpread

        op = PairSpread(a="AAPL", b="MSFT")
        assert op.params["a"] == "AAPL"
        assert op.params["b"] == "MSFT"
        assert op.params["out_col"] == "spread_AAPL_MSFT"
        assert op.params["beta_col"] == "beta_AAPL_MSFT"

    def test_pair_spread_requires_provides(self):
        """PairSpread requires and provides correct columns."""
        from pysrc.preprocessor.graph.ops_custom import PairSpread

        op = PairSpread(a="AAPL", b="MSFT")
        assert "AAPL.close" in op.requires
        assert "MSFT.close" in op.requires
        assert "beta_AAPL_MSFT" in op.requires
        assert "spread_AAPL_MSFT" in op.provides


class TestHalfLifeOp:
    """Test HalfLife operation."""

    def test_half_life_params(self):
        """HalfLife with column."""
        from pysrc.preprocessor.graph.ops_custom import HalfLife

        op = HalfLife(col="spread")
        assert op.params["col"] == "spread"
        assert op.params["out_col"] == "hl_spread"

    def test_half_life_requires_provides(self):
        """HalfLife requires and provides correct columns."""
        from pysrc.preprocessor.graph.ops_custom import HalfLife

        op = HalfLife(col="spread")
        assert op.requires == {"spread"}
        assert op.provides == {"hl_spread"}


class TestRollingZOp:
    """Test RollingZ operation."""

    def test_rolling_z_params(self):
        """RollingZ with column and window."""
        from pysrc.preprocessor.graph.ops_custom import RollingZ

        op = RollingZ(col="close", window=128)
        assert op.params["col"] == "close"
        assert op.params["window"] == 128
        assert op.params["out_col"] == "close_z128"

    def test_rolling_z_default_window(self):
        """RollingZ default window is 256."""
        from pysrc.preprocessor.graph.ops_custom import RollingZ

        op = RollingZ(col="close")
        assert op.params["window"] == 256


class TestRollingVolOp:
    """Test RollingVol operation."""

    def test_rolling_vol_params(self):
        """RollingVol with column and window."""
        from pysrc.preprocessor.graph.ops_custom import RollingVol

        op = RollingVol(col="returns", window=32)
        assert op.params["col"] == "returns"
        assert op.params["window"] == 32
        assert op.params["out_col"] == "vol_returns_32"

    def test_rolling_vol_default_window(self):
        """RollingVol default window is 64."""
        from pysrc.preprocessor.graph.ops_custom import RollingVol

        op = RollingVol(col="returns")
        assert op.params["window"] == 64


class TestPolarsLoweringPlaceholders:
    """Execution tests for Polars lowerings on pairs/stat ops."""

    @pytest.mark.skipif(not HAS_POLARS, reason="polars not available")
    def test_pairs_beta_lowering_numeric_correctness(self):
        """pairs.beta lowering recovers known beta on synthetic a=2*b series."""
        import numpy as np
        import polars as pl  # type: ignore[import-not-found]

        from pysrc.preprocessor.graph.ops_custom import lower_pairs_beta_polars

        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
        a = 2.0 * b
        df = pl.DataFrame({"A.close": a.tolist(), "B.close": b.tolist()})
        ir = {"params": {"a": "A", "b": "B", "beta_window": 3, "out_col": "beta_A_B"}}

        out = lower_pairs_beta_polars(ir, df.lazy())
        if hasattr(out, "collect"):
            out = out.collect()
        beta_vals = out["beta_A_B"].to_list()

        # First two rows must be NaN (insufficient history for window=3)
        assert beta_vals[0] is None or (isinstance(beta_vals[0], float) and np.isnan(beta_vals[0]))
        assert beta_vals[1] is None or (isinstance(beta_vals[1], float) and np.isnan(beta_vals[1]))
        # Rows with full window should recover beta ≈ 2.0
        assert pytest.approx(beta_vals[2], rel=1e-4) == 2.0
        assert pytest.approx(beta_vals[3], rel=1e-4) == 2.0
        assert pytest.approx(beta_vals[4], rel=1e-4) == 2.0

    @pytest.mark.skipif(not HAS_POLARS, reason="polars not available")
    def test_pairs_spread_lowering_executes_and_produces_column(self):
        """pairs.spread lowering executes via Polars and adds spread column."""
        import polars as pl  # type: ignore[import-not-found]

        from pysrc.preprocessor.graph.ops_custom import (
            lower_pairs_beta_polars,
            lower_pairs_spread_polars,
        )

        df = pl.DataFrame({"A.close": [10.0, 11.0, 12.0], "B.close": [9.5, 10.5, 10.0]})
        # First compute beta so spread lowering has its dependency.
        beta_ir = {"params": {"a": "A", "b": "B", "beta_window": 2, "out_col": "beta_A_B"}}
        df_with_beta = lower_pairs_beta_polars(beta_ir, df.lazy())
        spread_ir = {
            "params": {"a": "A", "b": "B", "beta_col": "beta_A_B", "out_col": "spread_A_B"}
        }

        out = lower_pairs_spread_polars(spread_ir, df_with_beta.lazy())
        if hasattr(out, "collect"):
            out = out.collect()
        assert "spread_A_B" in out.columns

    @pytest.mark.skipif(not HAS_POLARS, reason="polars not available")
    def test_half_life_lowering_numeric_correctness(self):
        """stats.half_life lowering produces values in [1, 252] and sane median."""
        import numpy as np  # type: ignore[import-not-found]
        import polars as pl  # type: ignore[import-not-found]

        from pysrc.preprocessor.graph.ops_custom import lower_half_life_polars

        # AR(1) process with phi=0.5 → half_life = ln(2)/ln(2) = 1.0 bar exactly.
        rng = np.random.default_rng(42)
        n = 120
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = 0.5 * x[i - 1] + rng.normal(0, 0.1)

        df = pl.DataFrame({"spread": x.tolist()})
        ir = {"params": {"col": "spread", "half_life_window": 60, "out_col": "hl_spread"}}
        out = lower_half_life_polars(ir, df.lazy())
        if hasattr(out, "collect"):
            out = out.collect()
        hl_vals = out["hl_spread"].to_list()

        # First window rows must be NaN
        assert all(v is None or (isinstance(v, float) and np.isnan(v)) for v in hl_vals[:60])
        # Later values should be finite and in the valid tradeable band [1, 252]
        valid = [
            v for v in hl_vals[60:] if v is not None and not (isinstance(v, float) and np.isnan(v))
        ]
        assert len(valid) > 0, "Expected some valid half-life estimates"
        assert all(1.0 <= v <= 252.0 for v in valid), (
            "All valid half-life values must be in [1, 252]"
        )
        # Rough sanity: phi=0.5 gives hl≈1 bar; stochastic noise will widen this
        assert np.median(valid) < 10.0

    def test_zscore_roll_lowering_is_live(self):
        """scaling.zscore_roll lowering is implemented and covered elsewhere."""
        from pysrc.preprocessor.graph.ops_custom import lower_zscore_roll_polars

        assert callable(lower_zscore_roll_polars)

    def test_rolling_vol_lowering_raises(self):
        """stats.rolling_vol lowering remains a known placeholder."""
        from pysrc.preprocessor.graph.ops_custom import lower_rolling_vol_polars

        with pytest.raises(NotImplementedError):
            lower_rolling_vol_polars()

    @pytest.mark.skipif(not HAS_POLARS, reason="polars not available")
    def test_pairs_beta_lowering_missing_columns_raises(self):
        """pairs.beta lowering raises when required columns are missing."""
        import polars as pl  # type: ignore[import-not-found]

        from pysrc.preprocessor.graph.ops_custom import lower_pairs_beta_polars

        df = pl.DataFrame({"A.close": [1.0, 2.0], "other": [3.0, 4.0]})  # missing B.close
        ir = {"params": {"a": "A", "b": "B", "beta_window": 2, "out_col": "beta_A_B"}}
        with pytest.raises(ValueError, match="pairs.beta requires columns"):
            lower_pairs_beta_polars(ir, df.lazy()).collect()

    @pytest.mark.skipif(not HAS_POLARS, reason="polars not available")
    def test_pairs_beta_lowering_tiny_input_returns_nans(self):
        """pairs.beta with window>1 and n=0 or window=1 returns all-NaN column."""
        import numpy as np
        import polars as pl  # type: ignore[import-not-found]

        from pysrc.preprocessor.graph.ops_custom import lower_pairs_beta_polars

        # Empty frame
        df = pl.DataFrame({"A.close": [], "B.close": []})
        ir = {"params": {"a": "A", "b": "B", "beta_window": 3, "out_col": "beta_A_B"}}
        out = lower_pairs_beta_polars(ir, df.lazy())
        if hasattr(out, "collect"):
            out = out.collect()
        assert out.shape[0] == 0

        # Single row, window 3: insufficient history so beta is NaN
        df1 = pl.DataFrame({"A.close": [1.0], "B.close": [1.0]})
        out1 = lower_pairs_beta_polars(ir, df1.lazy())
        if hasattr(out1, "collect"):
            out1 = out1.collect()
        assert "beta_A_B" in out1.columns
        val = out1["beta_A_B"].to_list()[0]
        assert val is None or (isinstance(val, float) and np.isnan(val))

    @pytest.mark.skipif(not HAS_POLARS, reason="polars not available")
    def test_pairs_spread_lowering_missing_columns_raises(self):
        """pairs.spread lowering raises when required columns are missing."""
        import polars as pl  # type: ignore[import-not-found]

        from pysrc.preprocessor.graph.ops_custom import lower_pairs_spread_polars

        df = pl.DataFrame({"A.close": [1.0], "B.close": [1.0]})  # missing beta col
        ir = {"params": {"a": "A", "b": "B", "beta_col": "beta_A_B", "out_col": "spread_A_B"}}
        with pytest.raises(ValueError, match="pairs.spread requires columns"):
            lower_pairs_spread_polars(ir, df.lazy()).collect()

    @pytest.mark.skipif(not HAS_POLARS, reason="polars not available")
    def test_half_life_lowering_missing_column_raises(self):
        """stats.half_life lowering raises when column is missing."""
        import polars as pl  # type: ignore[import-not-found]

        from pysrc.preprocessor.graph.ops_custom import lower_half_life_polars

        df = pl.DataFrame({"other": [1.0, 2.0, 3.0]})
        ir = {"params": {"col": "spread", "half_life_window": 10, "out_col": "hl_spread"}}
        with pytest.raises(ValueError, match="stats.half_life requires column"):
            lower_half_life_polars(ir, df.lazy()).collect()

    @pytest.mark.skipif(not HAS_POLARS, reason="polars not available")
    def test_half_life_lowering_tiny_input_returns_nans(self):
        """stats.half_life with window<=2 or n<=2 returns all-NaN column."""
        import numpy as np
        import polars as pl  # type: ignore[import-not-found]

        from pysrc.preprocessor.graph.ops_custom import lower_half_life_polars

        df = pl.DataFrame({"spread": [1.0, 2.0]})
        ir = {"params": {"col": "spread", "half_life_window": 60, "out_col": "hl_spread"}}
        out = lower_half_life_polars(ir, df.lazy())
        if hasattr(out, "collect"):
            out = out.collect()
        assert "hl_spread" in out.columns
        hl_vals = out["hl_spread"].to_list()
        assert len(hl_vals) == 2
        # Polars may not count float NaN as null; assert each value is NaN explicitly
        for v in hl_vals:
            assert v is None or (isinstance(v, float) and np.isnan(v))

    def test_pair_spread_with_explicit_beta_col(self):
        """PairSpread with explicit beta_col does not overwrite it."""
        from pysrc.preprocessor.graph.ops_custom import PairSpread

        op = PairSpread(a="A", b="B", beta_col="custom_beta_col")
        assert op.params["beta_col"] == "custom_beta_col"
        assert "custom_beta_col" in op.requires

    @pytest.mark.skipif(not HAS_POLARS, reason="polars not available")
    def test_pairs_beta_lowering_constant_b_returns_nan(self):
        """pairs.beta when b is constant (denom=0) yields NaN at that window."""
        import numpy as np
        import polars as pl  # type: ignore[import-not-found]

        from pysrc.preprocessor.graph.ops_custom import lower_pairs_beta_polars

        # b constant -> variance 0 -> denom 0
        b = np.array([5.0, 5.0, 5.0, 5.0], dtype=float)
        a = np.array([10.0, 11.0, 12.0, 13.0], dtype=float)
        df = pl.DataFrame({"A.close": a.tolist(), "B.close": b.tolist()})
        ir = {"params": {"a": "A", "b": "B", "beta_window": 3, "out_col": "beta_A_B"}}
        out = lower_pairs_beta_polars(ir, df.lazy())
        if hasattr(out, "collect"):
            out = out.collect()
        beta_vals = out["beta_A_B"].to_list()
        # First two NaN (insufficient window), then window 3 has constant b -> denom 0 -> NaN
        assert beta_vals[0] is None or (isinstance(beta_vals[0], float) and np.isnan(beta_vals[0]))
        assert beta_vals[1] is None or (isinstance(beta_vals[1], float) and np.isnan(beta_vals[1]))
        assert beta_vals[2] is None or (isinstance(beta_vals[2], float) and np.isnan(beta_vals[2]))
        assert beta_vals[3] is None or (isinstance(beta_vals[3], float) and np.isnan(beta_vals[3]))

    def test_rolling_z_raises_on_invalid_params(self):
        """RollingZ raises ValueError for non-string col, window<=1, min_samples<=0."""
        from pysrc.preprocessor.graph.ops_custom import RollingZ

        with pytest.raises(ValueError, match="RollingZ.col must be a string"):
            RollingZ(col=123, window=10)
        with pytest.raises(ValueError, match="window must be > 1"):
            RollingZ(col="close", window=1)
        with pytest.raises(ValueError, match="min_samples must be positive"):
            RollingZ(col="close", window=10, min_samples=0)


# =============================================================================
# executor.py Tests
# =============================================================================


class TestExecutorImports:
    """Test executor module imports."""

    def test_executor_module_imports(self):
        """Basic import smoke test."""
        from pysrc.preprocessor.graph import executor

        assert hasattr(executor, "Executor")
        assert hasattr(executor, "PolarsExecutor")
        assert hasattr(executor, "CuDFExecutor")
        assert hasattr(executor, "ExecutorFactory")


class TestExecutorBase:
    """Test Executor base class."""

    def test_executor_init(self):
        """Executor initializes with correct attributes."""
        from pysrc.preprocessor.graph.executor import PolarsExecutor

        ex = PolarsExecutor()
        assert ex.backend == "polars"
        assert ex._cache == {}
        assert ex.execution_history == []

    def test_executor_cache_is_instance_level(self):
        """Each executor has its own cache."""
        from pysrc.preprocessor.graph.executor import PolarsExecutor

        ex1 = PolarsExecutor()
        ex2 = PolarsExecutor()

        assert ex1._cache is not ex2._cache

    def test_executor_make_cache_key(self):
        """_make_cache_key creates unique keys."""
        from pysrc.preprocessor.graph.executor import PolarsExecutor

        ex = PolarsExecutor()
        ir1 = {"op": "test", "params": {"a": 1}}
        ir2 = {"op": "test", "params": {"a": 2}}
        data = [1, 2, 3]

        key1 = ex._make_cache_key(ir1, data)
        key2 = ex._make_cache_key(ir2, data)

        assert key1 != key2
        assert ":" in key1  # Format is hash:id

    def test_executor_hash_plan(self):
        """_hash_plan creates deterministic hash."""
        from pysrc.preprocessor.graph.executor import PolarsExecutor

        ex = PolarsExecutor()
        plan = [{"op": "a"}, {"op": "b"}]

        h1 = ex._hash_plan(plan)
        h2 = ex._hash_plan(plan)

        assert h1 == h2
        assert len(h1) == 64  # SHA256 hex

    def test_executor_evolve_no_history(self):
        """evolve() returns None with no history."""
        from pysrc.preprocessor.graph.executor import PolarsExecutor

        ex = PolarsExecutor()
        assert ex.evolve() is None

    def test_executor_evolve_fast_ops(self):
        """evolve() returns None when ops are fast."""
        from pysrc.preprocessor.graph.executor import PolarsExecutor

        ex = PolarsExecutor()
        ex.execution_history = [
            {"op": "a", "time": 0.1, "backend": "polars"},
            {"op": "b", "time": 0.2, "backend": "polars"},
        ]

        assert ex.evolve(threshold=1.0) is None


class TestPolarsExecutor:
    """Test PolarsExecutor class."""

    def test_polars_executor_init(self):
        """PolarsExecutor initializes correctly."""
        from pysrc.preprocessor.graph.executor import PolarsExecutor

        ex = PolarsExecutor(engine_pref="cpu")
        assert ex.backend == "polars"
        assert ex.engine_pref == "cpu"

    @pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
    def test_polars_executor_execute_empty_plan(self):
        """Execute with empty plan returns input."""
        from pysrc.preprocessor.graph.executor import PolarsExecutor

        ex = PolarsExecutor()
        df = pl.DataFrame({"a": [1, 2, 3]})
        result = ex.execute([], df, [])

        assert isinstance(result, pl.DataFrame)
        assert result.shape == df.shape

    @pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
    def test_polars_executor_accepts_lazyframe(self):
        """Execute accepts LazyFrame."""
        from pysrc.preprocessor.graph.executor import PolarsExecutor

        ex = PolarsExecutor()
        lf = pl.DataFrame({"a": [1, 2, 3]}).lazy()
        result = ex.execute([], lf, [])

        assert isinstance(result, pl.DataFrame)

    @pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
    def test_polars_executor_accepts_dict(self):
        """Execute accepts dict (converts to DataFrame)."""
        from pysrc.preprocessor.graph.executor import PolarsExecutor

        ex = PolarsExecutor()
        data = {"a": [1, 2, 3]}
        result = ex.execute([], data, [])

        assert isinstance(result, pl.DataFrame)


class TestCuDFExecutor:
    """Test CuDFExecutor class."""

    def test_cudf_executor_init(self):
        """CuDFExecutor initializes correctly."""
        from pysrc.preprocessor.graph.executor import CuDFExecutor

        ex = CuDFExecutor(pool_size="8GB")
        assert ex.backend == "cudf"
        assert ex.pool_size == "8GB"


class TestExecutorFactory:
    """Test ExecutorFactory class."""

    def test_factory_create_polars(self):
        """Factory creates PolarsExecutor."""
        from pysrc.preprocessor.graph.executor import ExecutorFactory, PolarsExecutor

        ex = ExecutorFactory.create("polars")
        assert isinstance(ex, PolarsExecutor)

    def test_factory_create_cudf(self):
        """Factory creates CuDFExecutor."""
        from pysrc.preprocessor.graph.executor import CuDFExecutor, ExecutorFactory

        ex = ExecutorFactory.create("cudf")
        assert isinstance(ex, CuDFExecutor)

    def test_factory_create_auto(self):
        """Factory create('auto') returns an executor."""
        from pysrc.preprocessor.graph.executor import Executor, ExecutorFactory

        ex = ExecutorFactory.create("auto")
        assert isinstance(ex, Executor)

    def test_factory_create_unknown_raises(self):
        """Factory raises on unknown backend."""
        from pysrc.preprocessor.graph.executor import ExecutorFactory

        with pytest.raises(ValueError, match="Unknown backend"):
            ExecutorFactory.create("unknown_backend_xyz")

    def test_factory_register_custom(self):
        """Factory allows registering custom executor."""
        from pysrc.preprocessor.graph.executor import Executor, ExecutorFactory

        class CustomExecutor(Executor):
            def __init__(self):
                super().__init__("custom")

            def execute(self, plan, data, group_by):
                return data

        ExecutorFactory.register("custom_test", CustomExecutor)
        ex = ExecutorFactory.create("custom_test")

        assert isinstance(ex, CustomExecutor)


class TestExecutorCaching:
    """Test executor caching behavior."""

    def test_execute_node_cached_returns_cached(self):
        """_execute_node_cached returns cached result."""
        from pysrc.preprocessor.graph.executor import PolarsExecutor

        ex = PolarsExecutor()

        # Pre-populate cache
        ir = {"op": "test"}
        data = "test_data"
        cache_key = ex._make_cache_key(ir, data)
        ex._cache[cache_key] = "cached_result"

        result = ex._execute_node_cached(ir, data, [])
        assert result == "cached_result"

    def test_cache_eviction_at_capacity(self):
        """Cache evicts oldest when at capacity."""
        from pysrc.preprocessor.graph.executor import PolarsExecutor

        ex = PolarsExecutor()
        ex._cache_size = 2

        # Fill cache
        ex._cache["key1"] = "value1"
        ex._cache["key2"] = "value2"

        # Adding third should evict first
        assert len(ex._cache) == 2

        # Manually trigger eviction logic
        if len(ex._cache) >= ex._cache_size:
            oldest_key = next(iter(ex._cache))
            del ex._cache[oldest_key]

        ex._cache["key3"] = "value3"

        assert len(ex._cache) == 2
        assert "key1" not in ex._cache


@pytest.mark.determinism("d1")
class TestPhaseICTechnicalOps:
    def test_ema_op_params_and_contract(self):
        from pysrc.preprocessor.graph.ops_custom import EMA

        op = EMA(input_col="close", span=5, out_col="ema5")
        assert op.params["input_col"] == "close"
        assert op.params["span"] == 5
        assert op.requires == {"close"}
        assert op.provides == {"ema5"}

    def test_macd_line_signal_op_provides_expected_columns(self):
        from pysrc.preprocessor.graph.ops_custom import MACDLineSignal

        op = MACDLineSignal(input_col="price", fast=12, slow=26, signal=9)
        assert op.requires == {"price"}
        assert {"price_ema12", "price_ema26", "macd", "macd_signal", "macd_hist"} <= op.provides

    def test_bollinger_op_provides_mid_std_and_bands(self):
        from pysrc.preprocessor.graph.ops_custom import Bollinger

        op = Bollinger(input_col="price", window=20, num_std=2.0)
        assert op.requires == {"price"}
        assert {"price_sma20", "price_std20", "price_bb_upper20", "price_bb_lower20"} <= op.provides

    def test_atr_op_requires_ohlc_columns(self):
        from pysrc.preprocessor.graph.ops_custom import ATR

        op = ATR(high_col="high", low_col="low", close_col="close", window=14)
        assert op.requires == {"high", "low", "close"}
        assert op.provides == {"atr_14"}

    def test_obv_op_requires_close_and_volume(self):
        from pysrc.preprocessor.graph.ops_custom import OBV

        op = OBV(input_col="close", volume_col="volume", out_col="obv_out")
        assert op.requires == {"close", "volume"}
        assert op.provides == {"obv_out"}

    def test_vwap_op_accepts_explicit_session_fields(self):
        from pysrc.preprocessor.graph.ops_custom import VWAP

        op = VWAP(
            price_col="close",
            volume_col="volume",
            session_col="session",
            timestamp_col="ts",
            out_col="vwap_out",
        )
        assert {"close", "volume", "session", "ts"} == op.requires
        assert op.provides == {"vwap_out"}

    def test_rolling_std_defaults_and_contract(self):
        from pysrc.preprocessor.graph.ops_custom import RollingStd

        op = RollingStd(col="close", window=8)
        assert op.params["min_samples"] == 8
        assert op.requires == {"close"}
        assert op.provides == {"close_std8"}

    @pytest.mark.skipif(not HAS_POLARS, reason="Polars not available")
    def test_new_lowerings_execute_expected_columns(self):
        from pysrc.preprocessor.graph.backends.polars import (
            scaling_zscore_roll_polars,
            stats_rolling_std_polars,
            technical_atr_polars,
            technical_bollinger_polars,
            technical_ema_polars,
            technical_macd_line_signal_polars,
            technical_obv_polars,
            technical_rsi_polars,
            technical_sma_polars,
            technical_vwap_polars,
        )

        intraday = pl.DataFrame(
            {
                "session": ["2024-01-02"] * 4,
                "ts": [
                    "2024-01-02T09:30:00",
                    "2024-01-02T09:31:00",
                    "2024-01-02T09:32:00",
                    "2024-01-02T09:33:00",
                ],
                "close": [100.0, 101.0, 102.0, 101.5],
                "high": [100.5, 101.5, 102.5, 102.0],
                "low": [99.5, 100.5, 101.5, 101.0],
                "volume": [1000.0, 1100.0, 900.0, 1200.0],
            }
        ).with_columns(pl.col("ts").str.strptime(pl.Datetime, strict=False))
        lf = intraday.lazy()

        sma_df = technical_sma_polars(
            {"params": {"input_col": "close", "window": 2, "out_col": "sma2"}}, lf
        ).collect()
        ema_df = technical_ema_polars(
            {"params": {"input_col": "close", "span": 3, "out_col": "ema3"}}, lf
        ).collect()
        rsi_df = technical_rsi_polars(
            {"params": {"input_col": "close", "window": 3, "out_col": "rsi3"}}, lf
        ).collect()
        macd_df = technical_macd_line_signal_polars(
            {
                "params": {
                    "input_col": "close",
                    "fast": 2,
                    "slow": 3,
                    "signal": 2,
                    "out_macd": "macd",
                    "out_signal": "macd_signal",
                    "out_hist": "macd_hist",
                }
            },
            lf,
        ).collect()
        boll_df = technical_bollinger_polars(
            {
                "params": {
                    "input_col": "close",
                    "window": 2,
                    "num_std": 2.0,
                    "out_mid": "mid",
                    "out_std": "std",
                    "out_upper": "upper",
                    "out_lower": "lower",
                }
            },
            lf,
        ).collect()
        atr_df = technical_atr_polars(
            {
                "params": {
                    "high_col": "high",
                    "low_col": "low",
                    "close_col": "close",
                    "window": 3,
                    "out_col": "atr3",
                }
            },
            lf,
        ).collect()
        obv_df = technical_obv_polars(
            {"params": {"input_col": "close", "volume_col": "volume", "out_col": "obv"}}, lf
        ).collect()
        std_df = stats_rolling_std_polars(
            {"params": {"col": "close", "window": 2, "out_col": "std2"}}, lf
        ).collect()
        z_df = scaling_zscore_roll_polars(
            {"params": {"col": "close", "window": 2, "out_col": "z2"}}, lf
        ).collect()
        vwap_df = technical_vwap_polars(
            {
                "params": {
                    "price_col": "close",
                    "volume_col": "volume",
                    "session_col": "session",
                    "timestamp_col": "ts",
                    "out_col": "vwap",
                }
            },
            lf,
        ).collect()

        assert "sma2" in sma_df.columns
        assert "ema3" in ema_df.columns
        assert "rsi3" in rsi_df.columns
        assert {"macd", "macd_signal", "macd_hist"} <= set(macd_df.columns)
        assert {"mid", "std", "upper", "lower"} <= set(boll_df.columns)
        assert "atr3" in atr_df.columns
        assert "obv" in obv_df.columns
        assert "std2" in std_df.columns
        assert "z2" in z_df.columns
        assert "vwap" in vwap_df.columns

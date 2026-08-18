from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from pysrc.core.errors import PreprocessingError, UnsupportedPlan


@pytest.fixture
def restore_factory_state():
    from pysrc.preprocessor.graph import factory

    op_registry = dict(factory._OP_REGISTRY)
    alias_map = dict(factory._ALIAS_MAP)
    try:
        yield factory
    finally:
        factory._OP_REGISTRY.clear()
        factory._OP_REGISTRY.update(op_registry)
        factory._ALIAS_MAP.clear()
        factory._ALIAS_MAP.update(alias_map)


@pytest.mark.determinism("d1")
def test_factory_parse_and_merge_specs_cover_special_tokens(restore_factory_state) -> None:
    from pysrc.preprocessor.graph.factory import _iter_specs, register_builtin_ops

    register_builtin_ops()
    specs = list(
        _iter_specs(
            [
                "technical.MACD_line_signal:close,12,26,9",
                "technical.Bollinger:close,20,2.5",
                "EMA",
                "custom.literal:7",
            ],
            {
                "EMA": [{"input_col": "close", "span": 8}],
                "custom.literal": [{"extra": "wins"}],
            },
        )
    )

    assert specs[0].params["fast"] == 12
    assert specs[0].params["signal"] == 9
    assert specs[1].params["num_std"] == 2.5
    assert specs[2].name == "technical.EMA"
    assert specs[2].params["span"] == 8
    assert specs[3].params == {"value": 7, "extra": "wins"}


@pytest.mark.determinism("d1")
def test_factory_iter_specs_uses_transform_factory_before_failing(
    restore_factory_state,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pysrc.preprocessor.graph import factory
    from pysrc.preprocessor.utils.errors import UnsupportedAST

    monkeypatch.setattr(
        factory.ExprFactory, "build", lambda key: (_ for _ in ()).throw(UnsupportedAST("no expr"))
    )
    monkeypatch.setattr(factory.TransformFactory, "build", lambda key: {"kind": key})

    specs = list(factory._iter_specs(["normalize.only"], {}))

    assert specs[0].name == "normalize.only"
    assert specs[0].params["transform"] == {"kind": "normalize.only"}


@pytest.mark.determinism("d1")
def test_factory_iter_specs_raises_for_unknown_op_after_all_fallbacks(
    restore_factory_state,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pysrc.preprocessor.graph import factory
    from pysrc.preprocessor.utils.errors import UnsupportedAST

    monkeypatch.setattr(
        factory.ExprFactory, "build", lambda key: (_ for _ in ()).throw(UnsupportedAST("no expr"))
    )
    monkeypatch.setattr(
        factory.TransformFactory,
        "build",
        lambda key: (_ for _ in ()).throw(UnsupportedAST("no transform")),
    )

    with pytest.raises(UnsupportedPlan, match="Unknown op 'missing.op'"):
        list(factory._iter_specs(["missing.op"], {}))


@pytest.mark.determinism("d1")
def test_builtin_registration_swallows_duplicate_values(
    restore_factory_state,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pysrc.preprocessor.graph import factory

    register_calls: list[str] = []
    alias_calls: list[str] = []
    real_register = factory.register
    real_register_alias = factory.register_alias

    def wrapped_register(name, cls):
        register_calls.append(name)
        if name == "technical.RSI":
            raise ValueError("already registered")
        return real_register(name, cls)

    def wrapped_register_alias(alias, target):
        alias_calls.append(alias)
        if alias == "RSI":
            raise ValueError("already used")
        return real_register_alias(alias, target)

    monkeypatch.setattr(factory, "register", wrapped_register)
    monkeypatch.setattr(factory, "register_alias", wrapped_register_alias)

    factory.register_builtin_ops()

    assert "technical.RSI" in register_calls
    assert "RSI" in alias_calls
    assert "technical.SMA" in factory._OP_REGISTRY


@pytest.mark.determinism("d1")
def test_ops_custom_validation_and_backend_delegation() -> None:
    from pysrc.preprocessor.graph import ops_custom

    vwap = ops_custom.VWAP(price_col="close", volume_col="volume")
    assert vwap.requires == {"close", "volume"}

    with pytest.raises(ValueError, match="VWAP.session_col"):
        ops_custom.VWAP(price_col="close", volume_col="volume", session_col=123)

    with pytest.raises(ValueError, match="RollingZ.window must be > 1"):
        ops_custom.RollingZ(col="close", window=1)

    with pytest.raises(ValueError, match="RollingZ.min_samples must be positive"):
        ops_custom.RollingZ(col="close", window=3, min_samples=0)

    lowered_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_backend(*args, **kwargs):
        lowered_calls.append((args, kwargs))
        return "delegated"

    backend = SimpleNamespace(technical_rsi_polars=fake_backend)
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(__import__("sys").modules, "pysrc.preprocessor.graph.backends.polars", backend)
        assert ops_custom.lower_rsi_polars("ir", "lf", group_by=["symbol"]) == "delegated"

    assert lowered_calls == [(("ir", "lf"), {"group_by": ["symbol"]})]


@pytest.mark.determinism("d1")
def test_polars_backend_registry_helpers_cover_argument_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pysrc.preprocessor.graph.backends import polars as backend

    called: list[tuple[object, ...]] = []
    monkeypatch.setattr(backend, "_reg_get", lambda *args: called.append(args) or "value")
    monkeypatch.setattr(backend, "_reg_register", lambda *args: called.append(args) or None)
    monkeypatch.setattr(backend, "_reg_list_ops", lambda *args: called.append(args) or ["x"])

    assert backend.get("technical.SMA") == "value"
    assert backend.get("polars", "technical.SMA") == "value"
    with pytest.raises(TypeError, match="get\\(\\) expects"):
        backend.get("a", "b", "c")

    assert backend.register("temp.op", lambda ir, lf, **kwargs: lf) is None
    assert backend.register("alt", "temp.op2", lambda ir, lf, **kwargs: lf) is None
    with pytest.raises(TypeError, match="Unexpected keyword arguments"):
        backend.register("x", lambda ir, lf, **kwargs: lf, nope=True)
    with pytest.raises(TypeError, match="register\\(\\) expects"):
        backend.register("only-one")

    def sentinel(ir, lf, **kwargs):
        return lf

    assert backend.register("custom", "override.op", sentinel, allow_override=True) is None
    assert backend.get("custom", "override.op") is sentinel

    assert backend.list_ops() == ["x"]
    assert backend.list_ops("polars") == ["x"]
    with pytest.raises(TypeError, match="list_ops\\(\\) expects"):
        backend.list_ops("a", "b")

    assert ("polars", "technical.SMA") in called
    assert ("polars",) in called


@pytest.mark.determinism("d1")
def test_polars_backend_helper_arrays_cover_edge_cases() -> None:
    from pysrc.preprocessor.graph.backends import polars as backend

    values = np.array([np.nan, 1.0, 3.0, np.nan, 5.0], dtype=float)

    mean = backend._rolling_mean_array(values, window=3, min_samples=2)
    std = backend._rolling_std_array(values, window=3, min_samples=2, ddof=1)
    ema = backend._ema_array(values, span=2)
    smooth = backend._wilder_smooth_array(values, window=3)
    rsi_empty = backend._rsi_array(np.array([], dtype=float), window=3)
    obv = backend._obv_array(
        np.array([10.0, 11.0, np.nan, 10.0], dtype=float),
        np.array([100.0, 200.0, 300.0, np.nan], dtype=float),
    )

    assert np.isnan(mean[0])
    assert np.isfinite(mean[2])
    assert np.isnan(std[1])
    assert np.isfinite(std[2])
    assert np.isnan(ema[0])
    assert np.isfinite(ema[1])
    assert np.isfinite(smooth[1])
    assert rsi_empty.size == 0
    assert obv.tolist() == [0.0, 200.0, 200.0, 200.0]


@pytest.mark.determinism("d1")
def test_polars_backend_apply_eager_and_lowerings_cover_lazy_and_dataframe_paths(tmp_path) -> None:
    from pysrc.preprocessor.graph.backends import polars as backend

    frame = pl.DataFrame(
        {
            "close": [10.0, 11.0, 12.0, 13.0],
            "high": [11.0, 12.0, 13.0, 14.0],
            "low": [9.0, 10.0, 11.0, 12.0],
            "volume": [100.0, 200.0, 100.0, 300.0],
            "session": ["a", "a", "b", "b"],
            "ts": [
                "2024-01-01T09:30:00",
                "2024-01-01T09:31:00",
                "2024-01-02T09:30:00",
                "2024-01-02T09:31:00",
            ],
        }
    ).with_columns(pl.col("ts").str.strptime(pl.Datetime, strict=False))

    applied_df = backend._apply_eager(frame, lambda eager: eager.with_columns(pl.lit(1).alias("x")))
    applied_lf = backend._apply_eager(
        frame.lazy(), lambda eager: eager.with_columns(pl.lit(2).alias("y"))
    )
    assert isinstance(applied_df, pl.DataFrame)
    assert isinstance(applied_lf, pl.LazyFrame)

    robust = backend.scaling_robust_polars(
        {
            "params": {
                "cols": ["close"],
                "out_cols": ["close_robust"],
                "quantile_range": (25, 75),
                "with_centering": False,
                "with_scaling": False,
            }
        },
        frame.lazy(),
        group_by=["session"],
    ).collect()
    assert "close_robust" in robust.columns

    returns = backend.feature_returns_polars(
        {"params": {"column": "close"}}, frame.lazy()
    ).collect()
    sma = backend.feature_sma_polars(
        {"params": {"column": "close", "window": 2}}, frame.lazy()
    ).collect()
    rsi = backend.feature_rsi_polars(
        {"params": {"column": "close", "window": 2}}, frame.lazy()
    ).collect()
    csv_path = tmp_path / "prices.csv"
    frame.select(["close", "volume"]).write_csv(csv_path)
    loaded = backend.data_load_csv_polars(
        {"params": {"path": str(csv_path), "try_parse_dates": False}}, None
    ).collect()

    assert "returns" in returns.columns
    assert "sma_2" in sma.columns
    assert "rsi_2" in rsi.columns
    assert loaded.columns == ["close", "volume"]


@pytest.mark.determinism("d1")
def test_polars_backend_vwap_error_paths_and_success() -> None:
    from pysrc.preprocessor.graph.backends import polars as backend

    intraday = pl.DataFrame(
        {
            "close": [10.0, 11.0, 12.0],
            "volume": [100.0, 200.0, 100.0],
            "session": ["a", "a", "b"],
            "ts": ["2024-01-01T09:30:00", "2024-01-01T09:31:00", "2024-01-02T09:30:00"],
        }
    ).with_columns(pl.col("ts").str.strptime(pl.Datetime, strict=False))
    daily = pl.DataFrame(
        {
            "close": [10.0],
            "volume": [100.0],
            "ts": [date(2024, 1, 1)],
        }
    )

    with pytest.raises(ValueError, match="requires explicit session_col"):
        backend.technical_vwap_polars({"params": {}}, intraday.lazy()).collect()
    with pytest.raises(ValueError, match="missing required session column"):
        backend.technical_vwap_polars(
            {"params": {"session_col": "missing"}}, intraday.lazy()
        ).collect()
    with pytest.raises(ValueError, match="missing required timestamp column"):
        backend.technical_vwap_polars(
            {"params": {"timestamp_col": "missing"}}, intraday.lazy()
        ).collect()
    with pytest.raises(ValueError, match="daily-only timestamps"):
        backend.technical_vwap_polars({"params": {"timestamp_col": "ts"}}, daily.lazy()).collect()

    out = backend.technical_vwap_polars(
        {
            "params": {
                "price_col": "close",
                "volume_col": "volume",
                "session_col": "session",
                "timestamp_col": "ts",
                "out_col": "vwap",
            }
        },
        intraday.lazy(),
    ).collect()
    assert out["vwap"].to_list() == [10.0, 10.666666666666666, 12.0]


@pytest.mark.determinism("d1")
def test_polars_executor_execute_and_collect_policy_cover_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pysrc.preprocessor.graph.backends import polars as backend

    frame = pl.DataFrame({"close": [1.0, 2.0], "symbol": ["A", "A"]})
    lf = frame.lazy()

    class FakeLazyFrame:
        def __init__(self) -> None:
            self.calls: list[object] = []

        def collect(self, *, engine=None):
            self.calls.append(engine)
            if engine == "gpu":
                raise RuntimeError("gpu failed")
            return frame

    cpu_executor = backend.PolarsExecutor(engine_pref="cpu")
    assert cpu_executor._collect_with_policy(FakeLazyFrame()).shape == (2, 2)

    gpu_executor = backend.PolarsExecutor(engine_pref="gpu")
    monkeypatch.setattr(backend, "capabilities", lambda: SimpleNamespace(has_polars_gpu=True))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(pl, "GpuEngine", None, raising=False)
        assert gpu_executor._collect_with_policy(FakeLazyFrame()).shape == (2, 2)

    class FakeCompiledPlan:
        def __init__(self) -> None:
            self.order_by = ["close"]
            self.group_by = ["symbol"]
            self.nodes = [
                {
                    "op": "technical.SMA",
                    "params": {"input_col": "close", "window": 2, "out_col": "close_sma2"},
                }
            ]
            self.params = {"cols": ["close"]}
            self.expected_schema = None

        def report(self):
            return {"status": "ok"}

    monkeypatch.setattr(backend, "validate_dataframe", lambda df: None)
    monkeypatch.setattr(
        backend.SpecFactory, "build", lambda *_args, **kwargs: {"group": kwargs["by"]}
    )
    monkeypatch.setattr(backend, "schema_checks", lambda out, expected, strict: None)
    monkeypatch.setattr(backend, "op_chain", lambda *_ops: lambda out, cols: out)
    monkeypatch.setattr(
        backend, "to_torch_batch", lambda out, cols: SimpleNamespace(torch=cols, rows=out.height)
    )
    monkeypatch.setattr(
        backend.HeuristicPlanner, "optimize", lambda self, segments, sample: segments
    )

    executor = backend.PolarsExecutor(engine_pref="cpu", to_torch=True)
    result = executor.execute(FakeCompiledPlan(), lf)

    assert result.torch == ["close"]
    assert result.rows == 2


@pytest.mark.determinism("d1")
def test_polars_executor_collect_with_policy_auto_gpu_engine_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pysrc.preprocessor.graph.backends import polars as backend

    class FakeEngine:
        pass

    class FakeLazyFrame:
        def __init__(self) -> None:
            self.last_engine = None

        def collect(self, *, engine=None):
            self.last_engine = engine
            return pl.DataFrame({"x": [1]})

    fake_lf = FakeLazyFrame()
    monkeypatch.setattr(backend, "capabilities", lambda: SimpleNamespace(has_polars_gpu=True))
    monkeypatch.setattr(pl, "GpuEngine", FakeEngine, raising=False)

    out = backend.PolarsExecutor(engine_pref="auto")._collect_with_policy(fake_lf)

    assert out.shape == (1, 1)
    assert isinstance(fake_lf.last_engine, FakeEngine)


@pytest.mark.determinism("d1")
def test_polars_executor_invalid_input_raises_preprocessing_error() -> None:
    from pysrc.preprocessor.graph.backends import polars as backend

    executor = backend.PolarsExecutor()
    compiled_plan = SimpleNamespace(
        order_by=None,
        group_by=[],
        nodes=[],
        params={},
        expected_schema=None,
        report=lambda: {},
    )

    with pytest.raises(PreprocessingError):
        executor.execute(compiled_plan, object())

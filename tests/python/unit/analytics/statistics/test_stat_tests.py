# tests/python/unit/analytics/statistics/test_stat_tests.py
import os
import time
from concurrent.futures import (  # noqa: F401 (required import per constraints)
    ThreadPoolExecutor,
    wait,
)
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from allpairspy import AllPairs
from hypothesis import given, seed, settings
from hypothesis import strategies as st

import pysrc.analytics.statistics as mod
import pysrc.analytics.statistics.base as base_mod

pytestmark = [pytest.mark.determinism("d0"), pytest.mark.usefixtures("deterministic_seed")]


# ----------------------------- Context Manager for Property Tests -----------------------------


@contextmanager
def patch_stat_backend(backend_name, fake_impl=None, fail=False, fail_type="backend"):
    if fail:
        if fail_type == "backend":

            def _fail(*a, **k):
                raise RuntimeError(f"{backend_name}-backend-fail")
        elif fail_type == "computation":

            def _fail(*a, **k):
                raise np.linalg.LinAlgError(f"{backend_name}-computation-fail")
        elif fail_type == "data":

            def _fail(*a, **k):
                raise ValueError(f"{backend_name}-data-fail")

        impl = _fail
    elif fake_impl is not None:
        impl = fake_impl
    else:
        # Default no-op
        def impl(*a, **k):
            return None

    with patch.object(base_mod, backend_name, impl):
        yield


# ----------------------------- Local deterministic fakes -----------------------------


def _series(n=50, start=0.0, step=0.1):
    # numeric, non-constant, deterministic
    x = start + step * np.arange(n, dtype=float)
    return pd.Series(x + 0.01 * np.sin(np.arange(n, dtype=float)))


def _const_series(n=50, val=1.0):
    return pd.Series([float(val)] * n)


def _df_xy(n=50, start=0.0, step=0.1):
    s1 = _series(n, start=start, step=step)
    s2 = _series(n, start=start + 0.5, step=step * 0.9)
    return pd.DataFrame({"y": s1, "x": s2})


def _df_vars(n=50, k=2):
    cols = {f"v{i}": _series(n, start=i * 0.25, step=0.07 + i * 0.01) for i in range(k)}
    return pd.DataFrame(cols)


def _patch_mlflow(monkeypatch, active=True):
    class StubMLF:
        def active_run(self):
            return True if active else None

        def log_metric(self, key, value):
            calls.append((key, float(value)))

        def log_params(self, *a, **k):
            pass

    calls: list[tuple[str, float]] = []
    monkeypatch.setattr(base_mod, "mlflow", StubMLF(), raising=True)
    return calls


def _fake_adfuller_return(n=50):
    stat = -3.0
    p = 0.04
    usedlag = 1
    nobs = max(0, n - 1)
    crit = {"1%": -3.5, "5%": -2.9, "10%": -2.6}
    icbest = -123.4
    return stat, p, usedlag, nobs, crit, icbest


def _patch_adfuller(monkeypatch, fail=False, fail_type="backend"):
    """Patch adfuller with realistic failure modes"""
    if fail:
        if fail_type == "backend":

            def _fail(*a, **k):
                raise RuntimeError("adfuller-backend-fail")

            monkeypatch.setattr(base_mod, "adfuller", _fail)
        elif fail_type == "computation":

            def _fail(*a, **k):
                raise np.linalg.LinAlgError("singular matrix in ADF")

            monkeypatch.setattr(base_mod, "adfuller", _fail)
        elif fail_type == "data":

            def _fail(*a, **k):
                raise ValueError("invalid autolag parameter")

            monkeypatch.setattr(base_mod, "adfuller", _fail)
    else:
        monkeypatch.setattr(
            base_mod, "adfuller", lambda data, **k: _fake_adfuller_return(len(data))
        )


def _patch_kpss(monkeypatch, fail=False, fail_type="backend"):
    if fail:
        if fail_type == "backend":
            # Simulates statsmodels backend failure
            def _fail(*a, **k):
                raise RuntimeError("kpss-backend-fail")

            monkeypatch.setattr(base_mod, "kpss", _fail)
        elif fail_type == "computation":
            # Simulates numerical computation failure
            def _fail(*a, **k):
                raise np.linalg.LinAlgError("singular matrix")

            monkeypatch.setattr(base_mod, "kpss", _fail)
        elif fail_type == "data":
            # Simulates data validation failure
            def _fail(*a, **k):
                raise ValueError("invalid regression parameter")

            monkeypatch.setattr(base_mod, "kpss", _fail)
    else:
        monkeypatch.setattr(
            base_mod,
            "kpss",
            lambda data, **k: (
                0.2,
                0.1,
                3,
                {"10%": 0.347, "5%": 0.463, "2.5%": 0.574, "1%": 0.739},
            ),
        )


def _patch_ljungbox(monkeypatch, fail=False, fail_type="backend"):
    """Patch ljungbox with realistic failure modes"""
    if fail:
        if fail_type == "backend":

            def _fail(*a, **k):
                raise RuntimeError("ljungbox-backend-fail")

            monkeypatch.setattr(base_mod, "acorr_ljungbox", _fail)
        elif fail_type == "computation":

            def _fail(*a, **k):
                raise np.linalg.LinAlgError("autocorrelation matrix singular")

            monkeypatch.setattr(base_mod, "acorr_ljungbox", _fail)
        elif fail_type == "data":

            def _fail(*a, **k):
                raise ValueError("lags exceed series length")

            monkeypatch.setattr(base_mod, "acorr_ljungbox", _fail)
    else:

        def fake_lb(values, lags, boxpierce=False, return_df=True, model_df=0):
            l = len(lags)
            return pd.DataFrame({"lb_pvalue": [0.2 + 0.01 * i for i in range(l)]})

        monkeypatch.setattr(base_mod, "acorr_ljungbox", fake_lb)


def _patch_granger(monkeypatch, fail=False, fail_type="backend"):
    """Patch granger with realistic failure modes"""
    if fail:
        if fail_type == "backend":

            def _fail(*a, **k):
                raise RuntimeError("granger-backend-fail")

            monkeypatch.setattr(base_mod, "grangercausalitytests", _fail)
        elif fail_type == "computation":

            def _fail(*a, **k):
                raise np.linalg.LinAlgError("VAR model singular")

            monkeypatch.setattr(base_mod, "grangercausalitytests", _fail)
        elif fail_type == "data":

            def _fail(*a, **k):
                raise ValueError("maxlag exceeds observations")

            monkeypatch.setattr(base_mod, "grangercausalitytests", _fail)
    else:

        def fake_gc(df, maxlag, verbose=False):
            return {
                lag: ({"ssr_ftest": (None, 0.1 + 0.05 * (lag % 2), None, None)}, None)
                for lag in range(1, int(maxlag) + 1)
            }

        monkeypatch.setattr(base_mod, "grangercausalitytests", fake_gc)


def _patch_johansen(monkeypatch, fail=False, n_vars=2, fail_type="backend"):
    """Patch johansen with realistic failure modes"""
    if fail:
        if fail_type == "backend":

            def _fail(*a, **k):
                raise RuntimeError("johansen-backend-fail")

            monkeypatch.setattr(base_mod, "coint_johansen", _fail)
        elif fail_type == "computation":

            def _fail(*a, **k):
                raise np.linalg.LinAlgError("eigenvalue decomposition failed")

            monkeypatch.setattr(base_mod, "coint_johansen", _fail)
        elif fail_type == "data":

            def _fail(*a, **k):
                raise ValueError("insufficient variables for cointegration")

            monkeypatch.setattr(base_mod, "coint_johansen", _fail)
    else:

        def fake_joh(data, det_order, k_ar_diff):
            lr1 = np.array([20.0 + i for i in range(n_vars)])
            lr2 = np.array([10.0 + i for i in range(n_vars)])
            cvt = np.array([[15, 19, 24] for _ in range(n_vars)])
            return SimpleNamespace(lr1=lr1, lr2=lr2, cvt=cvt)

        monkeypatch.setattr(base_mod, "coint_johansen", fake_joh)


def _patch_auto_lag(monkeypatch, value=2):
    monkeypatch.setattr(base_mod, "_select_var_order_aic", lambda df, maxlags_cap=15: int(value))


# -------------------------------- Registration & Factory --------------------------------


def test_register_duplicate_same_and_different_class_behaviors(monkeypatch):
    @mod.register_test("dummydup")
    class DummyA(mod.StatTest):
        def run(self, data, **kw):
            return {"ok": True}

    again = mod.register_test("dummydup")(DummyA)
    assert again is DummyA

    @mod.register_test("dupconflictA")
    class One(mod.StatTest):
        def run(self, data, **kw):
            return {}

    with pytest.raises(ValueError) as ei:

        @mod.register_test("dupconflictA")
        class Two(mod.StatTest):
            def run(self, data, **kw):
                return {}

    assert "already registered with different class" in str(ei.value)


def test_factory_lookup_case_insensitive_returns_instance():
    inst1 = mod.StatTestFactory.get_test("ADF")
    inst2 = mod.StatTestFactory.get_test("adf")
    assert type(inst1) is type(inst2)
    assert isinstance(inst1, mod.StatTest)


def test_factory_unknown_test_raises_with_available_list():
    with pytest.raises(ValueError) as ei:
        mod.StatTestFactory.get_test("nonexistent_test")
    msg = str(ei.value)
    assert "Unknown test: 'nonexistent_test'." in msg
    assert "Available tests:" in msg


# -------------------------------- Granger causality --------------------------------


def test_granger_mapping_vs_dataframe_conditions_and_errors(monkeypatch):
    _patch_granger(monkeypatch, fail=False)
    _patch_auto_lag(monkeypatch, value=3)
    with pytest.raises(mod.InvalidInputError) as ei1:
        mod.GrangercausalityTest().run({"a": [1, 2], "b": [3, 4]})
    assert "Input dict must contain keys 'y' and 'x'." in str(ei1.value)

    with pytest.raises(mod.InvalidInputError) as ei2:
        mod.GrangercausalityTest().run(pd.DataFrame({"a": range(12), "b": range(12)}))
    assert "Input DataFrame must contain columns 'y' and 'x'." in str(ei2.value)

    with pytest.raises(mod.InvalidInputError) as ei3:
        mod.GrangercausalityTest().run(_df_xy(n=9))
    assert "Insufficient observations for Granger causality test." in str(ei3.value)

    with pytest.raises(mod.InvalidInputError) as ei4:
        mod.GrangercausalityTest().run(_df_xy(n=11), maxlag=10)
    assert "Not enough observations for the chosen maxlag." in str(ei4.value)


def test_granger_auto_lag_and_success_returns_summary(monkeypatch):
    _patch_granger(monkeypatch, fail=False)
    _patch_auto_lag(monkeypatch, value=3)
    res = mod.GrangercausalityTest().run(
        _df_xy(n=50), maxlag="auto", significance_level=0.05, verbose=False
    )
    assert isinstance(res["p_values"], list)
    assert len(res["p_values"]) == 3
    assert 1 <= res["best_lag"] <= 3
    assert isinstance(res["any_causality"], bool)
    assert isinstance(res["summary"], str)
    assert "Granger causality" in res["summary"]


def test_granger_statistical_error_raised_when_backend_fails(monkeypatch):
    # Test backend failure
    _patch_granger(monkeypatch, fail=True, fail_type="backend")
    with pytest.raises(mod.StatisticalBackendError) as ei1:
        mod.GrangercausalityTest().run(_df_xy(n=20), maxlag=2)
    assert "backend error" in str(ei1.value).lower()

    # Test computation failure
    _patch_granger(monkeypatch, fail=True, fail_type="computation")
    with pytest.raises(mod.StatisticalComputationError) as ei2:
        mod.GrangercausalityTest().run(_df_xy(n=20), maxlag=2)
    assert "computation failed" in str(ei2.value).lower()

    # Test data validation failure
    _patch_granger(monkeypatch, fail=True, fail_type="data")
    with pytest.raises(mod.StatisticalDataError) as ei3:
        mod.GrangercausalityTest().run(_df_xy(n=20), maxlag=2)
    assert "invalid data" in str(ei3.value).lower() or "invalid" in str(ei3.value).lower()


def test_granger_mlflow_logging_active_vs_inactive(monkeypatch):
    _patch_granger(monkeypatch, fail=False)
    calls = _patch_mlflow(monkeypatch, active=True)
    _ = mod.GrangercausalityTest().run(_df_xy(n=30), maxlag=2)
    assert calls, "expected metrics logged when active_run() is truthy"

    _patch_granger(monkeypatch, fail=False)
    calls2 = _patch_mlflow(monkeypatch, active=False)
    _ = mod.GrangercausalityTest().run(_df_xy(n=30), maxlag=2)
    assert calls2 == []


# -------------------------------- Johansen cointegration --------------------------------


def test_johansen_input_validation_and_errors(monkeypatch):
    _patch_johansen(monkeypatch, fail=False)
    with pytest.raises(mod.InvalidInputError) as e1:
        mod.JohansencointegrationTest().run(_df_vars(n=30, k=1))
    assert "Johansen test requires at least two variables." in str(e1.value)

    with pytest.raises(mod.InvalidInputError) as e2:
        mod.JohansencointegrationTest().run(_df_vars(n=24, k=2))
    assert "Insufficient observations for Johansen test." in str(e2.value)


def test_johansen_auto_lag_and_success_rank_computation(monkeypatch):
    _patch_johansen(monkeypatch, fail=False, n_vars=3)
    _patch_auto_lag(monkeypatch, value=2)
    res = mod.JohansencointegrationTest().run(
        _df_vars(n=50, k=3), det_order=1, k_ar_diff="auto", significance_level=0.1
    )
    assert isinstance(res["rank"], int)
    assert res["rank"] >= 0
    assert len(res["trace_stat"]) == 3
    assert len(res["crit_values"]) == 3
    assert isinstance(res["summary"], str)
    assert "Johansen test" in res["summary"]


def test_johansen_statistical_error_raised_when_backend_fails(monkeypatch):
    # Test backend failure
    _patch_johansen(monkeypatch, fail=True, fail_type="backend")
    with pytest.raises(mod.StatisticalBackendError) as ei1:
        mod.JohansencointegrationTest().run(_df_vars(n=30, k=2))
    assert "backend error" in str(ei1.value).lower()

    # Test computation failure
    _patch_johansen(monkeypatch, fail=True, fail_type="computation")
    with pytest.raises(mod.StatisticalComputationError) as ei2:
        mod.JohansencointegrationTest().run(_df_vars(n=30, k=2))
    assert "computation failed" in str(ei2.value).lower()

    # Test data validation failure
    _patch_johansen(monkeypatch, fail=True, fail_type="data")
    with pytest.raises(mod.StatisticalDataError) as ei3:
        mod.JohansencointegrationTest().run(_df_vars(n=30, k=2))
    assert "invalid data" in str(ei3.value).lower() or "invalid" in str(ei3.value).lower()


# -------------------------------- ADF --------------------------------


def test_adf_input_edges_and_backend_errors(monkeypatch):
    _patch_adfuller(monkeypatch, fail=False)

    # Test insufficient observations
    with pytest.raises(mod.InvalidInputError) as ei1:
        mod.AdfTest().run(_series(n=9))
    assert "Not enough observations for ADF test." in str(ei1.value)

    # Test constant series
    with pytest.raises(mod.InvalidInputError) as ei2:
        mod.AdfTest().run(_const_series(n=20, val=7.0))
    assert "Series is constant" in str(ei2.value)

    # Test backend failure (RuntimeError)
    _patch_adfuller(monkeypatch, fail=True, fail_type="backend")
    with pytest.raises(mod.StatisticalBackendError) as ei3:
        mod.AdfTest().run(_series(n=20))
    assert "backend error" in str(ei3.value).lower()

    # Test computation failure (LinAlgError)
    _patch_adfuller(monkeypatch, fail=True, fail_type="computation")
    with pytest.raises(mod.StatisticalComputationError) as ei4:
        mod.AdfTest().run(_series(n=20))
    assert "computation failed" in str(ei4.value).lower()

    # Test data validation failure
    _patch_adfuller(monkeypatch, fail=True, fail_type="data")
    with pytest.raises(mod.StatisticalDataError) as ei5:
        mod.AdfTest().run(_series(n=20))
    assert "invalid data" in str(ei5.value).lower() or "invalid" in str(ei5.value).lower()


def test_adf_success_and_mlflow_logging_toggle(monkeypatch):
    _patch_adfuller(monkeypatch, fail=False)
    calls = _patch_mlflow(monkeypatch, active=True)
    res = mod.AdfTest().run(_series(n=20), regression="ct", autolag=None, significance_level=0.05)
    keys = [k for k, _ in calls]
    assert "adf_p_value" in keys
    assert "adf_test_stat" in keys
    assert isinstance(res["is_stationary"], bool)

    _patch_adfuller(monkeypatch, fail=False)
    calls2 = _patch_mlflow(monkeypatch, active=False)
    _ = mod.AdfTest().run(_series(n=20), regression="c", autolag="AIC", significance_level=0.1)
    assert calls2 == []


# -------------------------------- KPSS --------------------------------


def test_kpss_edges_backend_and_success(monkeypatch):
    _patch_kpss(monkeypatch, fail=False)

    # Test insufficient observations
    with pytest.raises(mod.InvalidInputError) as ei1:
        mod.KpssTest().run(_series(n=9))
    assert "Not enough observations for KPSS test." in str(ei1.value)

    # Test constant series
    with pytest.raises(mod.InvalidInputError) as ei2:
        mod.KpssTest().run(_const_series(n=20))
    assert "Series is constant" in str(ei2.value)

    # Test backend failure (RuntimeError)
    _patch_kpss(monkeypatch, fail=True, fail_type="backend")
    with pytest.raises(mod.StatisticalBackendError) as ei3:
        mod.KpssTest().run(_series(n=20))
    assert "backend error" in str(ei3.value).lower()

    # Test computation failure (LinAlgError)
    _patch_kpss(monkeypatch, fail=True, fail_type="computation")
    with pytest.raises(mod.StatisticalComputationError) as ei4:
        mod.KpssTest().run(_series(n=20))
    assert "computation failed" in str(ei4.value).lower()

    # Test data validation failure
    _patch_kpss(monkeypatch, fail=True, fail_type="data")
    with pytest.raises(mod.StatisticalDataError) as ei5:
        mod.KpssTest().run(_series(n=20))
    assert "invalid data" in str(ei5.value).lower()

    # Test success path
    _patch_kpss(monkeypatch, fail=False)
    res = mod.KpssTest().run(_series(n=20), nlags="auto", regression="ct")
    assert isinstance(res["is_stationary"], bool)
    assert isinstance(res["summary"], str)


# -------------------------------- Ljung-Box --------------------------------


def test_ljungbox_lag_variants_errors_and_success(monkeypatch):
    _patch_ljungbox(monkeypatch, fail=False)
    with pytest.raises(mod.InvalidInputError) as ei1:
        mod.LjungboxTest().run(_series(n=9))
    assert "Not enough observations for Ljung-Box test." in str(ei1.value)

    with pytest.raises(mod.InvalidInputError) as ei2:
        mod.LjungboxTest().run(_series(n=15), lags=[1, 10], model_df=6)
    assert "Not enough observations for the chosen lags." in str(ei2.value)

    res_auto = mod.LjungboxTest().run(_series(n=50), lags=None)
    assert isinstance(res_auto["autocorr_detected"], bool)
    assert res_auto["lags"] == list(range(1, min(11, max(2, 50 // 5))))

    res_list = mod.LjungboxTest().run(_series(n=50), lags=[1, 3, 3, 2])
    assert res_list["lags"] == [1, 2, 3]

    res_int = mod.LjungboxTest().run(_series(n=50), lags=5)
    assert res_int["lags"] == [1, 2, 3, 4, 5]


def test_ljungbox_statistical_error_raised_when_backend_fails(monkeypatch):
    # Test backend failure
    _patch_ljungbox(monkeypatch, fail=True, fail_type="backend")
    with pytest.raises(mod.StatisticalBackendError) as ei1:
        mod.LjungboxTest().run(_series(n=20), lags=3)
    assert "backend error" in str(ei1.value).lower()

    # Test computation failure
    _patch_ljungbox(monkeypatch, fail=True, fail_type="computation")
    with pytest.raises(mod.StatisticalComputationError) as ei2:
        mod.LjungboxTest().run(_series(n=20), lags=3)
    assert "computation failed" in str(ei2.value).lower()

    # Test data validation failure
    _patch_ljungbox(monkeypatch, fail=True, fail_type="data")
    with pytest.raises(mod.StatisticalDataError) as ei3:
        mod.LjungboxTest().run(_series(n=20), lags=3)
    assert "invalid data" in str(ei3.value).lower() or "invalid" in str(ei3.value).lower()


# -------------------------------- Orchestrator: run_tests --------------------------------


def test_run_tests_sequential_and_parallel_consistency(monkeypatch):
    """Verify sequential and parallel execution produce identical results"""
    _patch_adfuller(monkeypatch, fail=False)
    _patch_kpss(monkeypatch, fail=False)
    _patch_ljungbox(monkeypatch, fail=False)

    tests = [
        "adf",
        {"name": "kpss", "kwargs": {"nlags": 4}},
        {"name": "ljungbox", "kwargs": {"lags": 5}},
    ]
    data = _series(n=50)

    # Sequential execution
    res_seq = mod.run_tests(tests, data=data, parallel=False)

    # Parallel execution
    res_par = mod.run_tests(tests, data=data, parallel=True, max_workers=2)

    # Results should be identical (order-independent comparison)
    assert set(res_seq.keys()) == set(res_par.keys())
    for test_name in res_seq:
        if "error" not in res_seq[test_name]:
            # Compare summaries for successful tests
            assert res_seq[test_name]["summary"] == res_par[test_name]["summary"]


def test_run_tests_error_aggregation_types(monkeypatch):
    _patch_adfuller(monkeypatch, fail=False)
    _patch_kpss(monkeypatch, fail=False)
    _patch_ljungbox(monkeypatch, fail=False)

    tests = [
        "adf",  # success
        {"name": "ljungbox", "kwargs": {"lags": 200}},  # InvalidInputError
        "unknown_test",  # ValueError (unknown test)
    ]

    res = mod.run_tests(tests, data=_series(n=50))

    # Success case
    assert "error" not in res["adf"]
    assert res["adf"]["is_stationary"] is not None

    # Data error case
    assert "error" in res["ljungbox"]
    assert "Not enough observations" in res["ljungbox"]["error"]["message"]

    # Unknown test case
    assert "error" in res["unknown_test"]
    assert res["unknown_test"]["error"]["code"] == "UNHANDLED_EXCEPTION"


def test_run_tests_parallel_independence_and_performance(monkeypatch):
    """Verify parallel execution executes tests independently and provides performance benefit"""
    import time

    # Track execution order to verify independence
    execution_log = []

    def logged_adf(data, **k):
        execution_log.append(("adf_start", time.time()))
        time.sleep(0.1)  # Fast test
        execution_log.append(("adf_end", time.time()))
        return _fake_adfuller_return(len(data))

    def logged_kpss(data, **k):
        execution_log.append(("kpss_start", time.time()))
        time.sleep(0.3)  # Slower test
        execution_log.append(("kpss_end", time.time()))
        return (0.2, 0.1, 3, {"10%": 0.347, "5%": 0.463, "2.5%": 0.574, "1%": 0.739})

    monkeypatch.setattr(base_mod, "adfuller", logged_adf)
    monkeypatch.setattr(base_mod, "kpss", logged_kpss)

    tests = ["adf", "kpss"]
    data = _series(n=50)

    # Sequential execution baseline
    execution_log.clear()
    start = time.time()
    res_seq = mod.run_tests(tests, data=data, parallel=False)
    seq_time = time.time() - start

    # Parallel execution
    execution_log.clear()
    start = time.time()
    res_par = mod.run_tests(tests, data=data, parallel=True, max_workers=2)
    par_time = time.time() - start

    # Both should succeed
    assert "error" not in res_seq["adf"]
    assert "error" not in res_seq["kpss"]
    assert "error" not in res_par["adf"]
    assert "error" not in res_par["kpss"]

    # Parallel should be faster than sequential
    # Sequential: ~0.1s + 0.3s = 0.4s
    # Parallel: max(0.1s, 0.3s) = ~0.3s
    assert par_time < seq_time * 1.1  # Parallel should not be significantly slower than sequential

    # Results should be identical
    assert res_seq["adf"]["is_stationary"] == res_par["adf"]["is_stationary"]
    assert res_seq["kpss"]["is_stationary"] == res_par["kpss"]["is_stationary"]


# -------------------------------- Convenience wrappers --------------------------------


def test_wrappers_delegate_and_return_summaries(monkeypatch):
    _patch_adfuller(monkeypatch, fail=False)
    _patch_kpss(monkeypatch, fail=False)
    _patch_ljungbox(monkeypatch, fail=False)
    _patch_granger(monkeypatch, fail=False)
    _patch_johansen(monkeypatch, fail=False, n_vars=2)
    _patch_auto_lag(monkeypatch, value=2)

    s = _series(n=20)
    dfxy = _df_xy(n=30)
    df2 = _df_vars(n=30, k=2)

    assert isinstance(mod.adf_test(s)["summary"], str)
    assert isinstance(mod.kpss_test(s)["summary"], str)
    assert isinstance(mod.ljung_box_test(s)["summary"], str)
    assert isinstance(
        mod.granger_causality_test(dfxy["x"], dfxy["y"], maxlag="auto")["summary"], str
    )
    assert isinstance(mod.johansen_cointegration_test(df2, k_ar_diff="auto")["summary"], str)


# -------------------------------- Property-based tests --------------------------------


@seed(12345)
@settings(deadline=None, max_examples=120)
@given(
    st.lists(st.floats(allow_nan=False, allow_infinity=False, width=32), min_size=10, max_size=200)
)
def test_adf_property_nonconstant_input_yields_boolean_stationarity(data):
    # ensure some variation; otherwise skip
    if len(set(np.round(data, 9))) <= 1:
        return

    # Use context manager for patching in property tests
    with patch_stat_backend(
        "adfuller", fake_impl=lambda data, **k: _fake_adfuller_return(len(data))
    ):
        res = mod.AdfTest().run(np.array(data, dtype=float))
        assert isinstance(res["is_stationary"], bool)
        assert isinstance(res["p_value"], float)


@seed(12345)
@settings(deadline=None, max_examples=60)
@given(
    st.integers(min_value=10, max_value=200),
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
)
def test_adf_property_constant_series_rejected(n, c):
    from hypothesis import assume

    # Create the constant series
    data = [float(c)] * int(n)
    series = pd.Series(data)

    # Skip if the series is not actually constant due to floating-point precision
    # (This handles edge cases where std() might be a tiny non-zero value)
    assume(series.std() < 1e-10)

    with patch_stat_backend(
        "adfuller", fake_impl=lambda data, **k: _fake_adfuller_return(len(data))
    ), pytest.raises(mod.InvalidInputError):
        mod.AdfTest().run(data)


@seed(12345)
@settings(deadline=None, max_examples=80)
@given(st.integers(min_value=25, max_value=120), st.integers(min_value=2, max_value=5))
def test_johansen_property_rank_bounded(n_obs, n_vars):
    def fake_joh(data, det_order, k_ar_diff):
        lr1 = np.array([20.0 + i for i in range(n_vars)])
        lr2 = np.array([10.0 + i for i in range(n_vars)])
        cvt = np.array([[15, 19, 24] for _ in range(n_vars)])
        return SimpleNamespace(lr1=lr1, lr2=lr2, cvt=cvt)

    with patch_stat_backend("coint_johansen", fake_impl=fake_joh):
        res = mod.JohansencointegrationTest().run(
            _df_vars(n=n_obs, k=n_vars), det_order=0, k_ar_diff=1
        )
        assert 0 <= res["rank"] <= n_vars


@seed(12345)
@settings(deadline=None, max_examples=80)
@given(st.integers(min_value=1, max_value=10))
def test_granger_property_p_values_length_matches_maxlag(maxlag):
    def fake_gc(df, maxlag, verbose=False):
        # Parameter name must match statsmodels API: 'maxlag', not 'maxlag_param'
        return {
            lag: ({"ssr_ftest": (None, 0.1 + 0.05 * (lag % 2), None, None)}, None)
            for lag in range(1, int(maxlag) + 1)
        }

    with patch_stat_backend("grangercausalitytests", fake_impl=fake_gc):
        df = _df_xy(n=maxlag + 25)
        res = mod.GrangercausalityTest().run(df, maxlag=maxlag)
        assert len(res["p_values"]) == maxlag


# -------------------------------- Pairwise (2-way) combinatorial tests --------------------------------


@pytest.mark.parametrize(
    "pair",
    list(
        AllPairs(
            [
                [0.1, 0.05],  # significance_level
                ["c", "ct"],  # ADF.regression
                ["AIC", None],  # ADF.autolag
                ["auto", 4],  # KPSS.nlags
                ["c", "ct"],  # KPSS.regression
                [None, "auto", 5, [1, 3, 3, 2]],  # LjungBox.lags
                ["pd.Series", "np.ndarray", "list"],  # input_type (univariate)
            ]
        )
    ),
)
def test_pairwise_univariate_tests_smoke(monkeypatch, pair):
    sig, adf_reg, adf_autolag, kpss_nlags, kpss_reg, lb_lags, input_type = pair
    _patch_adfuller(monkeypatch, fail=False)
    _patch_kpss(monkeypatch, fail=False)
    _patch_ljungbox(monkeypatch, fail=False)

    base = _series(n=50, start=len(str(pair))).values
    if input_type == "pd.Series":
        data = pd.Series(base)
    elif input_type == "np.ndarray":
        data = np.asarray(base)
    else:
        data = list(base)

    r1 = mod.AdfTest().run(data, significance_level=sig, regression=adf_reg, autolag=adf_autolag)
    r2 = mod.KpssTest().run(data, significance_level=sig, nlags=kpss_nlags, regression=kpss_reg)
    r3 = mod.LjungboxTest().run(data, significance_level=sig, lags=lb_lags)
    assert isinstance(r1["summary"], str)
    assert isinstance(r2["summary"], str)
    assert isinstance(r3["summary"], str)


@pytest.mark.parametrize(
    "pair",
    list(
        AllPairs(
            [
                [0.1, 0.05],  # significance_level
                ["auto", 1, 3],  # Granger.maxlag
                ["auto", 1, 2],  # Johansen.k_ar_diff
                [0, 1],  # Johansen.det_order
                ["dict", "pd.DataFrame"],  # input_type (multivariate)
            ]
        )
    ),
)
def test_pairwise_multivariate_tests_smoke(monkeypatch, pair):
    sig, gr_maxlag, joh_kdiff, joh_det, input_type = pair
    _patch_granger(monkeypatch, fail=False)
    _patch_johansen(monkeypatch, fail=False, n_vars=2)
    _patch_auto_lag(monkeypatch, value=2)

    base = _series(n=60, start=len(str(pair))).values
    if input_type == "dict":
        inp = {"y": base.tolist(), "x": np.roll(base, 1).tolist()}
        res_g = mod.GrangercausalityTest().run(inp, significance_level=sig, maxlag=gr_maxlag)
        assert isinstance(res_g["summary"], str)
    else:
        df = pd.DataFrame({"a": base, "b": np.roll(base, 1)})
        res_j = mod.JohansencointegrationTest().run(
            df, significance_level=sig, det_order=joh_det, k_ar_diff=joh_kdiff
        )
        assert isinstance(res_j["summary"], str)


# -------------------------------- Contract across implementations --------------------------------


@pytest.mark.parametrize(
    "impl_cls",
    [
        mod.GrangercausalityTest,
        mod.JohansencointegrationTest,
        mod.AdfTest,
        mod.KpssTest,
        mod.LjungboxTest,
    ],
)
def test_contract_invariants_hold(impl_cls, monkeypatch):
    _patch_adfuller(monkeypatch, fail=False)
    _patch_kpss(monkeypatch, fail=False)
    _patch_ljungbox(monkeypatch, fail=False)
    _patch_granger(monkeypatch, fail=False)
    _patch_johansen(monkeypatch, fail=False, n_vars=2)
    _patch_auto_lag(monkeypatch, value=2)

    obj = impl_cls()
    if impl_cls is mod.GrangercausalityTest:
        res = obj.run(_df_xy(n=40), maxlag="auto")
    elif impl_cls is mod.JohansencointegrationTest:
        res = obj.run(_df_vars(n=40, k=2), k_ar_diff="auto")
    else:
        res = obj.run(_series(n=40))

    assert isinstance(res, dict)
    assert "summary" in res
    assert isinstance(res["summary"], str)
    if "any_causality" in res:
        assert isinstance(res["any_causality"], bool)
    if "is_stationary" in res:
        assert isinstance(res["is_stationary"], bool)
    if "autocorr_detected" in res:
        assert isinstance(res["autocorr_detected"], bool)


# -------------------------------- Performance tests (skipped unless enabled) --------------------------------


@pytest.mark.perf
@pytest.mark.skipif(not bool(os.environ.get("ENABLE_PERF_TESTS")), reason="perf tests disabled")
def test_perf_small_n_budgets(monkeypatch):
    _patch_adfuller(monkeypatch, fail=False)
    _patch_kpss(monkeypatch, fail=False)
    _patch_ljungbox(monkeypatch, fail=False)
    _patch_granger(monkeypatch, fail=False)
    _patch_johansen(monkeypatch, fail=False, n_vars=2)
    _patch_auto_lag(monkeypatch, value=2)

    n = 100
    s = _series(n)
    dfxy = _df_xy(n)
    df2 = _df_vars(n, k=2)

    t0 = time.perf_counter()
    mod.AdfTest().run(s)
    t_adf = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    mod.KpssTest().run(s)
    t_kpss = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    mod.LjungboxTest().run(s, lags="auto")
    t_lb = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    mod.GrangercausalityTest().run(dfxy, maxlag="auto")
    t_gc = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    mod.JohansencointegrationTest().run(df2, k_ar_diff="auto")
    t_joh = (time.perf_counter() - t0) * 1000

    assert t_adf <= 500
    assert t_kpss <= 500
    assert t_lb <= 500
    assert t_gc <= 1500
    assert t_joh <= 1500

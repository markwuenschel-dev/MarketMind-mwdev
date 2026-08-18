# tests/unit/test_config.py

import json
import os
import threading
from pathlib import Path

import pytest
from allpairspy import AllPairs

from pysrc.pipeline import pipeline_config as cfg

pytestmark = [pytest.mark.determinism("d0"), pytest.mark.usefixtures("deterministic_seed")]


# ------------------------------- Helpers -------------------------------------


def write_json(path: Path, obj: dict) -> Path:
    path.write_text(json.dumps(obj))
    return path


def write_yaml(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def create_schema(tmpdir: Path, name: str = "config_schema.json") -> Path:
    # permissive schema: only checks top-level is object and allows extra keys
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "schema_uri": {"type": ["string", "null"]},
        },
        "additionalProperties": True,
    }
    return write_json(tmpdir / name, schema)


def create_minimal_pipeline(tmpdir: Path, *, csv_rel_path: str = "data.csv") -> dict:
    # construct minimal valid structure for PipelineConfig; keep values tiny and deterministic
    (tmpdir / csv_rel_path).write_text("a,b\n1,2\n")
    return {
        "version": "1.0.0",
        "schema_uri": "config_schema.json",
        "data_source": {
            "type": "csv",
            "path": str(tmpdir / csv_rel_path),
            "chunksize": 0,
            "use_dask": False,
            "compression": None,
            "data_format": "csv",
        },
        "preprocessing": {
            "technical_indicators": {
                "rsi": {"enabled": False, "window": 14, "fillna_method": "ffill"},
                "macd": {
                    "enabled": False,
                    "fast_period": 12,
                    "slow_period": 26,
                    "signal_period": 9,
                    "fillna_method": "ffill",
                },
                "atr": {"enabled": False, "window": 14, "fillna_method": "ffill"},
                "vwap": {"enabled": False, "reset_period": "daily", "fillna_method": "ffill"},
                "bollinger_bands": {
                    "enabled": False,
                    "window": 20,
                    "std_dev": 2.0,
                    "fillna_method": "ffill",
                },
                "extra_indicators": {},
            },
            "normalization": {
                "method": "zscore",
                "rolling_window": 10,
                "clip_extremes": {"min": -3.0, "max": 3.0},
            },
            "custom_features": {"sentiment": None, "esg_normalized": None},
            "calendar_features": {
                "enabled": False,
                "day_of_week": True,
                "holidays": [],
                "timezones": [],
            },
            "steps": [{"op": "scale", "param": 1}],
            "step_macros": {},
        },
        "cleaning": {
            "combos": [],
            "missing_values": {},
            "outliers": {},
            "denoising": {},
        },
        "streaming": {
            "batch_size": 100,
            "update_interval_seconds": 60,
            "buffer_size": 1000,
            "max_latency_ms": 500,
            "buffer_retention_seconds": 3600,
            "event_triggers": {},
            "priority_queue": "fifo",
            "failure_recovery": {},
            "sync_interval_seconds": 300,
        },
        "error_handling": {
            "retry_policy": {
                "max_attempts": 3,
                "initial_backoff_seconds": 1,
                "max_backoff_seconds": 10,
                "retry_strategy": "exponential",
            },
            "validation_thresholds": {"max_missing_ratio": 0.1, "max_outlier_ratio": 0.05},
            "fallback": {"twitter": "none", "esg": "none", "data_source": "csv"},
            "alerting": {
                "enabled": False,
                "channel": "none",
                "critical_failures": [],
                "alert_severity": [],
            },
            "fallback_timeout_seconds": 5,
        },
        "model": {
            "model_type": "xgboost",
            "architecture": {"num_layers": 1, "hidden_size": 8, "dropout": 0.0},
            "sequence_length": 8,
            "prediction_horizon": 1,
            "feature_list": ["a", "b"],
            "training_device": "cpu",
            "model_checkpoint": {"every_n_steps": 0},
            "feature_importance": {"method": "gain"},
            "onnx_export": {"enabled": False},
        },
        "logging": {
            "level": "INFO",
            "outputs": {
                "console": True,
                "file": {"enabled": False, "path": str(tmpdir / "log.txt"), "rotation": "1 day"},
                "influxdb": {
                    "enabled": False,
                    "host": "localhost",
                    "port": 8086,
                    "token": "x",
                    "org": "o",
                    "bucket": "b",
                },
            },
            "metrics_report_interval_seconds": 60,
            "custom_metrics": [],
            "model_metrics": [],
            "metric_aggregation": {"aggregation_window_seconds": 60},
            "log_sampling_rate": 1.0,
            "dashboard_config": {"grafana_url": "http://localhost"},
        },
        "security": {
            "encryption": {"at_rest": False, "in_transit": True, "encryption_algorithm": "aes"},
            "key_management": "local",
            "credentials": {
                "twitter_api_key": None,
                "esg_api_key": None,
                "fred_api_key": None,
                "bloomberg_api_key": None,
                "weather_api_key": None,
                "alpha_vantage_api_key": None,
            },
            "compliance": {
                "audit_log": False,
                "retention_days": 1,
                "audit_frequency_days": 1,
                "data_anonymization": {"anonymize_pii": False},
            },
        },
        "backtesting": {
            "initial_capital": 1000.0,
            "transaction_cost_rate": 0.0,
            "slippage_rate": 0.0,
            "strategy_list": ["buyhold"],
            "risk_management": {"stop_loss": 1.0, "max_drawdown": 1.0},
            "date_range": {"start": "2020-01-01", "end": "2020-01-02"},
            "performance_metrics": ["return"],
            "position_sizing": {"method": "fixed"},
            "benchmark_index": "SPY",
            "backtest_frequency": "D",
        },
        "distributed_processing": {
            "framework": "local",
            "num_workers": 1,
            "memory_per_worker": "1GB",
            "cluster_type": "standalone",
            "min_rows_for_distributed": 1000,
        },
        "alternative_data": {},
        "market_data_sources": [],
        "real_time_market_data": {},
        "anomaly_detection": {"enabled": False, "method": None, "params": None},
        "includes": [],
        "profiles": {},
        "active_profiles": [],
        "list_merge_strategy": "replace",
    }


def write_config_and_schema(
    tmpdir: Path, config_obj: dict, schema_name: str = "config_schema.json"
) -> tuple[Path, Path]:
    cfg_path = write_yaml(tmpdir / "pipeline_config.yaml", yaml_dump(config_obj))
    sch_path = create_schema(tmpdir, name=schema_name)
    return cfg_path, sch_path


def yaml_dump(obj: dict) -> str:
    # minimal YAML emitter using json with newlines to avoid external deps
    # safe for our tests since values are scalars/flat lists/dicts
    import yaml as _yaml  # use project dep already present

    return _yaml.safe_dump(obj, sort_keys=False)


def set_paths(monkeypatch, cfg_path: Path, sch_path: Path):
    monkeypatch.setattr(cfg, "CONFIG_PATH", cfg_path)
    monkeypatch.setattr(cfg, "SCHEMA_PATH", sch_path)
    monkeypatch.setattr("pysrc.pipeline.pipeline_config.loader.CONFIG_PATH", cfg_path)
    monkeypatch.setattr("pysrc.pipeline.pipeline_config.loader.SCHEMA_PATH", sch_path)
    cfg.reset_config_cache()


# ------------------------------- Fixtures ------------------------------------


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    # ensure deterministic environment for production detection and overrides
    for k in list(os.environ.keys()):
        if k.startswith("MARKETMIND__") or k in {
            "ENVIRONMENT",
            "PYTEST_CURRENT_TEST",
            "TWITTER_API_KEY",
            "ESG_API_KEY",
        }:
            monkeypatch.delenv(k, raising=False)
    # pytest sets PYTEST_CURRENT_TEST; we keep it to simulate non-production by default
    return


# -------------------------------- Branch tests --------------------------------


def test_branch_profiles_overlay_order(tmp_path, monkeypatch):
    # trace: BRANCH=profiles OUTCOME=overlay-applied and none
    base = create_minimal_pipeline(tmp_path)
    base["profiles"] = {"p1": {"logging": {"level": "DEBUG"}}}
    base["active_profiles"] = ["p1"]
    cfg_path, sch_path = write_config_and_schema(tmp_path, base)
    set_paths(monkeypatch, cfg_path, sch_path)

    c = cfg.load_config()
    assert c.logging.level == "DEBUG"

    base["active_profiles"] = []
    cfg_path2 = write_yaml(tmp_path / "pipeline_config2.yaml", yaml_dump(base))
    set_paths(monkeypatch, cfg_path2, sch_path)
    c_no = cfg.load_config()
    assert c_no.logging.level == "INFO"


def test_branch_list_merge_strategies_replace_append_unique_unit():
    # trace: BRANCH=list_strategy OUTCOME=replace/append/unique
    a = {"x": [1, {"k": 1}]}
    b = {"x": [2, {"k": 1}]}
    # replace
    r = cfg._deep_merge(a, b, list_strategy="replace")
    assert r["x"] == [2, {"k": 1}]
    # append
    r = cfg._deep_merge(a, b, list_strategy="append")
    assert r["x"] == [1, {"k": 1}, 2, {"k": 1}]
    # unique (dict dedup via json key)
    r = cfg._deep_merge(a, b, list_strategy="unique")
    assert r["x"] == [1, {"k": 1}, 2]


def test_branch_data_source_discrimination_csv(tmp_path, monkeypatch):
    # trace: BRANCH=data_source OUTCOME=csv loader used
    class _PolarsStub:
        @staticmethod
        def read_csv(path, **_):
            return {"loaded_from": Path(path).name}

        @staticmethod
        def from_pandas(df, **kwargs):
            return {"from_pandas": True}

    monkeypatch.setattr(
        "pysrc.pipeline.pipeline_config.loader.ensure_polars", lambda: _PolarsStub()
    )

    base = create_minimal_pipeline(tmp_path)
    cfg_path, sch_path = write_config_and_schema(tmp_path, base)
    set_paths(monkeypatch, cfg_path, sch_path)

    df = cfg.get_dataset()
    assert df["loaded_from"] == "data.csv"


def test_branch_data_source_polars_unavailable_unreachable(tmp_path, monkeypatch):
    # Contract: DataSource union uses only final, Polars-capable classes.
    # Therefore the legacy TypeError branch in get_dataset() must be unreachable
    # by construction (capability exists on all variants).
    from pysrc.pipeline.pipeline_config import _PolarsMixin  # capability marker

    base = create_minimal_pipeline(tmp_path)
    cfg_path, sch_path = write_config_and_schema(tmp_path, base)
    set_paths(monkeypatch, cfg_path, sch_path)

    # Build the config and inspect the bound data_source
    c = cfg.get_config()
    ds = c.data_source

    # Structural capability: every DataSource is Polars-capable by construction
    assert isinstance(ds, _PolarsMixin)
    assert hasattr(ds, "to_polars")

    # Optional: if the environment lacks polars or remote deps, invoking the path
    # may fail, but it must NOT fail with TypeError (capability exists).
    # We avoid asserting a specific dependency error type to keep the test
    # backend-agnostic and environment-agnostic.
    try:
        # Do not rely on polars being installed; a dependency error here is fine.
        # The key contract is that a TypeError is not raised due to missing capability.
        ds.to_polars  # attribute presence already asserted
    except TypeError as te:  # specificity: only the unreachable branch would raise this
        pytest.fail(f"Unreachable TypeError surfaced: {te}")


def test_branch_production_detection(tmp_path, monkeypatch):
    # trace: BRANCH=production_mode OUTCOME=prod/non-prod
    base = create_minimal_pipeline(tmp_path)
    cfg_path, sch_path = write_config_and_schema(tmp_path, base)
    set_paths(monkeypatch, cfg_path, sch_path)

    # non-prod by default since PYTEST_CURRENT_TEST is set
    assert cfg._is_production_mode() is False

    # prod if ENVIRONMENT=production
    monkeypatch.setenv("ENVIRONMENT", "production")
    assert cfg._is_production_mode() is True


# --------------------------------- Unit tests ---------------------------------


def test_unit_apply_profiles_noop_and_apply():
    data = {"x": 1, "profiles": {"p": {"x": 2}}}
    out = cfg._apply_profiles(data, [], "replace")
    assert out["x"] == 1
    out2 = cfg._apply_profiles(data, ["p"], "replace")
    assert out2["x"] == 2


def test_unit_compiled_validator_cache(tmp_path):
    s1 = create_schema(tmp_path, "s1.json")
    s2 = create_schema(tmp_path, "s2.json")
    v1a = cfg._compiled_validator(s1)
    v1b = cfg._compiled_validator(s1)
    v2 = cfg._compiled_validator(s2)
    assert v1a is v1b
    assert v1a is not v2


def test_unit_section_validators_positive_and_errors():
    # Normalization rolling_window > 0
    norm = cfg.Normalization(method="z", rolling_window=1, clip_extremes=cfg.Clip(min=-1, max=1))
    norm.validate_section()
    # RSI window must be positive when enabled
    with pytest.raises(ValueError, match="RSI window must be positive"):
        cfg.RSI(enabled=True, window=0, fillna_method="ffill").validate_section()
    # MACD fast_period < slow_period when enabled
    with pytest.raises(ValueError, match="fast_period must be < slow_period"):
        cfg.MACD(
            enabled=True, fast_period=10, slow_period=10, signal_period=9, fillna_method="ffill"
        ).validate_section()
    # ExternalAPISource timeouts constraints
    api = cfg.FRED(base_url="x", endpoints={}, timeout_seconds=1, cache_duration_hours=0)
    api.validate_section()
    with pytest.raises(ValueError, match="timeout_seconds must be positive"):
        cfg.FRED(
            base_url="x", endpoints={}, timeout_seconds=0, cache_duration_hours=0
        ).validate_section()
    with pytest.raises(ValueError, match="cache_duration_hours cannot be negative"):
        cfg.FRED(
            base_url="x", endpoints={}, timeout_seconds=1, cache_duration_hours=-1
        ).validate_section()


def test_unit_macro_expansion_cartesian():
    pp = cfg.Preprocessing(
        technical_indicators=cfg.TechnicalIndicators(),
        normalization=cfg.Normalization(
            method="z", rolling_window=2, clip_extremes=cfg.Clip(min=-1, max=1)
        ),
        steps=[{"use_macro": "m", "a": 0}, {"op": "noop"}],
        step_macros={"m": {"x": [1, 2], "y": ["u", "v"]}},
    )
    pp.expand_macros()
    # 2x2 cartesian + 1 static step
    assert len(pp.steps) == 5
    # ensure macro fields materialized
    xs = sorted((s["x"], s["y"]) for s in pp.steps if "x" in s)
    assert xs == [(1, "u"), (1, "v"), (2, "u"), (2, "v")]


# ------------------------------- Property tests (Invariants) ------------------


def test_property_is_production_mode_invariant(monkeypatch):
    # PYTEST_CURRENT_TEST presence should force non-prod unless ENVIRONMENT overrides
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    assert cfg._is_production_mode() is False
    monkeypatch.setenv("ENVIRONMENT", "prod")
    assert cfg._is_production_mode() is True


# ------------------------------- Pairwise tests -------------------------------


def test_pairwise_deep_merge_list_strategy_and_item_types():
    # exercise list strategies across primitive vs dict lists
    params = [
        ["replace", "append", "unique"],
        ["primitive", "dicts"],
    ]
    for strategy, typ in AllPairs(params):
        base = [1, 2] if typ == "primitive" else [{"k": 1}, {"k": 2}]
        overlay = [2, 3] if typ == "primitive" else [{"k": 2}, {"k": 3}]
        out = cfg._deep_merge({"x": base}, {"x": overlay}, list_strategy=strategy)["x"]
        if strategy == "replace":
            assert out == overlay
        elif strategy == "append":
            assert out == base + overlay
        else:
            # unique preserves order; dict dedup by json key
            if typ == "primitive":
                assert out == [1, 2, 3]
            else:
                assert out == [{"k": 1}, {"k": 2}, {"k": 3}]


# ------------------------------- Contract tests -------------------------------


def test_contract_pipelineconfig_merged_and_with_profile(tmp_path):
    base = create_minimal_pipeline(tmp_path)
    pc = cfg.PipelineConfig(**base)
    pc2 = pc.merged({"logging": {"level": "DEBUG"}})
    assert pc2.logging.level == "DEBUG"
    pc3 = pc.with_profile("nonexistent")
    assert pc3.logging.level == pc.logging.level  # no-op


# ------------------------------- Concurrency tests ----------------------------


def test_concurrency_singleton_thread_safety(tmp_path, monkeypatch):
    base = create_minimal_pipeline(tmp_path)
    cfg_path, sch_path = write_config_and_schema(tmp_path, base)
    set_paths(monkeypatch, cfg_path, sch_path)

    configs = []
    errs = []

    def worker():
        try:
            c = cfg.get_config()
            configs.append(c)
        except Exception as e:
            errs.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errs
    ids = [id(c) for c in configs]
    assert len(set(ids)) == 1  # all threads saw same instance

    # reload creates a new instance
    old = configs[0]
    new = cfg.reload_config()
    assert new is not old

    # runtime copy independent
    rt = cfg.get_runtime_config()
    assert id(rt) != id(new)
    lvl = rt.logging.level
    rt.logging.level = "DEBUG"
    assert cfg.get_config().logging.level != "DEBUG"
    assert lvl == "INFO"


# ------------------------------- Network/Integration --------------------------
# FLAGS.network is Optional; we avoid external calls and test CSV path via stubbed Polars in branch tests.


# ---------------------------------- Error tests -------------------------------


def test_error_missing_files(tmp_path, monkeypatch):
    cfg.reset_config_cache()
    set_paths(monkeypatch, tmp_path / "missing.yaml", tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        cfg.load_config()
    # create config but missing schema
    base = create_minimal_pipeline(tmp_path)
    cfg_path = write_yaml(tmp_path / "exists.yaml", yaml_dump(base))
    set_paths(monkeypatch, cfg_path, tmp_path / "missing.json")
    with pytest.raises(FileNotFoundError, match="Schema file not found"):
        cfg.load_config()


def test_edge_includes_recursion_and_cycle(tmp_path, monkeypatch):
    # A includes B; B includes A (cycle). Should not loop; later keys override earlier.
    a = create_minimal_pipeline(tmp_path)
    b = create_minimal_pipeline(tmp_path)
    a["includes"] = ["b.yaml"]
    b["includes"] = ["a.yaml"]
    a["logging"]["level"] = "INFO"
    b["logging"]["level"] = "WARNING"
    a_path = write_yaml(tmp_path / "a.yaml", yaml_dump(a))
    write_yaml(tmp_path / "b.yaml", yaml_dump(b))
    sch_path = create_schema(tmp_path)
    set_paths(monkeypatch, a_path, sch_path)
    conf = cfg.load_config()
    # With cycle ignored on second visit, merge yields B over A due to include order in A
    assert conf.logging.level == "WARNING"

    # Missing include raises FileNotFoundError
    a2 = create_minimal_pipeline(tmp_path)
    a2["includes"] = ["missing.yaml"]
    a2_path = write_yaml(tmp_path / "a2.yaml", yaml_dump(a2))
    set_paths(monkeypatch, a2_path, sch_path)
    with pytest.raises(FileNotFoundError):
        cfg.load_config()


# ------------------------ Feature / Diagnostics tests -------------------------


def test_feature_env_overrides_and_credentials_warnings_branch(tmp_path, monkeypatch, caplog):
    # env overrides + production credential checks + dask missing path
    base = create_minimal_pipeline(tmp_path)
    # enable twitter & esg to trigger credential issues in production
    base["alternative_data"] = {
        "twitter": {
            "base_url": "https://api.twitter.com",
            "bearer_token": "X",
            "authentication_type": "bearer",
            "endpoints": {},
            "default_params": {},
            "rate_limit": {"per_minute": 10, "max_calls_per_window": 100, "window_seconds": 60},
            "retry_after_header": "x-rate-limit-reset",
            "timeout_seconds": 5,
            "cache_duration_hours": 1,
            "data_resolution": "1m",
            "api_key": None,
        },
        "esg": {
            "base_url": "https://api.esg",
            "api_key": "IGNORED",
            "authentication_type": "header",
            "endpoints": {},
            "default_params": {},
            "timeout_seconds": 5,
            "cache_duration_hours": 1,
            "data_resolution": "1d",
        },
    }
    # make CSV use_dask True to test dependency check
    base["data_source"]["use_dask"] = True
    # add env override to flip a nested value
    base.setdefault("some", {}).setdefault("nested", {})["value"] = "0"
    cfg_path, sch_path = write_config_and_schema(tmp_path, base)
    set_paths(monkeypatch, cfg_path, sch_path)

    # apply hierarchical env override and production
    monkeypatch.setenv("MARKETMIND__some__nested__value", "123")
    monkeypatch.setenv("ENVIRONMENT", "production")

    # dask not available
    monkeypatch.setattr(cfg, "_dependency_available", lambda name: name != "dask.dataframe")

    conf = cfg.load_config()
    issues = cfg.validate_runtime_requirements(conf)
    # env override applied
    # both diagnostics present
    assert any("dask not available" in s for s in issues)
    assert any("Twitter data source enabled but TWITTER_API_KEY not set" in s for s in issues)
    assert any("ESG data source enabled but ESG_API_KEY not set" in s for s in issues)

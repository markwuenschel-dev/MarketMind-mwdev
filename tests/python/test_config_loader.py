# tests/python/test_config_loader.py
from __future__ import annotations

import pytest
import yaml

from pysrc.core.errors import ConfigValidationError
from pysrc.pipeline.pipeline_config import load_config
from tests.python.test_config_integration import create_minimal_pipeline, create_schema

pytestmark = [pytest.mark.determinism("d0"), pytest.mark.usefixtures("deterministic_seed")]


@pytest.mark.elegant
def test_load_config_valid(tmp_path):
    cfg_path = tmp_path / "pipeline_config.yaml"
    schema_path = create_schema(tmp_path)
    cfg_path.write_text(yaml.dump(create_minimal_pipeline(tmp_path)), encoding="utf-8")

    loaded = load_config(cfg_path, schema_path=schema_path)

    assert loaded.data_source.type == "csv"


def test_load_config_invalid_yaml(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("version: [unclosed\n", encoding="utf-8")
    schema_path = create_schema(tmp_path)

    with pytest.raises(ConfigValidationError, match="YAML parsing failed"):
        load_config(bad, schema_path=schema_path)


def test_config_missing_required(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    schema_path = create_schema(tmp_path)
    cfg_path.write_text('version: "1.1"\nschema_uri: "config_schema.json"\n', encoding="utf-8")

    with pytest.raises(ConfigValidationError, match="Config structure validation failed"):
        load_config(cfg_path, schema_path=schema_path)


@pytest.mark.parametrize(
    ("env_var", "path"),
    [
        ("INFLUXDB_TOKEN", ("logging", "outputs", "influxdb", "token")),
        ("FRED_API_KEY", ("security", "credentials", "fred_api_key")),
    ],
)
def test_env_resolution(tmp_path, monkeypatch, env_var, path):
    cfg_path = tmp_path / "cfg.yaml"
    schema_path = create_schema(tmp_path)
    cfg_data = create_minimal_pipeline(tmp_path)
    cursor = cfg_data
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = f"${{{env_var}}}"
    cfg_path.write_text(yaml.dump(cfg_data), encoding="utf-8")

    monkeypatch.setenv(env_var, "TEST_TOKEN")
    loaded = load_config(cfg_path, schema_path=schema_path)

    cursor = loaded.model_dump(mode="python")
    for key in path:
        cursor = cursor[key]
    assert cursor == "TEST_TOKEN"


def test_load_config_invalid_yaml_again(tmp_path):
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text("bad: [", encoding="utf-8")
    schema_path = create_schema(tmp_path)

    with pytest.raises(ConfigValidationError, match="YAML parsing failed"):
        load_config(cfg_path, schema_path=schema_path)


def test_required_missing(tmp_path):
    cfg_path = tmp_path / "missing.yaml"
    schema_path = create_schema(tmp_path)
    cfg_path.write_text('version: "1.1"\nschema_uri: "config_schema.json"\n', encoding="utf-8")

    with pytest.raises(ConfigValidationError, match="Config structure validation failed"):
        load_config(cfg_path, schema_path=schema_path)

from __future__ import annotations

import ast
import importlib
import tomllib
from pathlib import Path

import pytest

from pysrc.core.runtime.capabilities import CAPABILITIES

pytestmark = [pytest.mark.determinism("d1"), pytest.mark.usefixtures("deterministic_seed")]

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_pyproject_targets_pysrc_as_live_package_root() -> None:
    pyproject_path = REPO_ROOT / "pyproject.toml"
    pyproject_text = pyproject_path.read_text(encoding="utf-8")
    parsed = tomllib.loads(pyproject_text)

    package_find = parsed["tool"]["setuptools"]["packages"]["find"]
    assert "." in package_find["where"]
    assert "pysrc*" in package_find["include"]
    assert "py*" not in package_find["include"]
    coverage_sources = parsed["tool"]["coverage"]["run"]["source"]
    assert coverage_sources
    assert all(source == "pysrc" or source.startswith("pysrc/") for source in coverage_sources)
    assert not any(source == "py" or source.startswith("py/") for source in coverage_sources)
    pytest_addopts = parsed["tool"]["pytest"]["ini_options"]["addopts"].split()
    assert "--cov=py" not in pytest_addopts
    assert "--cov=pysrc" in pytest_addopts
    assert "mypysrc" not in parsed["tool"]
    assert "tool.mypysrc" not in pyproject_text


@pytest.mark.parametrize(
    "plugin_path",
    [
        REPO_ROOT / "marketmind_cuml_backend" / "marketmind_cuml" / "plugin.py",
        REPO_ROOT / "marketmind_cupy_backend" / "marketmind_cupy" / "plugin.py",
        REPO_ROOT / "marketmind_polars_backend" / "marketmind_polars" / "plugin.py",
        REPO_ROOT / "marketmind_torch_backend" / "marketmind_torch" / "plugin.py",
        REPO_ROOT / "marketmind_xgboost_backend" / "marketmind_xgboost" / "plugin.py",
    ],
)
def test_backend_plugins_do_not_print(plugin_path: Path) -> None:
    tree = ast.parse(plugin_path.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id != "print"


@pytest.mark.parametrize(
    ("module_name", "backend_key", "capability"),
    [
        ("marketmind_cuml_backend.marketmind_cuml.plugin", "cuml", ("ml_engine", "cuml")),
        ("marketmind_cupy_backend.marketmind_cupy.plugin", "cupy", ("array", "cupy")),
        ("marketmind_polars_backend.marketmind_polars.plugin", "polars", ("dataframe", "polars")),
        ("marketmind_torch_backend.marketmind_torch.plugin", "torch", ("tensor", "torch")),
        (
            "marketmind_xgboost_backend.marketmind_xgboost.plugin",
            "xgboost",
            ("classifier", ("xgboost", "XGBClassifier")),
        ),
    ],
)
def test_backend_plugins_register_against_runtime_capabilities(
    module_name: str,
    backend_key: str,
    capability: tuple[str, object],
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = importlib.import_module(module_name)
    capsys.readouterr()

    module.register()

    captured = capsys.readouterr()
    capability_group, capability_value = capability
    assert captured.out == ""
    assert backend_key
    assert capability_value in CAPABILITIES[capability_group]


def test_legacy_runtime_dependency_module_removed() -> None:
    legacy_name = "dependency" + "_manager.py"
    assert not (REPO_ROOT / "pysrc" / "core" / "runtime" / legacy_name).exists()

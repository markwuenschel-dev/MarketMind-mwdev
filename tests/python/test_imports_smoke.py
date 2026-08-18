import importlib
import os
from pathlib import Path

import pytest


def iter_modules():
    root = Path(__file__).resolve().parents[2] / "py"
    for p in root.rglob("*.py"):
        if p.name == "__init__.py":
            continue
        mod = "pysrc." + ".".join(p.relative_to(root).with_suffix("").parts)
        yield mod


@pytest.mark.smoke
def test_import_every_module():
    for mod in iter_modules():
        if any(s in mod for s in os.getenv("MM_SKIP_IMPORT_SUBSTR", "").split(",")):
            continue
        importlib.import_module(mod)

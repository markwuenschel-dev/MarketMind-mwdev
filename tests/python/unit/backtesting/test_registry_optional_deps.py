from __future__ import annotations

import pytest

from pysrc.backtesting.contracts.errors import (
    NotImplementedLaneError,
    OptionalDependencyMissingError,
)
from pysrc.backtesting.contracts.registry import resolve_engine


@pytest.mark.determinism("d1")
def test_optional_backends_import_and_fail_only_at_runtime() -> None:
    with pytest.raises(OptionalDependencyMissingError):
        resolve_engine("jax.scaffold").run(None, None, None)
    with pytest.raises(OptionalDependencyMissingError):
        resolve_engine("backtrader.scaffold").run(None, None, None)
    with pytest.raises(NotImplementedLaneError):
        resolve_engine("event_driven.scaffold").run(None, None, None)

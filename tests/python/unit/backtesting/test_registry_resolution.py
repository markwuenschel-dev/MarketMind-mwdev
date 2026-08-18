from __future__ import annotations

import pytest

from pysrc.backtesting.contracts.errors import UnknownIdError
from pysrc.backtesting.contracts.registry import list_engines, resolve_engine


@pytest.mark.determinism("d1")
def test_registry_lists_default_engine_ids() -> None:
    assert "vectorized.sma" in list_engines()


@pytest.mark.determinism("d1")
def test_unknown_id_error_includes_choices_and_hint() -> None:
    with pytest.raises(UnknownIdError) as exc_info:
        resolve_engine("missing.engine")

    message = str(exc_info.value)
    assert "missing.engine" in message
    assert "vectorized.sma" in message
    assert "Register the engine" in message

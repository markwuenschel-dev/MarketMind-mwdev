"""MLC-0 · Unit tests for :class:`pysrc.registry.signal_abc.SignalABC`.

Covers brief §5 Step 7 acceptance:
- ``signal_embedding`` field exists and defaults to None
- No regressions on existing Signal surfaces — the existing
  :class:`SignalProtocol` in ``signal_catalog`` keeps its structural
  surface
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pysrc.registry import SignalABC, SignalCatalog
from pysrc.registry.signal_catalog import SignalProtocol


class _MinimalSignal(SignalABC):
    def __init__(self) -> None:
        self._slot = -1

    @property
    def slot_index(self) -> int:
        return self._slot

    def __call__(self, data: Any, params: dict[str, Any]) -> Any:
        return [0.0] * int(getattr(data, "height", len(data) if hasattr(data, "__len__") else 0))


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_signal_abc_signal_embedding_defaults_to_none() -> None:
    s = _MinimalSignal()
    assert s.signal_embedding is None


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_signal_abc_accepts_ndarray_embedding() -> None:
    s = _MinimalSignal()
    # Setting is permitted but the MLC-0 contract is None until MLC-1 populates it.
    s.signal_embedding = np.zeros(8, dtype=np.float32)
    assert isinstance(s.signal_embedding, np.ndarray)


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_signal_abc_is_abstract_without_slot_index_and_call() -> None:
    with pytest.raises(TypeError):

        class Incomplete(SignalABC):  # type: ignore[misc]
            pass

        Incomplete()  # type: ignore[abstract]


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_signal_abc_is_structurally_compatible_with_signal_protocol() -> None:
    s = _MinimalSignal()
    assert isinstance(s, SignalProtocol)


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_signal_abc_can_be_registered_in_signal_catalog() -> None:
    s = _MinimalSignal()
    cat = SignalCatalog()
    slot = cat.register(s, spec_hash="mlc0-abc-test", signal_name="mlc0.abc.smoke")
    assert slot == 0
    registered = cat.get_by_slot(slot)
    assert registered is not None
    assert registered.signal_embedding is None

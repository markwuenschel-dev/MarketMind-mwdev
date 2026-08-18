from __future__ import annotations

import pytest

from pysrc.strategies.momentum.spec import build_momentum_params

pytestmark = [pytest.mark.unit, pytest.mark.determinism("d1")]


def test_build_momentum_params_rejects_unknown_variant() -> None:
    with pytest.raises(ValueError, match="Unsupported momentum variant"):
        build_momentum_params("not_a_variant", {})


def test_build_momentum_params_normalizes_vol_and_windows() -> None:
    out = build_momentum_params(
        "xsec",
        {"target_vol": "0.2", "max_leverage": "1.5", "lookback_window": "63"},
    )
    assert out["variant"] == "xsec"
    assert out["target_vol"] == pytest.approx(0.2)
    assert out["max_leverage"] == pytest.approx(1.5)
    assert out["lookback_window"] == 63


def test_build_momentum_params_rejects_non_positive_target_vol() -> None:
    with pytest.raises(ValueError, match="target_vol"):
        build_momentum_params("xsec", {"target_vol": 0.0})

from __future__ import annotations

import pytest

from pysrc.strategies.momentum.validation.production_v1 import PRODUCTION_V1_PROFILE

pytestmark = [pytest.mark.unit, pytest.mark.determinism("d1")]


def test_production_v1_profile_is_locked() -> None:
    profile = PRODUCTION_V1_PROFILE
    assert profile.profile_id == "production_v1"
    assert profile.dsr_p_value_max == pytest.approx(0.05)
    assert profile.min_trl_target_confidence == pytest.approx(0.95)
    assert profile.pbo_max == pytest.approx(0.50)
    assert profile.cpcv.n_splits == 6
    assert profile.cpcv.n_test_paths == 2

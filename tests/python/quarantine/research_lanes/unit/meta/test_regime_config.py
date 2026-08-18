"""BOCPDConfig validation and serialization."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from pysrc.meta.regime_config import BOCPDConfig


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_content_hash_stable_for_equal_configs() -> None:
    a = BOCPDConfig()
    b = BOCPDConfig()
    assert a.content_hash() == b.content_hash()


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_to_dict_roundtrip_keys() -> None:
    c = BOCPDConfig()
    d = c.to_dict()
    assert d["config_version"] == c.config_version
    assert d["hazard_rate"] == c.hazard_rate


@pytest.mark.unit
@pytest.mark.determinism("d2")
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"hazard_rate": 0.0}, "hazard_rate"),
        ({"hazard_rate": 1.0}, "hazard_rate"),
        ({"prior_kappa0": 0.0}, "prior_kappa0"),
        ({"prior_alpha0": 0.0}, "prior_alpha0"),
        ({"prior_beta0": 0.0}, "prior_beta0"),
        ({"max_run_length": 0}, "max_run_length"),
        ({"vol_window": 0}, "vol_window"),
        ({"trend_window": 0}, "trend_window"),
        ({"trend_flat_epsilon": -0.1}, "trend_flat_epsilon"),
        ({"cp_threshold": 0.0}, "cp_threshold"),
        ({"transition_threshold": 0.0}, "transition_threshold"),
        ({"transition_max_rl": -1}, "transition_max_rl"),
        ({"cold_start_burn_in": -1}, "cold_start_burn_in"),
        ({"crisis_vol_score_percentile": 0.0}, "crisis_vol_score_percentile"),
        ({"crisis_vol_score_percentile": 100.0}, "crisis_vol_score_percentile"),
    ],
)
def test_post_init_rejects_invalid(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        BOCPDConfig(**kwargs)


@pytest.mark.unit
@pytest.mark.determinism("d2")
def test_fixture_config_json_validates_against_schema() -> None:
    root = Path(__file__).resolve().parents[4]
    schema = json.loads(
        (root / "schemas" / "rg09_fixture_config.schema.json").read_text(encoding="utf-8")
    )
    cfg_path = root / "docs" / "rg09" / "rg09_bocpd_fixture_config_v1.json"
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    jsonschema.validate(data, schema)
    assert BOCPDConfig(**data).config_version == "rg09_v1.0.2"

"""MLN-02 regime vocabulary and projection invariants."""

from __future__ import annotations

import pytest

from pysrc.core.errors import DataPreconditionError
from pysrc.meta_learning.regime_vocabulary import (
    PROJECTION_RULE_EXTENDED_ABLATION_ID,
    PROJECTION_RULE_LOGIC_ID_DEFAULT,
    PROJECTION_RULE_REFERENCE_BOCPD_ID,
    REGIME_CLASSES,
    project_regime_class,
    project_regime_class_bocpd_reference,
    project_regime_class_extended_ablation,
    projection_rule_display,
    projection_rule_version_id,
    validate_compositional_regime_id,
    validate_meta_task_regime_id,
    validate_regime_class,
    validate_row_count_dict_keys,
)


@pytest.mark.determinism("d0")
def test_validate_regime_class_accepts_five_labels() -> None:
    for label in sorted(REGIME_CLASSES):
        assert validate_regime_class(label) == label


@pytest.mark.determinism("d0")
def test_validate_regime_class_rejects_unknown() -> None:
    with pytest.raises(DataPreconditionError):
        validate_regime_class("Bull")
    with pytest.raises(DataPreconditionError):
        validate_regime_class("")


@pytest.mark.determinism("d0")
def test_default_projection_crisis_requires_severity_gate() -> None:
    assert project_regime_class("hi", "hi", "stable", severity_flag=False) == "high_vol"
    assert project_regime_class("hi", "hi", "cp", severity_flag=False) == "high_vol"
    assert project_regime_class("hi", "hi", "cp", severity_flag=True) == "crisis"


@pytest.mark.determinism("d0")
def test_bocpd_reference_crisis_without_severity_flag() -> None:
    assert project_regime_class_bocpd_reference("hi", "hi", "cp") == "crisis"
    assert project_regime_class_bocpd_reference("hi", "hi", "stable") == "high_vol"


@pytest.mark.determinism("d0")
def test_extended_ablation_is_distinct_from_default() -> None:
    assert project_regime_class_extended_ablation("hi", "hi", "transition") == "crisis"
    assert project_regime_class("hi", "hi", "transition", severity_flag=False) == "high_vol"


@pytest.mark.determinism("d0")
def test_projection_rule_ids_stable() -> None:
    assert PROJECTION_RULE_LOGIC_ID_DEFAULT.startswith("mln02.")
    assert "reference" in PROJECTION_RULE_REFERENCE_BOCPD_ID
    assert "diag" in PROJECTION_RULE_EXTENDED_ABLATION_ID.lower()
    assert projection_rule_display(severity_percentile=90.0) == "vol_hi AND severity_flag (p90)"
    assert (
        projection_rule_version_id(severity_percentile=90.0) == "mln02.v1.level2_severity_gate#p=90"
    )


@pytest.mark.determinism("d0")
def test_validate_compositional_regime_id() -> None:
    rid = "trend_hi__vol_hi__bocpd_stable"
    assert validate_compositional_regime_id(rid) == rid
    with pytest.raises(DataPreconditionError):
        validate_compositional_regime_id("not_compositional")


@pytest.mark.determinism("d0")
def test_validate_meta_task_regime_id_non_empty() -> None:
    assert validate_meta_task_regime_id("rg09.bundle") == "rg09.bundle"
    with pytest.raises(DataPreconditionError):
        validate_meta_task_regime_id("   ")


@pytest.mark.determinism("d0")
def test_validate_row_count_dict_keys() -> None:
    out = validate_row_count_dict_keys({"bull": 3, "crisis": 1})
    assert out["bull"] == 3
    assert out["crisis"] == 1
    assert out["bear"] == 0
    with pytest.raises(DataPreconditionError):
        validate_row_count_dict_keys({"invalid": 1})

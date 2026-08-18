"""MLN-02 canonical regime vocabulary and Level-2 projection (single source of truth).

**Identity (Architecture Vision / Resolution Ledger)**

- ``regime_id`` — Level-1 **primary** compositional task identity (e.g. ``trend_*__vol_*__bocpd_*``).
  It is **not** interchangeable with ``regime_class``.
- ``regime_class`` — Level-2 **derived** 5-class projection for curriculum, supervision, diagnostics,
  and reporting only.

**Fixed 5-class vocabulary (no aliases)**

``bull``, ``bear``, ``sideways``, ``high_vol``, ``crisis``.

**Default Level-2 crisis rule (MLN-02-AMD-01)**

``crisis := vol_hi AND severity_flag``, where ``severity_flag`` is a PIT-safe gate on
``vol_score_raw`` vs an expanding-window percentile of strict past history (default **p90** under
assumption **RG09-V12**, configurable via ``BOCPDConfig.crisis_vol_score_percentile``).
**BOCPD is not part of this rule.**

**BOCPD role**

BOCPD remains a Level-1 compositional dimension inside ``regime_id`` and a **segmentation /
reference** primitive. The function :func:`project_regime_class_bocpd_reference` is the **explicit
II-0A empirical reference condition** (``crisis := vol_hi AND bocpd_cp``). It is **not** governed
default truth for Level-2 ``regime_class``.

**Projection-rule versioning (holdout / GATE-II-01 hook)**

Use :func:`projection_rule_version_id` (and optionally :func:`projection_rule_display`) in fixture
summaries and diagnostics so Anti-Goodhart holdout manifests can bind to a **machine-stable** rule
id independent of human-readable text.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Final, Literal, cast

from pysrc.core.errors import DataPreconditionError

# --- 5-class vocabulary (fixed) ---

RegimeClassLabel = Literal["bull", "bear", "sideways", "high_vol", "crisis"]
TrendBucket = Literal["hi", "lo", "flat"]
VolBucket = Literal["hi", "med", "lo"]
BocpdState = Literal["stable", "transition", "cp"]

REGIME_CLASS_BULL: RegimeClassLabel = "bull"
REGIME_CLASS_BEAR: RegimeClassLabel = "bear"
REGIME_CLASS_SIDEWAYS: RegimeClassLabel = "sideways"
REGIME_CLASS_HIGH_VOL: RegimeClassLabel = "high_vol"
REGIME_CLASS_CRISIS: RegimeClassLabel = "crisis"

REGIME_CLASSES: Final[frozenset[str]] = frozenset(
    {
        REGIME_CLASS_BULL,
        REGIME_CLASS_BEAR,
        REGIME_CLASS_SIDEWAYS,
        REGIME_CLASS_HIGH_VOL,
        REGIME_CLASS_CRISIS,
    }
)

REGIME_CLASS_ORDER: Final[tuple[str, ...]] = (
    REGIME_CLASS_BULL,
    REGIME_CLASS_BEAR,
    REGIME_CLASS_SIDEWAYS,
    REGIME_CLASS_HIGH_VOL,
    REGIME_CLASS_CRISIS,
)

VOL_RISING_REGIME_CLASSES: Final[frozenset[str]] = frozenset(
    {REGIME_CLASS_CRISIS, REGIME_CLASS_HIGH_VOL}
)
TREND_STABLE_REGIME_CLASSES: Final[frozenset[str]] = frozenset(
    {REGIME_CLASS_BULL, REGIME_CLASS_BEAR, REGIME_CLASS_SIDEWAYS}
)

# --- Projection rule ids (versioning / diagnostics; not secret material) ---

PROJECTION_RULE_LOGIC_ID_DEFAULT: Final[str] = "mln02.v1.level2_severity_gate"
PROJECTION_RULE_REFERENCE_BOCPD_ID: Final[str] = "mln02.reference.ii0a_bocpd_cp_vol_hi"
PROJECTION_RULE_EXTENDED_ABLATION_ID: Final[str] = "rg09.diag001.vol_hi_bocpd_transition_or_cp"

DEFAULT_CRISIS_VOL_SCORE_PERCENTILE: Final[float] = 90.0

_COMPOSITIONAL_REGIME_ID_RE = re.compile(
    r"^trend_(hi|lo|flat)__vol_(hi|med|lo)__bocpd_(stable|transition|cp)$"
)


def projection_rule_display(*, severity_percentile: float) -> str:
    """Human-readable projection line for fixtures and reports."""
    return f"vol_hi AND severity_flag (p{severity_percentile:g})"


def projection_rule_version_id(*, severity_percentile: float) -> str:
    """Stable machine id for manifests, holdout binding, and GATE-II-01 re-eval hooks."""
    return f"{PROJECTION_RULE_LOGIC_ID_DEFAULT}#p={severity_percentile:g}"


def validate_regime_class(value: str) -> RegimeClassLabel:
    """Reject unknown labels; return the canonical 5-class string."""
    label = str(value).strip()
    if label not in REGIME_CLASSES:
        raise DataPreconditionError(
            "regime_class must be one of the five canonical Level-2 labels (MLN-02)",
            details={"regime_class": value, "allowed": sorted(REGIME_CLASSES)},
        )
    return cast(RegimeClassLabel, label)


def is_valid_compositional_regime_id(regime_id: str) -> bool:
    """True iff ``regime_id`` matches Level-1 compositional grammar (RG-09)."""
    return bool(_COMPOSITIONAL_REGIME_ID_RE.match(regime_id))


def validate_compositional_regime_id(regime_id: str) -> str:
    """Require compositional Level-1 identity; raises if malformed."""
    rid = str(regime_id).strip()
    if not is_valid_compositional_regime_id(rid):
        raise DataPreconditionError(
            "regime_id does not match compositional regime grammar (Level-1)",
            details={"regime_id": regime_id},
        )
    return rid


def validate_meta_task_regime_id(regime_id: str) -> str:
    """
    Primary task identity token for :class:`MetaTask` — non-empty, bounded string.

    Does **not** require compositional grammar so tests and bundle-level synthetic ids remain valid;
    use :func:`validate_compositional_regime_id` for Level-1 RG-09 replay rows.
    """
    rid = str(regime_id).strip()
    if not rid:
        raise DataPreconditionError(
            "regime_id must be non-empty (primary task identity)", details={}
        )
    if len(rid) > 512:
        raise DataPreconditionError("regime_id exceeds maximum length", details={"len": len(rid)})
    return rid


def validate_row_count_dict_keys(counts: Mapping[str, int]) -> dict[str, int]:
    """Normalize counts to the canonical 5 keys; reject unknown class labels."""
    out: dict[str, int] = dict.fromkeys(REGIME_CLASS_ORDER, 0)
    for k, v in counts.items():
        ks = validate_regime_class(str(k))
        out[ks] = int(v)
    return out


def project_regime_class(
    trend: TrendBucket,
    vol: VolBucket,
    bocpd_state: BocpdState,
    *,
    severity_flag: bool,
) -> RegimeClassLabel:
    """
    Governed default Level-2 projection (MLN-02-AMD-01). Evaluation order:
    ``crisis → high_vol → bear → bull → sideways``.

    Crisis requires ``vol == hi`` **and** ``severity_flag``. ``bocpd_state`` is ignored for crisis
    (BOCPD stays in Level-1 ``regime_id`` only).
    """
    _ = bocpd_state
    if vol == "hi" and severity_flag:
        return REGIME_CLASS_CRISIS
    if vol == "hi":
        return REGIME_CLASS_HIGH_VOL
    if trend == "lo":
        return REGIME_CLASS_BEAR
    if trend == "hi":
        return REGIME_CLASS_BULL
    return REGIME_CLASS_SIDEWAYS


def project_regime_class_bocpd_reference(
    trend: TrendBucket,
    vol: VolBucket,
    bocpd_state: BocpdState,
) -> RegimeClassLabel:
    """
    II-0A **reference** projection only: ``crisis := vol_hi AND bocpd_cp``.

    Not governed default truth for ``regime_class``; use for side-by-side diagnostics /
    ``diag_regime_class_bocpd_gated`` / agreement-rate baselines.
    """
    if vol == "hi" and bocpd_state == "cp":
        return REGIME_CLASS_CRISIS
    if vol == "hi":
        return REGIME_CLASS_HIGH_VOL
    if trend == "lo":
        return REGIME_CLASS_BEAR
    if trend == "hi":
        return REGIME_CLASS_BULL
    return REGIME_CLASS_SIDEWAYS


def project_regime_class_extended_ablation(
    trend: TrendBucket,
    vol: VolBucket,
    bocpd_state: BocpdState,
) -> RegimeClassLabel:
    """
    Ablation-only extended crisis (RG09-DIAG-001): ``vol_hi`` and BOCPD in ``{cp, transition}``.

    **Not** canonical ``regime_class``; diagnostics / experiments only.
    """
    if vol == "hi" and bocpd_state in ("cp", "transition"):
        return REGIME_CLASS_CRISIS
    if vol == "hi":
        return REGIME_CLASS_HIGH_VOL
    if trend == "lo":
        return REGIME_CLASS_BEAR
    if trend == "hi":
        return REGIME_CLASS_BULL
    return REGIME_CLASS_SIDEWAYS


__all__ = [
    "BocpdState",
    "DEFAULT_CRISIS_VOL_SCORE_PERCENTILE",
    "PROJECTION_RULE_EXTENDED_ABLATION_ID",
    "PROJECTION_RULE_LOGIC_ID_DEFAULT",
    "PROJECTION_RULE_REFERENCE_BOCPD_ID",
    "REGIME_CLASSES",
    "REGIME_CLASS_ORDER",
    "REGIME_CLASS_BEAR",
    "REGIME_CLASS_BULL",
    "REGIME_CLASS_CRISIS",
    "REGIME_CLASS_HIGH_VOL",
    "REGIME_CLASS_SIDEWAYS",
    "RegimeClassLabel",
    "TREND_STABLE_REGIME_CLASSES",
    "TrendBucket",
    "VOL_RISING_REGIME_CLASSES",
    "VolBucket",
    "is_valid_compositional_regime_id",
    "project_regime_class",
    "project_regime_class_bocpd_reference",
    "project_regime_class_extended_ablation",
    "projection_rule_display",
    "projection_rule_version_id",
    "validate_compositional_regime_id",
    "validate_meta_task_regime_id",
    "validate_regime_class",
    "validate_row_count_dict_keys",
]

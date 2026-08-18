"""Registry package: SignalCatalog, screening taxonomy, and screening report builder.

Phase I-E: Signal ABC with slot_index, ScreeningReportBuilder with REASON_CODE_TO_FAMILY,
and gate_to_screening mapping for bundle screening_report.json emission.
"""

from pysrc.registry.gate_to_screening import gate_result_to_stage_and_code
from pysrc.registry.screening_report import ScreeningReportBuilder
from pysrc.registry.screening_taxonomy import (
    REASON_CODE_TO_FAMILY,
    ReasonCode,
    ReasonFamily,
    ScreeningStage,
    ScreeningStatus,
)
from pysrc.registry.signal_abc import SignalABC
from pysrc.registry.signal_catalog import SignalCatalog

__all__ = [
    "SignalCatalog",
    "SignalABC",
    "ReasonCode",
    "ReasonFamily",
    "REASON_CODE_TO_FAMILY",
    "ScreeningStage",
    "ScreeningStatus",
    "ScreeningReportBuilder",
    "gate_result_to_stage_and_code",
]

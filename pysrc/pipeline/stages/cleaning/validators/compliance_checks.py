# Relocated from pysrc.data.compliance_checks
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import singledispatch
from typing import Any

from pysrc.core.runtime.optional_imports import optional_import

pl = optional_import("polars")
from pysrc.core.errors import DataValidationError
from pysrc.ops.mm_logkit import get_logger
from pysrc.pipeline.stages.cleaning.core.factory import build_cleaning_pipeline

logger = get_logger(__name__)

COMPLIANCE_CHECKS: dict[str, type["ComplianceCheck"]] = {}


def register_compliance_check(check_type: str):
    def decorator(cls: type["ComplianceCheck"]):
        COMPLIANCE_CHECKS[check_type] = cls
        logger.info("Registered compliance check", extra={"type": check_type})
        return cls

    return decorator


class ComplianceCheck(ABC):
    @abstractmethod
    def apply(
        self, df: pl.DataFrame | pl.LazyFrame, sample_size: int | None = None
    ) -> pl.DataFrame | pl.LazyFrame:
        pass


class ComplianceManager:
    def __init__(self, config, max_workers: int = 8):
        self.config = config
        self.checks: list[ComplianceCheck] = []
        self.max_workers = max_workers
        self._register_checks(getattr(config, "compliance_checks", []))

    def _register_checks(self, checks_cfg: list[dict]):
        for check_cfg in checks_cfg:
            check_type = check_cfg["type"]
            if check_type not in COMPLIANCE_CHECKS:
                raise ValueError(f"Unknown compliance check type: {check_type}")
            check = COMPLIANCE_CHECKS[check_type](config=check_cfg)
            self.checks.append(check)

    def add_check(self, check: ComplianceCheck):
        self.checks.append(check)
        logger.info("Dynamically added compliance check", extra={"type": check.__class__.__name__})

    @singledispatch
    def enforce(self, data: Any, eager: bool = False, sample_size: int | None = None) -> Any:
        raise NotImplementedError(f"Unsupported data type: {type(data)}")

    @enforce.register
    def _(
        self, df: pl.LazyFrame, eager: bool = False, sample_size: int | None = None
    ) -> pl.LazyFrame:
        for check in self.checks:
            df = check.apply(df, sample_size)
        if eager:
            return df.collect()
        return df

    @enforce.register
    def _(
        self, df: pl.DataFrame, eager: bool = False, sample_size: int | None = None
    ) -> pl.DataFrame:
        for check in self.checks:
            df = check.apply(df, sample_size)
        return df

    @enforce.register
    def _(self, dfs: dict, eager: bool = False, sample_size: int | None = None) -> dict:
        results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.enforce, df, eager, sample_size): key
                for key, df in dfs.items()
            }
            for future in as_completed(futures):
                key = futures[future]
                try:
                    results[key] = future.result()
                except Exception as e:
                    logger.error(
                        "Compliance check failed for key", extra={"key": key, "error": str(e)}
                    )
                    raise
        return results


@register_compliance_check("Schema")
class SchemaCompliance(ComplianceCheck):
    def __init__(self, config: dict | None = None):
        config = config or {}
        self.required_cols = config.get(
            "required_columns", ["timestamp", "open", "high", "low", "close", "volume"]
        )
        self.required_dtypes = dict.fromkeys(self.required_cols[1:], pl.Float64)

    def apply(
        self, df: pl.DataFrame | pl.LazyFrame, sample_size: int | None = None
    ) -> pl.DataFrame | pl.LazyFrame:
        schema = df.schema
        missing = set(self.required_cols) - set(schema.keys())
        if missing:
            raise DataValidationError(f"Missing required columns: {missing}")
        for col, dtype in self.required_dtypes.items():
            if schema.get(col) != dtype:
                raise DataValidationError(
                    f"Invalid dtype for {col}: expected {dtype}, got {schema.get(col)}"
                )
        return df


@register_compliance_check("GDPR")
class GDPRCompliance(ComplianceCheck):
    def __init__(self, config: dict | None = None):
        config = config or {}
        self.pii_cols = config.get("pii_columns", [])

    def apply(
        self, df: pl.DataFrame | pl.LazyFrame, sample_size: int | None = None
    ) -> pl.DataFrame | pl.LazyFrame:
        present_pii = set(self.pii_cols) & set(df.columns)
        if present_pii:
            raise DataValidationError(f"GDPR violation: PII columns present {present_pii}")
        return df


@register_compliance_check("Drift")
class DriftCompliance(ComplianceCheck):
    def __init__(self, config: dict | None = None):
        config = config or {}
        self.reference = config.get("reference_data")
        self.detector = build_cleaning_pipeline(
            {
                "steps": [
                    {
                        "step_id": "validate.drift",
                        "step_type": "validate.drift",
                        "version": "1",
                        "params": {
                            "enabled": True,
                            "threshold": config.get("threshold", 0.05),
                            "columns": tuple(config.get("columns", ())),
                            "strict": bool(config.get("strict", True)),
                            "reference_frame": self.reference,
                        },
                    }
                ]
            }
        )

    def apply(
        self, df: pl.DataFrame | pl.LazyFrame, sample_size: int | None = None
    ) -> pl.DataFrame | pl.LazyFrame:
        if isinstance(df, pl.LazyFrame):
            sample_df = df.head(sample_size or 1000).collect() if sample_size else df.collect()
        else:
            sample_df = df.head(sample_size or len(df)) if sample_size else df
        self.detector.run(sample_df)
        return df

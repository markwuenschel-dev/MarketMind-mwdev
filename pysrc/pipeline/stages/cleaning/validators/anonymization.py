# Relocated from pysrc.data.anonymization
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import singledispatch
from typing import Any

from pysrc.core.runtime.optional_imports import optional_import

np = optional_import("numpy")
pl = optional_import("polars")
from pysrc.core.errors import DataValidationError
from pysrc.core.validation import validate_dataframe
from pysrc.ops.mm_logkit import get_logger

logger = get_logger(__name__)

ANONYMIZERS: dict[str, type["Anonymizer"]] = {}


def register_anonymizer(anon_type: str):
    def decorator(cls: type["Anonymizer"]):
        ANONYMIZERS[anon_type] = cls
        logger.info("Registered anonymizer", extra={"type": anon_type})
        return cls

    return decorator


class Anonymizer(ABC):
    @abstractmethod
    def apply(
        self, df: pl.DataFrame | pl.LazyFrame, columns: list[str]
    ) -> pl.DataFrame | pl.LazyFrame:
        pass


class AnonymizationManager:
    def __init__(self, config, max_workers: int = 8):
        self.config = config
        self.anonymizers: dict[str, Anonymizer] = {}
        self.max_workers = max_workers
        self._register_anonymizers(config.anonymization_techniques)

    def _register_anonymizers(self, anon_cfg: list[dict]):
        for cfg in anon_cfg:
            anon_type = cfg["type"]
            if anon_type not in ANONYMIZERS:
                raise ValueError(f"Unknown anonymizer type: {anon_type}")
            anon = ANONYMIZERS[anon_type](config=cfg)
            self.anonymizers[anon_type] = anon

    def add_anonymizer(self, anon_type: str, anonymizer: Anonymizer):
        self.anonymizers[anon_type] = anonymizer
        logger.info("Dynamically added anonymizer", extra={"type": anon_type})

    @singledispatch
    def anonymize(
        self, data: Any, columns: dict[str, list[str]] | None = None, eager: bool = False
    ) -> Any:
        raise NotImplementedError(f"Unsupported data type: {type(data)}")

    @anonymize.register
    def _(
        self, df: pl.LazyFrame, columns: dict[str, list[str]] | None = None, eager: bool = False
    ) -> pl.LazyFrame:
        columns = columns or {}
        for anon_type, anon in self.anonymizers.items():
            cols = columns.get(anon_type, [])
            if cols:
                df = anon.apply(df, cols)
        if eager:
            collected = df.collect()
            validate_dataframe(collected)
            return collected
        return df

    @anonymize.register
    def _(
        self, df: pl.DataFrame, columns: dict[str, list[str]] | None = None, eager: bool = False
    ) -> pl.DataFrame:
        columns = columns or {}
        for anon_type, anon in self.anonymizers.items():
            cols = columns.get(anon_type, [])
            if cols:
                df = anon.apply(df, cols)
        validate_dataframe(df)
        return df

    @anonymize.register
    def _(
        self, dfs: dict, columns: dict[str, list[str]] | None = None, eager: bool = False
    ) -> dict:
        results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.anonymize, df, columns, eager): key for key, df in dfs.items()
            }
            for future in as_completed(futures):
                key = futures[future]
                try:
                    results[key] = future.result()
                except Exception as e:
                    logger.error(
                        "Anonymization failed for key", extra={"key": key, "error": str(e)}
                    )
                    raise
        return results


@register_anonymizer("Masking")
class MaskingAnonymizer(Anonymizer):
    def __init__(self, config: dict | None = None):
        config = config or {}
        self.mask_char = config.get("mask_char", "*")

    def apply(
        self, df: pl.DataFrame | pl.LazyFrame, columns: list[str]
    ) -> pl.DataFrame | pl.LazyFrame:
        for col in columns:
            if col in df.columns:
                df = df.with_columns(pl.col(col).str.replace_all(r".", self.mask_char))
        return df


@register_anonymizer("Hashing")
class HashingAnonymizer(Anonymizer):
    def __init__(self, config: dict | None = None):
        config = config or {}
        self.algorithm = config.get("algorithm", "polars_hash")

    def apply(
        self, df: pl.DataFrame | pl.LazyFrame, columns: list[str]
    ) -> pl.DataFrame | pl.LazyFrame:
        for col in columns:
            if col in df.columns:
                hashed = pl.col(col).hash()
                hex_hash = hashed.map_elements(
                    lambda x: f"{x:016x}", return_dtype=pl.Utf8, parallel=True
                )
                df = df.with_columns(hex_hash.alias(col))
        return df


@register_anonymizer("DifferentialPrivacy")
class DPAnonymizer(Anonymizer):
    def __init__(self, config: dict | None = None):
        config = config or {}
        self.epsilon = config.get("epsilon", 1.0)
        self.governance_mode = str(config.get("governance_mode", "nongoverned")).lower()
        self.seed = config.get("seed")
        if self.governance_mode == "governed":
            raise DataValidationError(
                "DifferentialPrivacy anonymization is non-governed only until seeded runtime lineage is implemented",
                details={"governance_mode": self.governance_mode},
            )
        if np is None:
            raise DataValidationError(
                "DifferentialPrivacy anonymization requires numpy",
                details={"dependency": "numpy"},
            )

    def apply(
        self, df: pl.DataFrame | pl.LazyFrame, columns: list[str]
    ) -> pl.DataFrame | pl.LazyFrame:
        rng = np.random.default_rng(self.seed)
        for col in columns:
            if col in df.columns and df.schema[col].is_numeric():

                def noise_func(s):
                    return s + pl.Series(rng.laplace(0, 1 / self.epsilon, len(s)))

                df = df.with_columns(pl.col(col).map_batches(noise_func))
        return df

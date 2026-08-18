"""Pipeline output paths consumed by research lanes and backtest orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import polars as pl

    from pysrc.pipeline.contracts.p2 import P2Config
    from pysrc.pipeline.meta_router_products import MetaRouterProductPaths

PROCESSED_DATA_ROOT = Path("data/processed")
FULL_INDICATOR_FEATURE_PANEL_DIR = PROCESSED_DATA_ROOT / "full_indicator_feature_panel"
MACRO_STATE_PANEL_DIR = PROCESSED_DATA_ROOT / "macro_state_panel"
_INDICATOR_PANEL_FILENAME = "panel.parquet"
_MACRO_STATE_PANEL_FILENAME = "panel.parquet"
_MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True, slots=True)
class PipelineProductPaths:
    root: Path
    full_indicator_feature_panel: Path
    manifest: Path

    @property
    def indicator_panel_parquet(self) -> Path:
        return self.full_indicator_feature_panel / _INDICATOR_PANEL_FILENAME

    @property
    def macro_state_panel_parquet(self) -> Path:
        return MACRO_STATE_PANEL_DIR / _MACRO_STATE_PANEL_FILENAME


def resolve_pipeline_product_paths(root: Path | str | None = None) -> PipelineProductPaths:
    base = Path(root) if root is not None else PROCESSED_DATA_ROOT
    return PipelineProductPaths(
        root=base,
        full_indicator_feature_panel=base / "full_indicator_feature_panel",
        manifest=base / _MANIFEST_FILENAME,
    )


def pipeline_indicator_panel_path(config: P2Config) -> Path:
    return resolve_pipeline_product_paths(config.processed_data_root).indicator_panel_parquet


def resolve_pipeline_indicator_features_path(config: P2Config) -> tuple[Path, str]:
    """Return (path, source_kind) for ticker-level indicator features."""

    pipeline_path = pipeline_indicator_panel_path(config)
    if pipeline_path.is_file():
        return pipeline_path, "pipeline_preprocessing"

    return pipeline_path, "missing"


def require_pipeline_indicator_panel(config: P2Config) -> Path:
    path, source_kind = resolve_pipeline_indicator_features_path(config)
    if source_kind == "missing":
        raise FileNotFoundError(
            f"Pipeline indicator panel missing at {path}. "
            "Run: python -m pysrc.cli.marketmind dataprep run -c <pipeline_config.yaml>"
        )
    return path


def meta_router_product_paths(run_dir: Path | str) -> MetaRouterProductPaths:
    from pysrc.pipeline.meta_router_products import resolve_meta_router_products

    return resolve_meta_router_products(run_dir)


def load_pipeline_indicator_panel_dataframe(
    processed_root: Path | str | None = None,
    *,
    symbol: str | None = None,
) -> pd.DataFrame:
    """Load the indicator feature panel product as a pandas DataFrame."""

    path = resolve_pipeline_product_paths(processed_root).indicator_panel_parquet
    if not path.is_file():
        raise FileNotFoundError(f"Pipeline indicator panel missing at {path}. Run dataprep first.")
    frame = pd.read_parquet(path)
    if symbol and "instrument" in frame.columns:
        frame = frame.loc[frame["instrument"].astype(str) == symbol].copy()
    elif symbol and "symbol" in frame.columns:
        frame = frame.loc[frame["symbol"].astype(str) == symbol].copy()
    return frame


def load_pipeline_indicator_panel_polars(
    processed_root: Path | str | None = None,
    *,
    symbol: str | None = None,
) -> pl.DataFrame:
    """Load the indicator feature panel product as a polars DataFrame."""

    import polars as pl

    pdf = load_pipeline_indicator_panel_dataframe(processed_root, symbol=symbol)
    return pl.from_pandas(pdf)


def normalize_pipeline_panel_for_backtest(frame: object) -> pl.DataFrame:
    """Map pipeline panel columns to orchestrator/backtest conventions (symbol, close, date)."""

    import polars as pl

    df = frame if isinstance(frame, pl.DataFrame) else pl.from_pandas(frame)
    rename: dict[str, str] = {}
    if "symbol" not in df.columns and "instrument" in df.columns:
        rename["instrument"] = "symbol"
    if "close" not in df.columns and "adj_close" in df.columns:
        rename["adj_close"] = "close"
    if rename:
        df = df.rename(rename)
    return df


def load_macro_state_panel_fixture(*, n_days: int = 5) -> pd.DataFrame:
    """Synthetic macro_state_panel for opt-in channel tests (research lane)."""

    from pysrc.contracts.meta_router import MACRO_STATE_PANEL_COLUMNS

    dates = pd.date_range("2024-01-02", periods=n_days, freq="B")
    rows: list[dict[str, object]] = []
    for date in dates:
        rows.append(
            {
                "date": date.strftime("%Y-%m-%d"),
                "interval": "1d",
                "risk_on_probability": 0.55,
                "risk_off_probability": 0.45,
                "expected_volatility": 0.12,
                "liquidity_stress": 0.1,
                "macro_regime_probabilities": "{}",
                "sector_tilts": "{}",
                "asset_class_tilts": "{}",
                "confidence": 0.5,
            }
        )
    frame = pd.DataFrame(rows)
    return frame[list(MACRO_STATE_PANEL_COLUMNS)]


__all__ = [
    "PROCESSED_DATA_ROOT",
    "FULL_INDICATOR_FEATURE_PANEL_DIR",
    "PipelineProductPaths",
    "load_pipeline_indicator_panel_dataframe",
    "load_pipeline_indicator_panel_polars",
    "load_macro_state_panel_fixture",
    "MACRO_STATE_PANEL_DIR",
    "meta_router_product_paths",
    "normalize_pipeline_panel_for_backtest",
    "pipeline_indicator_panel_path",
    "require_pipeline_indicator_panel",
    "resolve_pipeline_indicator_features_path",
    "resolve_pipeline_product_paths",
]

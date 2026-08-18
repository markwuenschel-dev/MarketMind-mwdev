"""
DataView: the single point-in-time (PIT) front door for mutable data access.

- as_of() uses snapshot semantics: one row per symbol as known at knowledge_date.
- Visible rows satisfy valid_time <= T and knowledge_time <= T.
- Phase I-A operates at daily granularity: canonical temporal columns are normalized to date;
  sub-day TTLs and intraday restatement ordering are out of scope for this phase.
- Non-bitemporal input fails closed. Symbol identity is validated at registration time.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any

import pandas as pd

from pysrc.core.errors import PITViolationError, StalenessError
from pysrc.data.pit_config import (
    FieldTTLConfig,
    FillPolicy,
    MissingPolicy,
    ResolvedFieldConfig,
    resolve_field_config,
)
from pysrc.data.universe import Universe
from pysrc.ops.telemetry import SPAN_DATAVIEW_AS_OF, tracer

_CANONICAL_VALID = "valid_time"
_CANONICAL_KNOW = "knowledge_time"


def _empty_snapshot(fields: Sequence[str]) -> pd.DataFrame:
    """Return a DataFrame with columns symbol, valid_time, knowledge_time, and fields (all empty)."""
    cols = ["symbol", _CANONICAL_VALID, _CANONICAL_KNOW] + list(fields)
    return pd.DataFrame(columns=cols)


def _validate_symbol_identity(series: pd.Series) -> None:
    """Reject null/NaN and empty or whitespace-only symbols at registration time."""
    for _idx, val in series.items():
        if pd.isna(val) or val is None:
            raise PITViolationError(
                "register_source: symbol column contains null or NaN",
                details={"symbol_column": "symbol"},
            )
        s = str(val).strip()
        if not s:
            raise PITViolationError(
                "register_source: symbol column contains empty or whitespace-only value",
                details={"symbol_column": "symbol"},
            )


def _resolve_field_value(
    visible: pd.DataFrame,
    anchor: pd.Series,
    field: str,
    fill_policy: FillPolicy,
    T: date,
) -> tuple[Any, date | None]:
    """
    Resolve value and knowledge_time for one field.
    REJECT: only anchor row. FORWARD: walk backward through visible rows for latest non-null.
    Returns (value, knowledge_time of observation used).
    """
    if field not in visible.columns:
        return (float("nan"), anchor[_CANONICAL_KNOW])
    if fill_policy == FillPolicy.REJECT:
        v = anchor[field]
        return (v if pd.notna(v) and v is not None else float("nan"), anchor[_CANONICAL_KNOW])
    # FORWARD: anchor if non-null, else walk backward (visible sorted desc by valid_time, knowledge_time)
    if pd.notna(anchor.get(field)) and anchor[field] is not None:
        return (anchor[field], anchor[_CANONICAL_KNOW])
    for _i, row in visible.iloc[1:].iterrows():
        if pd.notna(row.get(field)) and row[field] is not None:
            return (row[field], row[_CANONICAL_KNOW])
    return (float("nan"), anchor[_CANONICAL_KNOW])


class DataView:
    """
    In-memory PIT view over registered source frames. Append-only store.
    Symbol identity is validated at registration time.
    """

    def __init__(
        self,
        universe: Universe | None = None,
        pit_config: FieldTTLConfig | None = None,
        *,
        pit_required: bool = False,
    ) -> None:
        self._universe = universe or Universe()
        self._config = pit_config or FieldTTLConfig()
        self._chunks: list[pd.DataFrame] = []
        self._pit_required = pit_required

    def register_source(
        self,
        df: pd.DataFrame,
        *,
        valid_time_col: str = _CANONICAL_VALID,
        knowledge_time_col: str = _CANONICAL_KNOW,
        seed_fixture_membership: bool = False,
    ) -> None:
        """
        Register a source frame. Reject empty; require and validate symbol column at write time;
        validate temporal columns; rename to canonical; coerce to date; sort; append only.
        Optionally seed universe membership (FIXTURE_SEED). Fixture seeding supports relisting:
        if the symbol is inactive at the seed date (e.g. previously delisted), a new list event
        is appended; if already active at the seed date, no duplicate event is added.
        """
        if df is None or df.empty:
            raise PITViolationError(
                "register_source requires a non-empty frame", details={"empty": True}
            )
        if "symbol" not in df.columns:
            raise PITViolationError(
                "register_source requires a symbol column for symbol identity validation",
                details={"missing_column": "symbol"},
            )
        _validate_symbol_identity(df["symbol"])
        if valid_time_col not in df.columns:
            raise PITViolationError(
                "register_source requires valid_time column",
                details={"missing_column": "valid_time", "expected": valid_time_col},
            )
        if knowledge_time_col not in df.columns:
            raise PITViolationError(
                "register_source requires knowledge_time column",
                details={"missing_column": "knowledge_time", "expected": knowledge_time_col},
            )
        out = df.copy()
        if valid_time_col != _CANONICAL_VALID:
            out = out.rename(columns={valid_time_col: _CANONICAL_VALID})
        if knowledge_time_col != _CANONICAL_KNOW:
            out = out.rename(columns={knowledge_time_col: _CANONICAL_KNOW})
        out[_CANONICAL_VALID] = pd.to_datetime(out[_CANONICAL_VALID]).dt.date
        out[_CANONICAL_KNOW] = pd.to_datetime(out[_CANONICAL_KNOW]).dt.date
        out = out.sort_values([_CANONICAL_KNOW, _CANONICAL_VALID], kind="stable")
        self._chunks.append(out)
        if seed_fixture_membership:
            from pysrc.data.universe import MembershipReasonCode

            for sym in out["symbol"].unique().tolist():
                if pd.isna(sym) or sym is None:
                    continue
                sym_str = str(sym).strip()
                if not sym_str:
                    continue
                min_date = out.loc[out["symbol"] == sym, _CANONICAL_VALID].min()
                if pd.isna(min_date):
                    continue
                seed_date = min_date if isinstance(min_date, date) else min_date
                is_active, _ = self._universe.effective_state_at(sym_str, seed_date)
                if not is_active:
                    self._universe.register(
                        sym_str, seed_date, reason_code=MembershipReasonCode.FIXTURE_SEED
                    )

    def _store(self) -> pd.DataFrame:
        """Combined append-only store (all chunks concatenated)."""
        if not self._chunks:
            return pd.DataFrame()
        return pd.concat(self._chunks, axis=0, ignore_index=True, sort=False)

    def as_of(
        self,
        symbols: Sequence[str],
        fields: Sequence[str],
        knowledge_date: date,
    ) -> pd.DataFrame:
        """
        Return one snapshot row per symbol as known at knowledge_date.
        Visibility: valid_time <= T and knowledge_time <= T. Columns: symbol, valid_time, knowledge_time, fields.
        """
        with tracer.start_as_current_span(SPAN_DATAVIEW_AS_OF) as span:
            span.set_attribute("as_of_date", knowledge_date.isoformat())
            span.set_attribute("symbol_count", len(symbols))
            span.set_attribute("pit_required", bool(self._pit_required))
            return self._as_of_impl(symbols, fields, knowledge_date)

    def _as_of_impl(
        self,
        symbols: Sequence[str],
        fields: Sequence[str],
        knowledge_date: date,
    ) -> pd.DataFrame:
        store = self._store()
        if store.empty:
            return _empty_snapshot(fields)
        if symbols and "symbol" not in store.columns:
            raise PITViolationError(
                "as_of requires symbol column in store when querying by symbols",
                details={"columns": list(store.columns)},
            )
        if _CANONICAL_VALID not in store.columns or _CANONICAL_KNOW not in store.columns:
            raise PITViolationError(
                "as_of encountered store missing canonical temporal columns",
                details={"columns": list(store.columns)},
            )
        rows: list[dict[str, Any]] = []
        for symbol in symbols:
            visible = store[
                (store["symbol"] == symbol)
                & (store[_CANONICAL_VALID] <= knowledge_date)
                & (store[_CANONICAL_KNOW] <= knowledge_date)
            ]
            if visible.empty:
                continue
            row = self._resolve_snapshot(visible, symbol, list(fields), knowledge_date)
            if row is not None:
                rows.append(row)
        if not rows:
            return _empty_snapshot(fields)
        out = pd.DataFrame(rows)
        out[_CANONICAL_KNOW] = knowledge_date
        return out

    def _resolve_snapshot(
        self,
        visible: pd.DataFrame,
        symbol: str,
        fields: list[str],
        T: date,
    ) -> dict[str, Any] | None:
        """
        Resolve one snapshot row. Anchor = greatest (valid_time, knowledge_time).
        Per-field: use resolve_field_config (namespace-aware) for TTL, FillPolicy, MissingPolicy.
        Staleness from the knowledge_time of the resolved observation for that field.
        """
        if visible.empty:
            return None
        if _CANONICAL_VALID not in visible.columns or _CANONICAL_KNOW not in visible.columns:
            raise PITViolationError(
                "corrupted store entry missing canonical temporal columns",
                details={"columns": list(visible.columns)},
            )
        visible = visible.sort_values(
            [_CANONICAL_VALID, _CANONICAL_KNOW],
            ascending=[False, False],
            kind="stable",
        )
        anchor = visible.iloc[0]
        anchor_valid = anchor[_CANONICAL_VALID]
        row: dict[str, Any] = {
            "symbol": symbol,
            _CANONICAL_VALID: anchor_valid,
            _CANONICAL_KNOW: T,
        }
        for f in fields:
            if f in (_CANONICAL_VALID, _CANONICAL_KNOW, "symbol"):
                continue
            field_cfg: ResolvedFieldConfig = resolve_field_config(f, self._config)
            value, obs_know = _resolve_field_value(visible, anchor, f, field_cfg.fill_policy, T)
            if obs_know is None:
                obs_know = anchor[_CANONICAL_KNOW]
            staleness_days = (T - obs_know).days if isinstance(obs_know, date) else 0
            stale = staleness_days > field_cfg.ttl_days
            if pd.isna(value) or (value is None):
                if field_cfg.missing_policy == MissingPolicy.FAIL:
                    raise StalenessError(
                        f"missing field {f}",
                        details={"field": f, "symbol": symbol, "knowledge_date": str(T)},
                    )
                row[f] = float("nan")
                continue
            if stale:
                if field_cfg.missing_policy == MissingPolicy.FAIL:
                    raise StalenessError(
                        f"stale field {f}",
                        details={
                            "field": f,
                            "symbol": symbol,
                            "knowledge_date": str(T),
                            "ttl_days": field_cfg.ttl_days,
                        },
                    )
                row[f] = float("nan")
                continue
            row[f] = value
        for f in fields:
            if f not in row:
                row[f] = float("nan")
        return row

    def universe_as_of(
        self,
        knowledge_date: date,
        filters: Any | None = None,
    ) -> set[str]:
        """Return symbols that are members of the universe at knowledge_date."""
        _ = filters  # reserved for Phase I-C
        return self._universe.members_as_of(knowledge_date)

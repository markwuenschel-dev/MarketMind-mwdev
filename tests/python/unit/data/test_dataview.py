# Unit tests for DataView.as_of() Phase I-A contract and PIT invariants.
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from pysrc.core.errors import PITViolationError, StalenessError
from pysrc.data.dataview import DataView
from pysrc.data.pit_config import (
    FieldTTLConfig,
    FillPolicy,
    MissingPolicy,
    ResolvedFieldConfig,
)
from pysrc.data.universe import MembershipReasonCode, Universe


def _df(
    symbol: str,
    valid_time: date,
    knowledge_time: date,
    **kwargs: object,
) -> pd.DataFrame:
    d = {"symbol": [symbol], "valid_time": [valid_time], "knowledge_time": [knowledge_time]}
    d.update(kwargs)
    return pd.DataFrame(d)


@pytest.mark.determinism("d1")
def test_register_fails_missing_valid_time_col() -> None:
    df = pd.DataFrame({"symbol": ["A"], "knowledge_time": [date(2024, 1, 1)], "close": [100.0]})
    dv = DataView()
    with pytest.raises(PITViolationError) as exc:
        dv.register_source(df)
    assert "valid_time" in str(exc.value).lower() or "valid_time" in (exc.value.details or {}).get(
        "expected", ""
    )


@pytest.mark.determinism("d1")
def test_register_fails_missing_knowledge_time_col() -> None:
    df = pd.DataFrame({"symbol": ["A"], "valid_time": [date(2024, 1, 1)], "close": [100.0]})
    dv = DataView()
    with pytest.raises(PITViolationError) as exc:
        dv.register_source(df)
    assert "knowledge_time" in str(exc.value).lower() or "knowledge_time" in (
        exc.value.details or {}
    ).get("expected", "")


@pytest.mark.determinism("d1")
def test_register_fails_on_empty_frame() -> None:
    df = pd.DataFrame(columns=["symbol", "valid_time", "knowledge_time"])
    dv = DataView()
    with pytest.raises(PITViolationError):
        dv.register_source(df)


@pytest.mark.determinism("d1")
def test_register_fails_missing_symbol_col() -> None:
    """Symbol identity is validated at registration: symbol column is required."""
    df = pd.DataFrame(
        {
            "valid_time": [date(2024, 1, 1)],
            "knowledge_time": [date(2024, 1, 1)],
            "close": [100.0],
        }
    )
    dv = DataView()
    with pytest.raises(PITViolationError) as exc:
        dv.register_source(df)
    assert "symbol" in str(exc.value).lower()


@pytest.mark.determinism("d1")
def test_register_fails_empty_symbol_value() -> None:
    """Symbol identity is strict: empty or whitespace-only symbol rejected at registration."""
    df = pd.DataFrame(
        {
            "symbol": ["A", "  ", "B"],
            "valid_time": [date(2024, 1, 1), date(2024, 1, 1), date(2024, 1, 1)],
            "knowledge_time": [date(2024, 1, 1), date(2024, 1, 1), date(2024, 1, 1)],
            "close": [100.0, 101.0, 50.0],
        }
    )
    dv = DataView()
    with pytest.raises(PITViolationError) as exc:
        dv.register_source(df)
    assert "symbol" in str(exc.value).lower() or "empty" in str(exc.value).lower()


@pytest.mark.determinism("d1")
def test_row_with_valid_time_after_T_is_invisible() -> None:
    df = pd.DataFrame(
        {
            "symbol": ["A"],
            "valid_time": [date(2024, 1, 15)],
            "knowledge_time": [date(2024, 1, 10)],
            "close": [100.0],
        }
    )
    dv = DataView()
    dv.register_source(df)
    out = dv.as_of(["A"], ["close"], date(2024, 1, 10))
    assert out.empty


@pytest.mark.determinism("d1")
def test_row_with_knowledge_time_after_T_is_invisible() -> None:
    df = pd.DataFrame(
        {
            "symbol": ["A"],
            "valid_time": [date(2024, 1, 5)],
            "knowledge_time": [date(2024, 1, 15)],
            "close": [100.0],
        }
    )
    dv = DataView()
    dv.register_source(df)
    out = dv.as_of(["A"], ["close"], date(2024, 1, 10))
    assert out.empty


@pytest.mark.determinism("d1")
def test_returns_one_row_per_symbol() -> None:
    df = pd.DataFrame(
        {
            "symbol": ["A", "A", "B"],
            "valid_time": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 1)],
            "knowledge_time": [date(2024, 1, 2), date(2024, 1, 2), date(2024, 1, 2)],
            "close": [99.0, 100.0, 50.0],
        }
    )
    dv = DataView()
    dv.register_source(df)
    out = dv.as_of(["A", "B"], ["close"], date(2024, 1, 2))
    assert len(out) == 2
    assert set(out["symbol"].tolist()) == {"A", "B"}
    assert out[out["symbol"] == "A"]["close"].iloc[0] == 100.0
    assert out[out["symbol"] == "B"]["close"].iloc[0] == 50.0


@pytest.mark.determinism("d1")
def test_restatement_visible_only_after_later_knowledge_time() -> None:
    df = pd.DataFrame(
        {
            "symbol": ["A", "A"],
            "valid_time": [date(2024, 1, 1), date(2024, 1, 1)],
            "knowledge_time": [date(2024, 1, 2), date(2024, 1, 5)],
            "close": [100.0, 101.0],
        }
    )
    dv = DataView()
    dv.register_source(df)
    out_jan3 = dv.as_of(["A"], ["close"], date(2024, 1, 3))
    assert out_jan3["close"].iloc[0] == 100.0
    out_jan6 = dv.as_of(["A"], ["close"], date(2024, 1, 6))
    assert out_jan6["close"].iloc[0] == 101.0


@pytest.mark.determinism("d1")
def test_ttl_is_per_field_not_global() -> None:
    # close has short TTL, volume long TTL; at T both visible but close is stale
    default = ResolvedFieldConfig(30, FillPolicy.FORWARD, MissingPolicy.WARN)
    cfg = FieldTTLConfig(
        default_config=default,
        field_configs={
            "close": ResolvedFieldConfig(2, FillPolicy.FORWARD, MissingPolicy.WARN),
            "volume": ResolvedFieldConfig(30, FillPolicy.FORWARD, MissingPolicy.WARN),
        },
    )
    df = pd.DataFrame(
        {
            "symbol": ["A"],
            "valid_time": [date(2024, 1, 1)],
            "knowledge_time": [date(2024, 1, 1)],
            "close": [100.0],
            "volume": [1e6],
        }
    )
    dv = DataView(pit_config=cfg)
    dv.register_source(df)
    T = date(2024, 1, 10)  # 9 days later: close stale (TTL 2), volume ok (TTL 30)
    out = dv.as_of(["A"], ["close", "volume"], T)
    assert len(out) == 1
    assert pd.isna(out["close"].iloc[0])
    assert out["volume"].iloc[0] == 1e6


@pytest.mark.determinism("d1")
def test_reject_fill_policy_does_not_walk_backward_from_anchor_gap() -> None:
    """REJECT must only allow the anchor row for that field; no backward-fill."""
    default = ResolvedFieldConfig(30, FillPolicy.REJECT, MissingPolicy.WARN)
    cfg = FieldTTLConfig(default_config=default)
    df = pd.DataFrame(
        {
            "symbol": ["A", "A"],
            "valid_time": [date(2024, 1, 1), date(2024, 1, 2)],
            "knowledge_time": [date(2024, 1, 2), date(2024, 1, 2)],
            "close": [99.0, float("nan")],
        }
    )
    dv = DataView(pit_config=cfg)
    dv.register_source(df)
    out = dv.as_of(["A"], ["close"], date(2024, 1, 2))
    assert len(out) == 1
    assert pd.isna(out["close"].iloc[0])


@pytest.mark.determinism("d1")
def test_missing_fail_policy_raises_staleness_error() -> None:
    default = ResolvedFieldConfig(30, FillPolicy.FORWARD, MissingPolicy.FAIL)
    cfg = FieldTTLConfig(default_config=default)
    df = pd.DataFrame(
        {
            "symbol": ["A"],
            "valid_time": [date(2024, 1, 1)],
            "knowledge_time": [date(2024, 1, 1)],
            "close": [float("nan")],
        }
    )
    dv = DataView(pit_config=cfg)
    dv.register_source(df)
    with pytest.raises(StalenessError):
        dv.as_of(["A"], ["close"], date(2024, 1, 2))


@pytest.mark.determinism("d1")
def test_universe_excludes_symbol_on_and_after_delist_date() -> None:
    u = Universe()
    u.register("A", date(2024, 1, 1), reason_code=MembershipReasonCode.GOVERNED)
    u.delist("A", date(2024, 2, 1), reason_code=MembershipReasonCode.GOVERNED)
    assert "A" not in u.members_as_of(date(2024, 2, 1))
    assert "A" not in u.members_as_of(date(2024, 2, 15))


@pytest.mark.determinism("d1")
def test_universe_preserves_symbol_before_delist_boundary() -> None:
    u = Universe()
    u.register("A", date(2024, 1, 1), reason_code=MembershipReasonCode.GOVERNED)
    u.delist("A", date(2024, 2, 1), reason_code=MembershipReasonCode.GOVERNED)
    assert "A" in u.members_as_of(date(2024, 1, 15))
    assert "A" in u.members_as_of(date(2024, 1, 31))


@pytest.mark.determinism("d1")
def test_universe_not_inferred_from_store() -> None:
    df = pd.DataFrame(
        {
            "symbol": ["B"],
            "valid_time": [date(2024, 1, 1)],
            "knowledge_time": [date(2024, 1, 1)],
            "close": [50.0],
        }
    )
    u = Universe()
    u.register("A", date(2024, 1, 1), reason_code=MembershipReasonCode.GOVERNED)
    dv = DataView(universe=u)
    dv.register_source(df, seed_fixture_membership=False)
    members = dv.universe_as_of(date(2024, 1, 2))
    assert "A" in members
    assert "B" not in members


@pytest.mark.determinism("d1")
def test_as_of_empty_symbols_returns_empty_snapshot() -> None:
    df = pd.DataFrame(
        {
            "symbol": ["A"],
            "valid_time": [date(2024, 1, 1)],
            "knowledge_time": [date(2024, 1, 1)],
            "close": [100.0],
        }
    )
    dv = DataView()
    dv.register_source(df)
    out = dv.as_of([], ["close"], date(2024, 1, 1))
    assert out.empty
    assert list(out.columns) == ["symbol", "valid_time", "knowledge_time", "close"]


@pytest.mark.determinism("d1")
def test_register_accepts_renamed_temporal_columns() -> None:
    df = pd.DataFrame(
        {
            "symbol": ["A"],
            "bar_date": [date(2024, 1, 1)],
            "as_of_date": [date(2024, 1, 1)],
            "close": [100.0],
        }
    )
    dv = DataView()
    dv.register_source(df, valid_time_col="bar_date", knowledge_time_col="as_of_date")
    out = dv.as_of(["A"], ["close"], date(2024, 1, 1))
    assert len(out) == 1
    assert "valid_time" in out.columns
    assert "knowledge_time" in out.columns


@pytest.mark.determinism("d1")
def test_universe_relist_works_after_delist() -> None:
    """Relisting after a prior delist must work: append list event after delist."""
    u = Universe()
    u.register("A", date(2024, 1, 1), reason_code=MembershipReasonCode.GOVERNED)
    u.delist("A", date(2024, 2, 1), reason_code=MembershipReasonCode.GOVERNED)
    u.register("A", date(2024, 3, 1), reason_code=MembershipReasonCode.RELIST)
    assert "A" not in u.members_as_of(date(2024, 2, 15))
    assert "A" in u.members_as_of(date(2024, 3, 15))
    assert "A" in u.members_as_of(date(2024, 4, 1))


@pytest.mark.determinism("d1")
def test_universe_delist_excludes_on_and_after_boundary() -> None:
    """Delist excludes symbol on and after delist_date; includes before."""
    u = Universe()
    u.register("X", date(2024, 1, 1), reason_code=MembershipReasonCode.GOVERNED)
    u.delist("X", date(2024, 6, 15), reason_code=MembershipReasonCode.GOVERNED)
    assert "X" in u.members_as_of(date(2024, 6, 14))
    assert "X" not in u.members_as_of(date(2024, 6, 15))
    assert "X" not in u.members_as_of(date(2024, 7, 1))


@pytest.mark.determinism("d1")
def test_namespace_ttl_policy_matching_exercised() -> None:
    """Namespace config (e.g. price.*) is used when field name implies namespace."""
    default = ResolvedFieldConfig(100, FillPolicy.FORWARD, MissingPolicy.WARN)
    # price.close and price.open get TTL 5 from namespace "price.*"
    cfg = FieldTTLConfig(
        default_config=default,
        namespace_configs={
            "price.*": ResolvedFieldConfig(5, FillPolicy.FORWARD, MissingPolicy.WARN),
        },
    )
    df = pd.DataFrame(
        {
            "symbol": ["A"],
            "valid_time": [date(2024, 1, 1)],
            "knowledge_time": [date(2024, 1, 1)],
            "price.close": [100.0],
            "price.open": [99.0],
        }
    )
    dv = DataView(pit_config=cfg)
    dv.register_source(df)
    T = date(2024, 1, 20)  # 19 days later: namespace TTL 5 -> both stale
    out = dv.as_of(["A"], ["price.close", "price.open"], T)
    assert len(out) == 1
    assert pd.isna(out["price.close"].iloc[0])
    assert pd.isna(out["price.open"].iloc[0])


@pytest.mark.determinism("d1")
def test_per_field_fill_and_missing_policies_differ() -> None:
    """One field FORWARD+WARN, another REJECT+FAIL; per-field config is applied."""
    cfg = FieldTTLConfig(
        default_config=ResolvedFieldConfig(30, FillPolicy.FORWARD, MissingPolicy.WARN),
        field_configs={
            "a": ResolvedFieldConfig(30, FillPolicy.FORWARD, MissingPolicy.WARN),
            "b": ResolvedFieldConfig(30, FillPolicy.REJECT, MissingPolicy.FAIL),
        },
    )
    df = pd.DataFrame(
        {
            "symbol": ["A", "A"],
            "valid_time": [date(2024, 1, 1), date(2024, 1, 2)],
            "knowledge_time": [date(2024, 1, 2), date(2024, 1, 2)],
            "a": [10.0, float("nan")],
            "b": [float("nan"), float("nan")],
        }
    )
    dv = DataView(pit_config=cfg)
    dv.register_source(df)
    # a: FORWARD so can walk back to 10.0; b: REJECT so anchor only -> nan, FAIL -> raise
    with pytest.raises(StalenessError):
        dv.as_of(["A"], ["a", "b"], date(2024, 1, 2))


@pytest.mark.determinism("d1")
def test_future_corrections_do_not_alter_earlier_queries() -> None:
    """Later restatements (knowledge_time > T1) must not change as_of(..., T1) result."""
    df = pd.DataFrame(
        {
            "symbol": ["A", "A"],
            "valid_time": [date(2024, 1, 1), date(2024, 1, 1)],
            "knowledge_time": [date(2024, 1, 2), date(2024, 1, 10)],
            "close": [100.0, 105.0],
        }
    )
    dv = DataView()
    dv.register_source(df)
    out_t1 = dv.as_of(["A"], ["close"], date(2024, 1, 3))
    assert out_t1["close"].iloc[0] == 100.0
    out_t2 = dv.as_of(["A"], ["close"], date(2024, 1, 15))
    assert out_t2["close"].iloc[0] == 105.0
    out_t1_again = dv.as_of(["A"], ["close"], date(2024, 1, 3))
    assert out_t1_again["close"].iloc[0] == 100.0


# ----- Same-day event ordering (delist wins over list) -----


@pytest.mark.determinism("d1")
def test_universe_same_date_list_then_delist_excludes_at_T() -> None:
    """When list and delist occur on the same date, delist wins; symbol excluded at T."""
    u = Universe()
    u.register("A", date(2024, 2, 1), reason_code=MembershipReasonCode.GOVERNED)
    u.delist("A", date(2024, 2, 1), reason_code=MembershipReasonCode.GOVERNED)
    assert "A" not in u.members_as_of(date(2024, 2, 1))
    assert "A" not in u.members_as_of(date(2024, 2, 2))


@pytest.mark.determinism("d1")
def test_universe_same_date_delist_then_relist_still_excluded_on_that_date() -> None:
    """Delist then relist on same date: tie-break delist wins, so excluded on that date.
    Relist on a later date is required for symbol to become active again."""
    u = Universe()
    u.register("B", date(2024, 1, 1), reason_code=MembershipReasonCode.GOVERNED)
    u.delist("B", date(2024, 3, 15), reason_code=MembershipReasonCode.GOVERNED)
    u.register("B", date(2024, 3, 15), reason_code=MembershipReasonCode.RELIST)
    assert "B" not in u.members_as_of(date(2024, 3, 15))
    assert "B" not in u.members_as_of(date(2024, 3, 16))
    u.register("B", date(2024, 3, 17), reason_code=MembershipReasonCode.RELIST)
    assert "B" in u.members_as_of(date(2024, 3, 17))
    assert "B" in u.members_as_of(date(2024, 3, 18))


@pytest.mark.determinism("d1")
def test_universe_relist_on_later_date_after_prior_delist() -> None:
    """Relist on a later date after delist: symbol becomes active from relist date."""
    u = Universe()
    u.register("C", date(2024, 1, 1), reason_code=MembershipReasonCode.GOVERNED)
    u.delist("C", date(2024, 2, 1), reason_code=MembershipReasonCode.GOVERNED)
    u.register("C", date(2024, 3, 1), reason_code=MembershipReasonCode.RELIST)
    assert "C" not in u.members_as_of(date(2024, 2, 15))
    assert "C" in u.members_as_of(date(2024, 3, 1))
    assert "C" in u.members_as_of(date(2024, 4, 1))


# ----- Fixture seeding: first-time, no duplicate when active, relist after delist -----


@pytest.mark.determinism("d1")
def test_fixture_seeding_first_time_seeds_list_event() -> None:
    """First-time fixture registration with seed_fixture_membership=True seeds a list event."""
    u = Universe()
    dv = DataView(universe=u)
    df = pd.DataFrame(
        {
            "symbol": ["S1"],
            "valid_time": [date(2024, 1, 5)],
            "knowledge_time": [date(2024, 1, 5)],
            "close": [10.0],
        }
    )
    dv.register_source(df, seed_fixture_membership=True)
    assert "S1" in u.members_as_of(date(2024, 1, 10))


@pytest.mark.determinism("d1")
def test_fixture_seeding_while_already_active_does_not_duplicate_event() -> None:
    """Second fixture registration while symbol already active at seed date does not add duplicate list."""
    u = Universe()
    u.register("S2", date(2024, 1, 1), reason_code=MembershipReasonCode.GOVERNED)
    dv = DataView(universe=u)
    df1 = pd.DataFrame(
        {
            "symbol": ["S2"],
            "valid_time": [date(2024, 1, 2)],
            "knowledge_time": [date(2024, 1, 2)],
            "close": [20.0],
        }
    )
    dv.register_source(df1, seed_fixture_membership=True)
    df2 = pd.DataFrame(
        {
            "symbol": ["S2"],
            "valid_time": [date(2024, 1, 10)],
            "knowledge_time": [date(2024, 1, 10)],
            "close": [21.0],
        }
    )
    dv.register_source(df2, seed_fixture_membership=True)
    assert "S2" in u.members_as_of(date(2024, 1, 15))
    out = dv.as_of(["S2"], ["close"], date(2024, 1, 15))
    assert out["close"].iloc[0] == 21.0


@pytest.mark.determinism("d1")
def test_fixture_seeding_after_prior_delist_appends_relist_and_symbol_active_again() -> None:
    """Fixture registration after a prior delist appends a relist event; symbol becomes active again."""
    u = Universe()
    u.register("S3", date(2024, 1, 1), reason_code=MembershipReasonCode.GOVERNED)
    u.delist("S3", date(2024, 2, 1), reason_code=MembershipReasonCode.GOVERNED)
    assert "S3" not in u.members_as_of(date(2024, 2, 15))
    dv = DataView(universe=u)
    df = pd.DataFrame(
        {
            "symbol": ["S3"],
            "valid_time": [date(2024, 3, 1)],
            "knowledge_time": [date(2024, 3, 1)],
            "close": [30.0],
        }
    )
    dv.register_source(df, seed_fixture_membership=True)
    assert "S3" in u.members_as_of(date(2024, 3, 15))


# ----- Symbol identity: reject None, NaN, blank -----


@pytest.mark.determinism("d1")
def test_register_fails_none_symbol() -> None:
    """Null/None symbol rejected at registration time."""
    df = pd.DataFrame(
        {
            "symbol": ["A", None, "B"],
            "valid_time": [date(2024, 1, 1), date(2024, 1, 1), date(2024, 1, 1)],
            "knowledge_time": [date(2024, 1, 1), date(2024, 1, 1), date(2024, 1, 1)],
            "close": [100.0, 101.0, 50.0],
        }
    )
    dv = DataView()
    with pytest.raises(PITViolationError) as exc:
        dv.register_source(df)
    assert (
        "null" in str(exc.value).lower()
        or "nan" in str(exc.value).lower()
        or "symbol" in str(exc.value).lower()
    )


@pytest.mark.determinism("d1")
def test_register_fails_nan_symbol() -> None:
    """NaN symbol rejected at registration time."""
    df = pd.DataFrame(
        {
            "symbol": ["A", float("nan"), "B"],
            "valid_time": [date(2024, 1, 1), date(2024, 1, 1), date(2024, 1, 1)],
            "knowledge_time": [date(2024, 1, 1), date(2024, 1, 1), date(2024, 1, 1)],
            "close": [100.0, 101.0, 50.0],
        }
    )
    dv = DataView()
    with pytest.raises(PITViolationError) as exc:
        dv.register_source(df)
    assert (
        "null" in str(exc.value).lower()
        or "nan" in str(exc.value).lower()
        or "symbol" in str(exc.value).lower()
    )


# ----- Config precedence: exact beats namespace, namespace for price.close, default for unmatched -----


@pytest.mark.determinism("d1")
def test_exact_match_beats_namespace_config() -> None:
    """Exact field config overrides namespace config (e.g. price.close exact vs price.*)."""
    default = ResolvedFieldConfig(100, FillPolicy.FORWARD, MissingPolicy.WARN)
    cfg = FieldTTLConfig(
        default_config=default,
        field_configs={
            "price.close": ResolvedFieldConfig(2, FillPolicy.FORWARD, MissingPolicy.WARN),
        },
        namespace_configs={
            "price.*": ResolvedFieldConfig(50, FillPolicy.FORWARD, MissingPolicy.WARN),
        },
    )
    from pysrc.data.pit_config import resolve_field_config

    exact_cfg = resolve_field_config("price.close", cfg)
    assert exact_cfg.ttl_days == 2
    other_cfg = resolve_field_config("price.open", cfg)
    assert other_cfg.ttl_days == 50


@pytest.mark.determinism("d1")
def test_namespace_config_used_for_price_close_style_fields() -> None:
    """Namespace config (price.*) is used when field has namespace prefix and no exact match."""
    default = ResolvedFieldConfig(200, FillPolicy.REJECT, MissingPolicy.FAIL)
    cfg = FieldTTLConfig(
        default_config=default,
        namespace_configs={
            "price.*": ResolvedFieldConfig(7, FillPolicy.FORWARD, MissingPolicy.WARN),
        },
    )
    from pysrc.data.pit_config import resolve_field_config

    c = resolve_field_config("price.close", cfg)
    assert c.ttl_days == 7
    assert c.fill_policy == FillPolicy.FORWARD


@pytest.mark.determinism("d1")
def test_default_config_applies_for_unmatched_field() -> None:
    """Default config is used when no exact or namespace match exists."""
    default = ResolvedFieldConfig(42, FillPolicy.REJECT, MissingPolicy.FAIL)
    cfg = FieldTTLConfig(
        default_config=default,
        field_configs={"close": ResolvedFieldConfig(1, FillPolicy.FORWARD, MissingPolicy.WARN)},
        namespace_configs={
            "price.*": ResolvedFieldConfig(5, FillPolicy.FORWARD, MissingPolicy.WARN)
        },
    )
    from pysrc.data.pit_config import resolve_field_config

    c = resolve_field_config("unknown_field", cfg)
    assert c.ttl_days == 42
    assert c.fill_policy == FillPolicy.REJECT
    assert c.missing_policy == MissingPolicy.FAIL


@pytest.mark.determinism("d1")
def test_snapshot_one_row_per_symbol_anchor_deterministic() -> None:
    """Snapshot contract: one row per symbol, anchor chosen deterministically from visible rows."""
    df = pd.DataFrame(
        {
            "symbol": ["A", "A"],
            "valid_time": [date(2024, 1, 1), date(2024, 1, 2)],
            "knowledge_time": [date(2024, 1, 2), date(2024, 1, 2)],
            "close": [98.0, 100.0],
        }
    )
    dv = DataView()
    dv.register_source(df)
    out = dv.as_of(["A"], ["close"], date(2024, 1, 2))
    assert len(out) == 1
    assert out["symbol"].iloc[0] == "A"
    assert out["close"].iloc[0] == 100.0
    assert out["knowledge_time"].iloc[0] == date(2024, 1, 2)


@pytest.mark.determinism("d1")
def test_stale_fail_policy_raises_on_non_missing_value() -> None:
    cfg = FieldTTLConfig(
        default_config=ResolvedFieldConfig(1, FillPolicy.FORWARD, MissingPolicy.FAIL),
    )
    df = pd.DataFrame(
        {
            "symbol": ["A"],
            "valid_time": [date(2024, 1, 1)],
            "knowledge_time": [date(2024, 1, 1)],
            "close": [100.0],
        }
    )
    dv = DataView(pit_config=cfg)
    dv.register_source(df)

    with pytest.raises(StalenessError):
        dv.as_of(["A"], ["close"], date(2024, 1, 3))


@pytest.mark.determinism("d1")
def test_forward_fill_walks_back_to_prior_visible_row() -> None:
    cfg = FieldTTLConfig(
        default_config=ResolvedFieldConfig(30, FillPolicy.FORWARD, MissingPolicy.WARN),
    )
    df = pd.DataFrame(
        {
            "symbol": ["A", "A"],
            "valid_time": [date(2024, 1, 1), date(2024, 1, 2)],
            "knowledge_time": [date(2024, 1, 2), date(2024, 1, 2)],
            "close": [99.0, float("nan")],
        }
    )
    dv = DataView(pit_config=cfg)
    dv.register_source(df)

    out = dv.as_of(["A"], ["close"], date(2024, 1, 2))

    assert len(out) == 1
    assert out["close"].iloc[0] == 99.0


@pytest.mark.determinism("d1")
def test_as_of_raises_when_store_missing_symbol_column() -> None:
    dv = DataView()
    dv._chunks = [
        pd.DataFrame(
            {
                "valid_time": [date(2024, 1, 1)],
                "knowledge_time": [date(2024, 1, 1)],
                "close": [100.0],
            }
        ),
    ]

    with pytest.raises(PITViolationError):
        dv.as_of(["A"], ["close"], date(2024, 1, 1))


@pytest.mark.determinism("d1")
def test_as_of_raises_when_store_missing_temporal_columns() -> None:
    dv = DataView()
    dv._chunks = [
        pd.DataFrame(
            {
                "symbol": ["A"],
                "close": [100.0],
            }
        ),
    ]

    with pytest.raises(PITViolationError):
        dv.as_of(["A"], ["close"], date(2024, 1, 1))


@pytest.mark.determinism("d1")
def test_missing_requested_field_resolves_to_nan() -> None:
    df = pd.DataFrame(
        {
            "symbol": ["A"],
            "valid_time": [date(2024, 1, 1)],
            "knowledge_time": [date(2024, 1, 1)],
            "close": [100.0],
        }
    )
    dv = DataView()
    dv.register_source(df)

    out = dv.as_of(["A"], ["open"], date(2024, 1, 2))

    assert len(out) == 1
    assert pd.isna(out["open"].iloc[0])


@pytest.mark.determinism("d1")
def test_resolve_snapshot_returns_none_for_empty_visible_frame() -> None:
    dv = DataView()
    visible = pd.DataFrame(columns=["symbol", "valid_time", "knowledge_time", "close"])

    out = dv._resolve_snapshot(visible, "A", ["close"], date(2024, 1, 2))

    assert out is None


@pytest.mark.determinism("d1")
def test_resolve_snapshot_raises_on_missing_temporal_columns() -> None:
    dv = DataView()
    visible = pd.DataFrame({"symbol": ["A"], "close": [100.0]})

    with pytest.raises(PITViolationError):
        dv._resolve_snapshot(visible, "A", ["close"], date(2024, 1, 2))


@pytest.mark.determinism("d1")
def test_as_of_skips_symbol_when_resolver_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    df = pd.DataFrame(
        {
            "symbol": ["A"],
            "valid_time": [date(2024, 1, 1)],
            "knowledge_time": [date(2024, 1, 1)],
            "close": [100.0],
        }
    )
    dv = DataView()
    dv.register_source(df)

    def _fake_resolver(
        _visible: pd.DataFrame,
        _symbol: str,
        _fields: list[str],
        _knowledge_date: date,
    ) -> None:
        return None

    monkeypatch.setattr(dv, "_resolve_snapshot", _fake_resolver)

    out = dv.as_of(["A"], ["close"], date(2024, 1, 1))

    assert out.empty
    assert list(out.columns) == ["symbol", "valid_time", "knowledge_time", "close"]


@pytest.mark.determinism("d1")
def test_register_source_fixture_seeding_defensive_skips_invalid_seed_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    u = Universe()
    dv = DataView(universe=u)
    df = pd.DataFrame(
        {
            "symbol": [pd.NA, "   ", "B"],
            "valid_time": [date(2024, 1, 1), date(2024, 1, 1), None],
            "knowledge_time": [date(2024, 1, 1), date(2024, 1, 1), date(2024, 1, 1)],
            "close": [1.0, 2.0, 3.0],
        }
    )

    monkeypatch.setattr("pysrc.data.dataview._validate_symbol_identity", lambda _series: None)

    dv.register_source(df, seed_fixture_membership=True)

    assert u.members_as_of(date(2024, 1, 2)) == set()


@pytest.mark.determinism("d1")
def test_requested_reserved_field_is_not_added_twice() -> None:
    df = pd.DataFrame(
        {
            "symbol": ["A"],
            "valid_time": [date(2024, 1, 1)],
            "knowledge_time": [date(2024, 1, 1)],
            "close": [100.0],
        }
    )
    dv = DataView()
    dv.register_source(df)

    out = dv.as_of(["A"], ["symbol", "close"], date(2024, 1, 1))

    assert list(out.columns) == ["symbol", "valid_time", "knowledge_time", "close"]


@pytest.mark.determinism("d1")
def test_missing_observation_knowledge_time_falls_back_to_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = pd.DataFrame(
        {
            "symbol": ["A"],
            "valid_time": [date(2024, 1, 1)],
            "knowledge_time": [date(2024, 1, 1)],
            "close": [100.0],
        }
    )
    dv = DataView()
    dv.register_source(df)

    def _fake_resolve_field_value(
        _visible: pd.DataFrame,
        anchor: pd.Series,
        _field: str,
        _fill_policy: FillPolicy,
        _knowledge_date: date,
    ) -> tuple[float, None]:
        return 123.0, None

    monkeypatch.setattr("pysrc.data.dataview._resolve_field_value", _fake_resolve_field_value)

    out = dv.as_of(["A"], ["close"], date(2024, 1, 1))

    assert out["close"].iloc[0] == 123.0
    assert out["knowledge_time"].iloc[0] == date(2024, 1, 1)


@pytest.mark.determinism("d1")
def test_glob_match_applies_when_namespace_shortcut_missing() -> None:
    """Fallback glob matching works even when namespace.* is not present explicitly."""
    from pysrc.data.pit_config import resolve_field_config

    cfg = FieldTTLConfig(
        default_config=ResolvedFieldConfig(42, FillPolicy.REJECT, MissingPolicy.FAIL),
        namespace_configs={
            "*.close": ResolvedFieldConfig(9, FillPolicy.FORWARD, MissingPolicy.WARN),
        },
    )

    resolved = resolve_field_config("price.close", cfg)

    assert resolved.ttl_days == 9
    assert resolved.fill_policy == FillPolicy.FORWARD
    assert resolved.missing_policy == MissingPolicy.WARN

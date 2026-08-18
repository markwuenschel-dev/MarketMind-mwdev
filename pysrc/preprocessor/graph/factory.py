# py/preprocessor/graph/factory.py
from __future__ import annotations

import contextlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from pysrc.core.errors import UnsupportedPlan
from pysrc.ops.mm_logkit import get_logger
from pysrc.preprocessor.graph.graph import Graph
from pysrc.preprocessor.graph.ops import Op
from pysrc.preprocessor.utils.errors import UnsupportedAST

logger = get_logger(__name__)


class _ExprFactoryProxy:
    def build(self, *args, **kwargs):
        from pysrc.preprocessor.utils.expr_builders import ExprFactory as _ExprFactory

        return _ExprFactory.build(*args, **kwargs)


class _TransformFactoryProxy:
    def build(self, *args, **kwargs):
        from pysrc.preprocessor.utils.transforms import TransformFactory as _TransformFactory

        return _TransformFactory.build(*args, **kwargs)


ExprFactory = _ExprFactoryProxy()
TransformFactory = _TransformFactoryProxy()

_OP_REGISTRY: dict[str, type[Op]] = {}
_ALIAS_MAP: dict[str, str] = {}


@dataclass(frozen=True)
class OpSpec:
    name: str
    params: dict[str, Any]


def register(name: str, op_cls: type[Op]) -> None:
    if name in _OP_REGISTRY:
        raise ValueError(f"Op '{name}' already registered")
    _OP_REGISTRY[name] = op_cls


def register_alias(alias: str, target: str) -> None:
    if alias in _OP_REGISTRY or alias in _ALIAS_MAP:
        raise ValueError(f"Alias '{alias}' already used")
    _ALIAS_MAP[alias] = target


def resolve_name(name: str) -> str:
    return _ALIAS_MAP.get(name, name)


def _parse_token(sym: str) -> tuple[str, dict]:
    if ":" not in sym:
        return sym, {}

    name, payload = sym.split(":", 1)
    parts = [p.strip() for p in payload.split(",") if p.strip()]
    if name == "sequence.lags":
        return name, {"n": int(parts[0])}
    if name in ("technical.SMA", "technical.EMA"):
        key = "window" if name == "technical.SMA" else "span"
        return name, {"input_col": parts[0], key: int(parts[1])}
    if name == "technical.RSI":
        if len(parts) == 1:
            return name, {"input_col": parts[0]}
        if len(parts) == 2:
            return name, {"input_col": parts[0], "window": int(parts[1])}
    if name == "technical.MACD_line_signal" and len(parts) == 4:
        return name, {
            "input_col": parts[0],
            "fast": int(parts[1]),
            "slow": int(parts[2]),
            "signal": int(parts[3]),
        }
    if name == "technical.Bollinger":
        if len(parts) == 2:
            return name, {"input_col": parts[0], "window": int(parts[1])}
        if len(parts) == 3:
            return name, {
                "input_col": parts[0],
                "window": int(parts[1]),
                "num_std": float(parts[2]),
            }
    if len(parts) == 1 and parts[0].isdigit():
        return name, {"value": int(parts[0])}
    return name, {}


def _iter_specs(ops: list[str], params_map: dict[str, list[dict[str, Any]]]) -> Iterable[OpSpec]:
    cursors: dict[str, int] = {}
    for sym in ops:
        base_sym, inline_params = _parse_token(sym)
        key = resolve_name(base_sym)
        if key not in _OP_REGISTRY:
            try:
                expr = ExprFactory.build(key)
                inline_params["expr"] = expr
            except UnsupportedAST:
                pass
            try:
                transform = TransformFactory.build(key)
                inline_params["transform"] = transform
            except UnsupportedAST:
                # Support tokenized literal-only shorthands (e.g. custom.literal:7)
                # for parser coverage paths. Truly unknown ops without inline params
                # still fail loudly.
                if not inline_params:
                    raise UnsupportedPlan(f"Unknown op '{sym}'")

        idx = cursors.get(key, 0)
        plist = params_map.get(sym) or params_map.get(base_sym) or params_map.get(key) or []
        spec_params = dict(plist[idx]) if idx < len(plist) else {}
        merged = {**inline_params, **spec_params}

        cursors[key] = idx + 1
        yield OpSpec(name=key, params=merged)


def build_graph(ops: list[str], params: dict[str, list[dict[str, Any]]]) -> Graph:
    g = Graph()
    for spec in _iter_specs(ops, params):
        op_cls = _OP_REGISTRY.get(spec.name)
        if op_cls is None:
            raise UnsupportedPlan(f"Unknown op '{spec.name}'")
        op: Op = op_cls(**spec.params)
        g.add_op(op)
    return g


def registry_snapshot() -> tuple[dict[str, str], dict[str, str]]:
    return (
        {k: v.__name__ for k, v in _OP_REGISTRY.items()},
        dict(_ALIAS_MAP),
    )


def register_builtin_ops() -> None:
    """Register builtin ops. Fails loudly on import errors to avoid silent bugs."""
    from .ops_custom import (
        ATR,
        EMA,
        OBV,
        RSI,
        SMA,
        VWAP,
        Bollinger,
        DataLoadCSV,
        FeatureReturns,
        FeatureRSI,
        FeatureSMA,
        HalfLife,
        IndustryScore,
        Lags,
        MACDLineSignal,
        PairBeta,
        PairSpread,
        ResidualKF,
        ResidualOLS,
        RobustScaler,
        RollingStd,
        RollingVol,
        RollingZ,
        SentimentLexicon,
        VolScale,
        XSecRank,
        ZScore,
    )

    builtin_ops = [
        ("technical.RSI", RSI),
        ("technical.SMA", SMA),
        ("technical.EMA", EMA),
        ("technical.MACD_line_signal", MACDLineSignal),
        ("technical.Bollinger", Bollinger),
        ("technical.ATR", ATR),
        ("technical.OBV", OBV),
        ("technical.VWAP", VWAP),
        ("sequence.lags", Lags),
        ("scaling.zscore", ZScore),
        ("scaling.robust", RobustScaler),
        ("pairs.beta", PairBeta),
        ("pairs.spread", PairSpread),
        ("stats.half_life", HalfLife),
        ("scaling.zscore_roll", RollingZ),
        ("stats.rolling_std", RollingStd),
        ("stats.rolling_vol", RollingVol),
        ("momentum.xsec_rank", XSecRank),
        ("momentum.vol_scale", VolScale),
        ("momentum.residual_ols", ResidualOLS),
        ("momentum.residual_kf", ResidualKF),
        ("momentum.industry_score", IndustryScore),
        ("external.sentiment_lex", SentimentLexicon),
        ("feature.returns", FeatureReturns),
        ("feature.sma", FeatureSMA),
        ("feature.rsi", FeatureRSI),
        ("data.load_csv", DataLoadCSV),
    ]

    for name, cls in builtin_ops:
        with contextlib.suppress(ValueError):
            register(name, cls)

    builtin_aliases = [
        ("RSI", "technical.RSI"),
        ("SMA", "technical.SMA"),
        ("ROLL_MEAN", "technical.SMA"),
        ("ROLL_STD", "stats.rolling_std"),
        ("EMA", "technical.EMA"),
        ("MACD", "technical.MACD_line_signal"),
        ("Z_SCORE", "scaling.zscore_roll"),
    ]

    for alias, target in builtin_aliases:
        with contextlib.suppress(ValueError):
            register_alias(alias, target)

    logger.debug("Registered %d builtin ops and %d aliases", len(builtin_ops), len(builtin_aliases))

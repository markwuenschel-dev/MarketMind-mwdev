# preprocessor/graph/ops_custom.py
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol, cast

from pysrc.preprocessor.graph.expr import register_polars_lowering
from pysrc.preprocessor.graph.ops import ElementwiseOp, RollingOp, ScalingOp, SequenceOp
from pysrc.preprocessor.ops.common.columns import _as_list, _derive_out_names

if TYPE_CHECKING:
    import polars as pl


class _HasContract(Protocol):
    """Protocol for ops that expose requires/provides (e.g. RollingOp via Op)."""

    @property
    def requires(self) -> set[str]: ...

    @property
    def provides(self) -> set[str]: ...


class _ProvidesMixin:
    def _with_contracts(self: _HasContract, ir: dict[str, Any]) -> dict[str, Any]:
        ir.update({"requires": list(self.requires), "provides": list(self.provides)})
        return ir


# --------- TECHNICAL / ROLLING OPS ---------


class RSI(_ProvidesMixin, RollingOp):
    NAME = "technical.RSI"

    def validate_params(self) -> None:
        input_col = self.params.get("input_col", "close")
        window = self.params.get("window", 14)
        out_col = self.params.get("out_col", "rsi")
        if not isinstance(input_col, str):
            raise ValueError("RSI.input_col must be a string")
        if not isinstance(window, int) or window <= 1:
            raise ValueError("RSI.window must be an int > 1")
        if not isinstance(out_col, str):
            raise ValueError("RSI.out_col must be a string")
        self.params.setdefault("input_col", input_col)
        self.params.setdefault("window", window)
        self.params.setdefault("out_col", out_col)

    def _compute_requires(self) -> set[str]:
        return {self.params["input_col"]}

    def _compute_provides(self) -> set[str]:
        return {self.params["out_col"]}

    def to_ir(self) -> dict[str, Any]:
        return self._with_contracts(super().to_ir())


class SMA(_ProvidesMixin, RollingOp):
    NAME = "technical.SMA"

    def validate_params(self) -> None:
        input_col = self.params.get("input_col")
        window = self.params.get("window")
        if not isinstance(input_col, str):
            raise ValueError("SMA.input_col (str) is required")
        if not isinstance(window, int) or window <= 1:
            raise ValueError("SMA.window must be an int > 1")
        self.params.setdefault("out_col", f"{input_col}_sma{window}")

    def _compute_requires(self) -> set[str]:
        return {self.params["input_col"]}

    def _compute_provides(self) -> set[str]:
        return {self.params["out_col"]}

    def to_ir(self) -> dict[str, Any]:
        return self._with_contracts(super().to_ir())


class EMA(_ProvidesMixin, RollingOp):
    NAME = "technical.EMA"

    def validate_params(self) -> None:
        input_col = self.params.get("input_col")
        span = self.params.get("span")
        if not isinstance(input_col, str):
            raise ValueError("EMA.input_col (str) is required")
        if not isinstance(span, int) or span <= 1:
            raise ValueError("EMA.span must be an int > 1")
        self.params.setdefault("adjust", False)
        self.params.setdefault("out_col", f"{input_col}_ema{span}")

    def _compute_requires(self) -> set[str]:
        return {self.params["input_col"]}

    def _compute_provides(self) -> set[str]:
        return {self.params["out_col"]}

    def to_ir(self) -> dict[str, Any]:
        return self._with_contracts(super().to_ir())


class MACDLineSignal(_ProvidesMixin, RollingOp):
    NAME = "technical.MACD_line_signal"

    def validate_params(self) -> None:
        input_col = self.params.get("input_col", "close")
        fast = self.params.get("fast", 12)
        slow = self.params.get("slow", 26)
        signal = self.params.get("signal", 9)
        if not isinstance(input_col, str):
            raise ValueError("MACD.input_col must be a string")
        if not all(isinstance(v, int) and v > 0 for v in (fast, slow, signal)):
            raise ValueError("MACD fast/slow/signal must be positive integers")
        if fast >= slow:
            raise ValueError("MACD fast must be < slow")
        self.params.setdefault("input_col", input_col)
        self.params.setdefault("fast", fast)
        self.params.setdefault("slow", slow)
        self.params.setdefault("signal", signal)
        self.params.setdefault("out_fast", f"{input_col}_ema{fast}")
        self.params.setdefault("out_slow", f"{input_col}_ema{slow}")
        self.params.setdefault("out_macd", "macd")
        self.params.setdefault("out_signal", "macd_signal")
        self.params.setdefault("out_hist", "macd_hist")

    def _compute_requires(self) -> set[str]:
        return {self.params["input_col"]}

    def _compute_provides(self) -> set[str]:
        return {
            self.params["out_fast"],
            self.params["out_slow"],
            self.params["out_macd"],
            self.params["out_signal"],
            self.params["out_hist"],
        }

    def to_ir(self) -> dict[str, Any]:
        return self._with_contracts(super().to_ir())


class Bollinger(_ProvidesMixin, RollingOp):
    NAME = "technical.Bollinger"

    def validate_params(self) -> None:
        input_col = self.params.get("input_col", "close")
        window = self.params.get("window", 20)
        num_std = self.params.get("num_std", 2.0)
        if not isinstance(input_col, str):
            raise ValueError("Bollinger.input_col must be a string")
        if not isinstance(window, int) or window <= 1:
            raise ValueError("Bollinger.window must be an int > 1")
        if not isinstance(num_std, (int, float)) or float(num_std) <= 0.0:
            raise ValueError("Bollinger.num_std must be positive")
        self.params.setdefault("input_col", input_col)
        self.params.setdefault("window", window)
        self.params.setdefault("num_std", float(num_std))
        self.params.setdefault("out_mid", f"{input_col}_sma{window}")
        self.params.setdefault("out_std", f"{input_col}_std{window}")
        self.params.setdefault("out_upper", f"{input_col}_bb_upper{window}")
        self.params.setdefault("out_lower", f"{input_col}_bb_lower{window}")

    def _compute_requires(self) -> set[str]:
        return {self.params["input_col"]}

    def _compute_provides(self) -> set[str]:
        return {
            self.params["out_mid"],
            self.params["out_std"],
            self.params["out_upper"],
            self.params["out_lower"],
        }

    def to_ir(self) -> dict[str, Any]:
        return self._with_contracts(super().to_ir())


class ATR(_ProvidesMixin, RollingOp):
    NAME = "technical.ATR"

    def validate_params(self) -> None:
        high_col = self.params.get("high_col", "high")
        low_col = self.params.get("low_col", "low")
        close_col = self.params.get("close_col", "close")
        window = self.params.get("window", 14)
        if not all(isinstance(c, str) for c in (high_col, low_col, close_col)):
            raise ValueError("ATR high_col/low_col/close_col must be strings")
        if not isinstance(window, int) or window <= 1:
            raise ValueError("ATR.window must be an int > 1")
        self.params.setdefault("high_col", high_col)
        self.params.setdefault("low_col", low_col)
        self.params.setdefault("close_col", close_col)
        self.params.setdefault("window", window)
        self.params.setdefault("out_col", f"atr_{window}")

    def _compute_requires(self) -> set[str]:
        return {self.params["high_col"], self.params["low_col"], self.params["close_col"]}

    def _compute_provides(self) -> set[str]:
        return {self.params["out_col"]}

    def to_ir(self) -> dict[str, Any]:
        return self._with_contracts(super().to_ir())


class OBV(_ProvidesMixin, ElementwiseOp):
    NAME = "technical.OBV"

    def validate_params(self) -> None:
        input_col = self.params.get("input_col", "close")
        volume_col = self.params.get("volume_col", "volume")
        if not isinstance(input_col, str) or not isinstance(volume_col, str):
            raise ValueError("OBV input_col and volume_col must be strings")
        self.params.setdefault("input_col", input_col)
        self.params.setdefault("volume_col", volume_col)
        self.params.setdefault("out_col", "obv")

    def _compute_requires(self) -> set[str]:
        return {self.params["input_col"], self.params["volume_col"]}

    def _compute_provides(self) -> set[str]:
        return {self.params["out_col"]}

    def to_ir(self) -> dict[str, Any]:
        return self._with_contracts(super().to_ir())


class VWAP(_ProvidesMixin, ElementwiseOp):
    NAME = "technical.VWAP"

    def validate_params(self) -> None:
        price_col = self.params.get("price_col", "close")
        volume_col = self.params.get("volume_col", "volume")
        session_col = self.params.get("session_col")
        timestamp_col = self.params.get("timestamp_col")
        if not isinstance(price_col, str) or not isinstance(volume_col, str):
            raise ValueError("VWAP price_col and volume_col must be strings")
        if session_col is not None and not isinstance(session_col, str):
            raise ValueError("VWAP.session_col must be a string when provided")
        if timestamp_col is not None and not isinstance(timestamp_col, str):
            raise ValueError("VWAP.timestamp_col must be a string when provided")
        self.params.setdefault("price_col", price_col)
        self.params.setdefault("volume_col", volume_col)
        self.params.setdefault("session_col", session_col)
        self.params.setdefault("timestamp_col", timestamp_col)
        self.params.setdefault("out_col", "vwap")

    def _compute_requires(self) -> set[str]:
        reqs = {self.params["price_col"], self.params["volume_col"]}
        session_col = self.params.get("session_col")
        timestamp_col = self.params.get("timestamp_col")
        if isinstance(session_col, str):
            reqs.add(session_col)
        if isinstance(timestamp_col, str):
            reqs.add(timestamp_col)
        return reqs

    def _compute_provides(self) -> set[str]:
        return {self.params["out_col"]}

    def to_ir(self) -> dict[str, Any]:
        return self._with_contracts(super().to_ir())


class RollingStd(_ProvidesMixin, RollingOp):
    NAME = "stats.rolling_std"

    def validate_params(self) -> None:
        col = self.params.get("col") or self.params.get("input_col")
        window = self.params.get("window", 20)
        min_samples = self.params.get("min_samples", window)
        if not isinstance(col, str):
            raise ValueError("RollingStd.col (str) is required")
        if not isinstance(window, int) or window <= 1:
            raise ValueError("RollingStd.window must be an int > 1")
        if not isinstance(min_samples, int) or min_samples <= 0:
            raise ValueError("RollingStd.min_samples must be a positive int")
        self.params["col"] = col
        self.params["window"] = window
        self.params["min_samples"] = min_samples
        self.params.setdefault("out_col", f"{col}_std{window}")

    def _compute_requires(self) -> set[str]:
        return {self.params["col"]}

    def _compute_provides(self) -> set[str]:
        return {self.params["out_col"]}

    def to_ir(self) -> dict[str, Any]:
        return self._with_contracts(super().to_ir())


# --------- SEQUENCE OPS ---------


class Lags(SequenceOp):
    NAME = "sequence.lags"

    def validate_params(self) -> None:
        cols = _as_list(self.params.get("cols"))
        n = self.params.get("n")
        out_cols = _as_list(self.params.get("out_cols"))
        if not cols:
            raise ValueError("Lags.cols required (str or list[str])")
        if not isinstance(n, int) or n <= 0:
            raise ValueError("Lags.n must be positive int")
        if out_cols and len(out_cols) != len(cols) * n:
            raise ValueError("out_cols must match cols * n if provided")
        self.params["cols"] = cols
        self.params["n"] = n
        self.params["out_cols"] = out_cols or [f"{c}_lag{i}" for c in cols for i in range(1, n + 1)]

    def _compute_requires(self) -> set[str]:
        return set(self.params["cols"])

    def _compute_provides(self) -> set[str]:
        return set(self.params["out_cols"])


# --------- SCALING OPS ---------


class ZScore(ScalingOp):
    NAME = "scaling.zscore"

    def validate_params(self) -> None:
        cols = _as_list(self.params.get("cols"))
        out_cols = _as_list(self.params.get("out_cols"))
        if not cols:
            raise ValueError("ZScore.cols required")
        if out_cols and len(out_cols) != len(cols):
            raise ValueError("out_cols must match cols if provided")
        self.params["cols"] = cols
        self.params["out_cols"] = _derive_out_names(cols, "z", out_cols)

    def _compute_requires(self) -> set[str]:
        return set(self.params["cols"])

    def _compute_provides(self) -> set[str]:
        return set(self.params["out_cols"])

    def state_dict(self) -> dict[str, Any]:
        return {"means": self.params.get("means", []), "stds": self.params.get("stds", [])}


class RobustScaler(ScalingOp):
    NAME = "scaling.robust"

    def validate_params(self) -> None:
        input_col = self.params.get("input_col")
        out_col = self.params.get("out_col")
        cols = self.params.get("cols")
        out_cols = self.params.get("out_cols")

        if input_col:
            cols = [input_col]
            out_cols = [out_col] if out_col else [f"{input_col}_robust"]
        elif cols:
            cols = _as_list(cols)
            out_cols = _as_list(out_cols) if out_cols else [f"{c}_robust" for c in cols]
        else:
            raise ValueError("RobustScaler requires either 'input_col' or 'cols' parameter")

        if out_cols and len(out_cols) != len(cols):
            raise ValueError("out_cols must match cols if provided")

        self.params["cols"] = cols
        self.params["out_cols"] = out_cols
        self.params.setdefault("quantile_range", (25, 75))
        self.params.setdefault("with_centering", True)
        self.params.setdefault("with_scaling", True)

    def _compute_requires(self) -> set[str]:
        return set(self.params["cols"])

    def _compute_provides(self) -> set[str]:
        return set(self.params["out_cols"])

    def state_dict(self) -> dict[str, Any]:
        return {}


# --------- EXTERNAL OPS ---------


class SentimentLexicon(ElementwiseOp):
    NAME = "external.sentiment_lex"

    def validate_params(self) -> None:
        text_col = self.params.get("text_col")
        if not isinstance(text_col, str):
            raise ValueError("SentimentLexicon.text_col (str) is required")
        self.params.setdefault("out_col", "sentiment")

    def _compute_requires(self) -> set[str]:
        return {self.params["text_col"]}

    def _compute_provides(self) -> set[str]:
        return {self.params["out_col"]}


# --------- PAIRS / STATS OPS ---------


class PairBeta(RollingOp):
    NAME = "pairs.beta"

    def validate_params(self) -> None:
        a = self.params.get("a")
        b = self.params.get("b")
        self.params.setdefault("method", "ols")
        self.params.setdefault("out_col", f"beta_{a}_{b}")

    def _compute_requires(self) -> set[str]:
        return {f"{self.params['a']}.close", f"{self.params['b']}.close"}

    def _compute_provides(self) -> set[str]:
        return {self.params["out_col"]}


class PairSpread(ElementwiseOp):
    NAME = "pairs.spread"

    def validate_params(self) -> None:
        a = self.params.get("a")
        b = self.params.get("b")
        beta = self.params.get("beta_col")
        self.params.setdefault("out_col", f"spread_{a}_{b}")
        if beta is None:
            self.params["beta_col"] = f"beta_{a}_{b}"

    def _compute_requires(self) -> set[str]:
        return {f"{self.params['a']}.close", f"{self.params['b']}.close", self.params["beta_col"]}

    def _compute_provides(self) -> set[str]:
        return {self.params["out_col"]}


class HalfLife(RollingOp):
    NAME = "stats.half_life"

    def validate_params(self) -> None:
        col = self.params.get("col")
        self.params.setdefault("out_col", f"hl_{col}")

    def _compute_requires(self) -> set[str]:
        return {self.params["col"]}

    def _compute_provides(self) -> set[str]:
        return {self.params["out_col"]}


class RollingZ(_ProvidesMixin, ScalingOp):
    NAME = "scaling.zscore_roll"

    def validate_params(self) -> None:
        col = self.params.get("col")
        w = int(self.params.get("window", 256))
        min_samples = int(self.params.get("min_samples", w))
        if not isinstance(col, str):
            raise ValueError("RollingZ.col must be a string")
        if w <= 1:
            raise ValueError("RollingZ.window must be > 1")
        if min_samples <= 0:
            raise ValueError("RollingZ.min_samples must be positive")
        self.params["window"] = w
        self.params["min_samples"] = min_samples
        self.params.setdefault("out_col", f"{col}_z{w}")

    def _compute_requires(self) -> set[str]:
        return {self.params["col"]}

    def _compute_provides(self) -> set[str]:
        return {self.params["out_col"]}

    def to_ir(self) -> dict[str, Any]:
        return self._with_contracts(super().to_ir())


class RollingVol(RollingOp):
    NAME = "stats.rolling_vol"

    def validate_params(self) -> None:
        col = self.params.get("col")
        w = int(self.params.get("window", 64))
        self.params["window"] = w
        self.params.setdefault("out_col", f"vol_{col}_{w}")

    def _compute_requires(self) -> set[str]:
        return {self.params["col"]}

    def _compute_provides(self) -> set[str]:
        return {self.params["out_col"]}


class XSecRank(_ProvidesMixin, RollingOp):
    NAME = "momentum.xsec_rank"

    def validate_params(self) -> None:
        col = self.params.get("col") or self.params.get("input_col")
        window = self.params.get("window")
        skip = self.params.get("skip", 21)
        out_col = self.params.get("out_col", "mom_rank")
        if not isinstance(col, str):
            raise ValueError("XSecRank.col (str) is required")
        if not isinstance(window, int) or window <= 0:
            raise ValueError("XSecRank.window must be a positive int")
        if not isinstance(skip, int) or skip < 0:
            raise ValueError("XSecRank.skip must be a non-negative int")
        if not isinstance(out_col, str):
            raise ValueError("XSecRank.out_col must be a string")
        self.params["col"] = col
        self.params["window"] = window
        self.params["skip"] = skip
        self.params["out_col"] = out_col

    def _compute_requires(self) -> set[str]:
        return {self.params["col"]}

    def _compute_provides(self) -> set[str]:
        return {self.params["out_col"]}

    def to_ir(self) -> dict[str, Any]:
        return self._with_contracts(super().to_ir())


class VolScale(_ProvidesMixin, RollingOp):
    NAME = "momentum.vol_scale"

    def validate_params(self) -> None:
        signal_col = self.params.get("signal_col")
        vol_col = self.params.get("vol_col")
        target_vol = self.params.get("target_vol", 0.15)
        max_leverage = self.params.get("max_leverage", 2.0)
        out_col = self.params.get("out_col", "mom_scaled")
        if not isinstance(signal_col, str):
            raise ValueError("VolScale.signal_col must be a string")
        if not isinstance(vol_col, str):
            raise ValueError("VolScale.vol_col must be a string")
        if not isinstance(target_vol, (int, float)) or float(target_vol) <= 0.0:
            raise ValueError("VolScale.target_vol must be positive")
        if not isinstance(max_leverage, (int, float)) or float(max_leverage) <= 0.0:
            raise ValueError("VolScale.max_leverage must be positive")
        if not isinstance(out_col, str):
            raise ValueError("VolScale.out_col must be a string")
        self.params["signal_col"] = signal_col
        self.params["vol_col"] = vol_col
        self.params["target_vol"] = float(target_vol)
        self.params["max_leverage"] = float(max_leverage)
        self.params["out_col"] = out_col

    def _compute_requires(self) -> set[str]:
        return {self.params["signal_col"], self.params["vol_col"]}

    def _compute_provides(self) -> set[str]:
        return {self.params["out_col"]}

    def to_ir(self) -> dict[str, Any]:
        return self._with_contracts(super().to_ir())


class ResidualOLS(_ProvidesMixin, RollingOp):
    NAME = "momentum.residual_ols"

    def validate_params(self) -> None:
        asset_ret_col = self.params.get("asset_ret_col")
        factor_ret_cols = self.params.get("factor_ret_cols")
        window = self.params.get("window")
        out_col = self.params.get("out_col", "residual_ols")
        if not isinstance(asset_ret_col, str):
            raise ValueError("ResidualOLS.asset_ret_col must be a string")
        if (
            not isinstance(factor_ret_cols, list)
            or not factor_ret_cols
            or not all(isinstance(col, str) for col in factor_ret_cols)
        ):
            raise ValueError("ResidualOLS.factor_ret_cols must be a non-empty list[str]")
        if not isinstance(window, int) or window <= 0:
            raise ValueError("ResidualOLS.window must be a positive int")
        if not isinstance(out_col, str):
            raise ValueError("ResidualOLS.out_col must be a string")
        self.params["asset_ret_col"] = asset_ret_col
        self.params["factor_ret_cols"] = factor_ret_cols
        self.params["window"] = window
        self.params["out_col"] = out_col

    def _compute_requires(self) -> set[str]:
        return {self.params["asset_ret_col"], *self.params["factor_ret_cols"]}

    def _compute_provides(self) -> set[str]:
        return {self.params["out_col"]}

    def to_ir(self) -> dict[str, Any]:
        return self._with_contracts(super().to_ir())


class ResidualKF(_ProvidesMixin, RollingOp):
    NAME = "momentum.residual_kf"

    def validate_params(self) -> None:
        asset_ret_col = self.params.get("asset_ret_col")
        factor_ret_cols = self.params.get("factor_ret_cols")
        process_noise = self.params.get("process_noise")
        obs_noise = self.params.get("obs_noise")
        out_col = self.params.get("out_col", "residual_kf")
        if not isinstance(asset_ret_col, str):
            raise ValueError("ResidualKF.asset_ret_col must be a string")
        if (
            not isinstance(factor_ret_cols, list)
            or not factor_ret_cols
            or not all(isinstance(col, str) for col in factor_ret_cols)
        ):
            raise ValueError("ResidualKF.factor_ret_cols must be a non-empty list[str]")
        if not isinstance(process_noise, (int, float)) or float(process_noise) <= 0.0:
            raise ValueError("ResidualKF.process_noise must be positive")
        if not isinstance(obs_noise, (int, float)) or float(obs_noise) <= 0.0:
            raise ValueError("ResidualKF.obs_noise must be positive")
        if not isinstance(out_col, str):
            raise ValueError("ResidualKF.out_col must be a string")
        self.params["asset_ret_col"] = asset_ret_col
        self.params["factor_ret_cols"] = factor_ret_cols
        self.params["process_noise"] = float(process_noise)
        self.params["obs_noise"] = float(obs_noise)
        self.params["out_col"] = out_col

    def _compute_requires(self) -> set[str]:
        return {self.params["asset_ret_col"], *self.params["factor_ret_cols"]}

    def _compute_provides(self) -> set[str]:
        return {self.params["out_col"]}

    def to_ir(self) -> dict[str, Any]:
        return self._with_contracts(super().to_ir())


class IndustryScore(_ProvidesMixin, RollingOp):
    NAME = "momentum.industry_score"

    def validate_params(self) -> None:
        ret_col = self.params.get("ret_col")
        sector_col = self.params.get("sector_col")
        window = self.params.get("window")
        out_col = self.params.get("out_col", "industry_score")
        if not isinstance(ret_col, str):
            raise ValueError("IndustryScore.ret_col must be a string")
        if not isinstance(sector_col, str):
            raise ValueError("IndustryScore.sector_col must be a string")
        if not isinstance(window, int) or window <= 0:
            raise ValueError("IndustryScore.window must be a positive int")
        if not isinstance(out_col, str):
            raise ValueError("IndustryScore.out_col must be a string")
        self.params["ret_col"] = ret_col
        self.params["sector_col"] = sector_col
        self.params["window"] = window
        self.params["out_col"] = out_col

    def _compute_requires(self) -> set[str]:
        return {self.params["ret_col"], self.params["sector_col"]}

    def _compute_provides(self) -> set[str]:
        return {self.params["out_col"]}

    def to_ir(self) -> dict[str, Any]:
        return self._with_contracts(super().to_ir())


# --------- POLARS LOWERINGS ---------


def _delegate_to_backend(fn_name: str) -> Callable[..., Any]:
    def _lowering(*args: Any, **kwargs: Any) -> Any:
        import importlib

        polars_backend = importlib.import_module("pysrc.preprocessor.graph.backends.polars")

        fn = getattr(polars_backend, fn_name)
        return fn(*args, **kwargs)

    return _lowering


def lower_pairs_beta_polars(
    ir: dict[str, Any],
    lf: pl.LazyFrame | pl.DataFrame,
    *,
    group_by: Any = None,
) -> pl.LazyFrame | pl.DataFrame:
    # Rolling OLS hedge ratio: regress a.close on b.close over a fixed window.
    import numpy as np
    import polars as pl

    from pysrc.preprocessor.graph.backends.polars import _apply_eager

    params = ir["params"]
    a = params["a"]
    b = params["b"]
    window = int(params.get("window", params.get("beta_window", 60)))
    out_col = params.get("out_col", f"beta_{a}_{b}")

    a_col = f"{a}.close"
    b_col = f"{b}.close"

    def _lower(frame: pl.DataFrame) -> pl.DataFrame:
        if a_col not in frame.columns or b_col not in frame.columns:
            missing = [c for c in (a_col, b_col) if c not in frame.columns]
            raise ValueError(f"pairs.beta requires columns {missing}")
        x = frame.get_column(b_col).to_numpy().astype(float)
        y = frame.get_column(a_col).to_numpy().astype(float)
        n = len(frame)
        beta = np.full(n, np.nan, dtype=float)
        if window <= 1 or n == 0:
            return frame.with_columns(pl.Series(out_col, beta))

        cumsum_x = np.cumsum(x)
        cumsum_y = np.cumsum(y)
        cumsum_x2 = np.cumsum(x * x)
        cumsum_xy = np.cumsum(x * y)

        for i in range(window - 1, n):
            start = i - window + 1
            sum_x = cumsum_x[i] - (cumsum_x[start - 1] if start > 0 else 0.0)
            sum_y = cumsum_y[i] - (cumsum_y[start - 1] if start > 0 else 0.0)
            sum_x2 = cumsum_x2[i] - (cumsum_x2[start - 1] if start > 0 else 0.0)
            sum_xy = cumsum_xy[i] - (cumsum_xy[start - 1] if start > 0 else 0.0)
            w = window
            denom = (w * sum_x2) - (sum_x * sum_x)
            if denom == 0.0:
                beta[i] = np.nan
            else:
                beta[i] = ((w * sum_xy) - (sum_x * sum_y)) / denom

        return frame.with_columns(pl.Series(out_col, beta))

    return cast(pl.LazyFrame | pl.DataFrame, _apply_eager(lf, _lower))


def lower_pairs_spread_polars(
    ir: dict[str, Any],
    lf: pl.LazyFrame | pl.DataFrame,
    *,
    group_by: Any = None,
) -> pl.LazyFrame | pl.DataFrame:
    import numpy as np
    import polars as pl

    from pysrc.preprocessor.graph.backends.polars import _apply_eager

    params = ir["params"]
    a = params["a"]
    b = params["b"]
    beta_col = params.get("beta_col") or f"beta_{a}_{b}"
    out_col = params.get("out_col", f"spread_{a}_{b}")

    a_col = f"{a}.close"
    b_col = f"{b}.close"

    def _lower(frame: pl.DataFrame) -> pl.DataFrame:
        missing = [c for c in (a_col, b_col, beta_col) if c not in frame.columns]
        if missing:
            raise ValueError(f"pairs.spread requires columns {missing}")
        a_vals = frame.get_column(a_col).to_numpy().astype(float)
        b_vals = frame.get_column(b_col).to_numpy().astype(float)
        beta_vals = frame.get_column(beta_col).to_numpy().astype(float)
        spread = a_vals - beta_vals * b_vals
        spread[~np.isfinite(beta_vals)] = np.nan
        return frame.with_columns(pl.Series(out_col, spread))

    return cast(pl.LazyFrame | pl.DataFrame, _apply_eager(lf, _lower))


def lower_half_life_polars(
    ir: dict[str, Any],
    lf: pl.LazyFrame | pl.DataFrame,
    *,
    group_by: Any = None,
) -> pl.LazyFrame | pl.DataFrame:
    # OU-style half-life via AR(1) regression over the configured window.
    import numpy as np
    import polars as pl

    from pysrc.preprocessor.graph.backends.polars import _apply_eager

    params = ir["params"]
    col = params["col"]
    window = int(params.get("window", params.get("half_life_window", 60)))
    out_col = params.get("out_col", f"hl_{col}")

    def _lower(frame: pl.DataFrame) -> pl.DataFrame:
        if col not in frame.columns:
            raise ValueError(f"stats.half_life requires column '{col}'")
        x = frame.get_column(col).to_numpy().astype(float)
        n = len(frame)
        hl = np.full(n, np.nan, dtype=float)
        if window <= 2 or n <= 2:
            return frame.with_columns(pl.Series(out_col, hl))

        for i in range(window, n):
            start = i - window
            end = i
            y = x[start + 1 : end + 1]
            x_lag = x[start:end]
            mask = np.isfinite(y) & np.isfinite(x_lag)
            if mask.sum() < 3:
                continue
            y_valid = y[mask]
            x_valid = x_lag[mask]
            x_mean = x_valid.mean()
            y_mean = y_valid.mean()
            num = ((x_valid - x_mean) * (y_valid - y_mean)).sum()
            den = ((x_valid - x_mean) ** 2).sum()
            if den == 0.0:
                continue
            phi = num / den
            if not np.isfinite(phi) or phi <= 0.0 or phi >= 1.0:
                continue
            hl[i] = -np.log(2.0) / np.log(phi)
            if np.isfinite(hl[i]) and not (1.0 <= hl[i] <= 252.0):
                hl[i] = np.nan

        return frame.with_columns(pl.Series(out_col, hl))

    return cast(pl.LazyFrame | pl.DataFrame, _apply_eager(lf, _lower))


def lower_rolling_vol_polars(*args: Any, **kwargs: Any) -> pl.LazyFrame | pl.DataFrame:
    raise NotImplementedError("stats.rolling_vol polars lowering not implemented")


def lower_xsec_rank_polars(
    ir: dict[str, Any],
    lf: pl.LazyFrame | pl.DataFrame,
    *,
    group_by: Any = None,
) -> pl.LazyFrame | pl.DataFrame:
    import numpy as np
    import pandas as pd
    import polars as pl

    from pysrc.preprocessor.graph.backends.polars import _apply_eager

    params = ir["params"]
    col = params["col"]
    window = int(params["window"])
    skip = int(params.get("skip", 21))
    out_col = params.get("out_col", "mom_rank")
    date_col = params.get("date_col")
    asset_col = params.get("asset_col")

    def _pick_column(frame: pl.DataFrame, explicit: Any, candidates: tuple[str, ...]) -> str | None:
        if isinstance(explicit, str):
            return explicit if explicit in frame.columns else None
        for candidate in candidates:
            if candidate in frame.columns:
                return candidate
        return None

    def _cross_sectional_rank(series: pd.Series) -> pd.Series:
        valid = series.dropna()
        if len(valid) < 2:
            return pd.Series(np.nan, index=series.index, dtype=float)
        return series.rank(method="average", pct=True)

    def _lower(frame: pl.DataFrame) -> pl.DataFrame:
        if col not in frame.columns:
            raise ValueError(f"momentum.xsec_rank requires column '{col}'")

        resolved_date_col = _pick_column(
            frame,
            date_col,
            ("date", "datetime", "timestamp", "as_of", "valid_time"),
        )
        resolved_asset_col = _pick_column(
            frame,
            asset_col,
            ("asset", "symbol", "ticker", "sid", "instrument"),
        )
        if resolved_date_col is None or resolved_asset_col is None:
            return frame.with_columns(pl.Series(out_col, np.full(len(frame), np.nan, dtype=float)))

        pdf = frame.to_pandas().copy()
        pdf["_row_order"] = np.arange(len(pdf), dtype=int)
        pdf = pdf.sort_values([resolved_asset_col, resolved_date_col, "_row_order"], kind="stable")

        lagged = (
            pdf.groupby(resolved_asset_col, sort=False)[col].shift(skip) if skip > 0 else pdf[col]
        )
        pdf["_lagged_signal"] = pd.Series(lagged, index=pdf.index, dtype=float)
        pdf["_lookback_signal"] = pdf.groupby(resolved_asset_col, sort=False)[
            "_lagged_signal"
        ].transform(lambda values: values.rolling(window=window, min_periods=window).sum())
        pdf[out_col] = pdf.groupby(resolved_date_col, sort=False)["_lookback_signal"].transform(
            _cross_sectional_rank
        )
        pdf = pdf.sort_values("_row_order", kind="stable")
        return frame.with_columns(
            pl.Series(out_col, pdf[out_col].to_numpy(dtype=float, copy=False))
        )

    return cast(pl.LazyFrame | pl.DataFrame, _apply_eager(lf, _lower))


def lower_vol_scale_polars(
    ir: dict[str, Any],
    lf: pl.LazyFrame | pl.DataFrame,
    *,
    group_by: Any = None,
) -> pl.LazyFrame | pl.DataFrame:
    import math

    import numpy as np
    import polars as pl

    from pysrc.preprocessor.graph.backends.polars import _apply_eager

    params = ir["params"]
    signal_col = params["signal_col"]
    vol_col = params["vol_col"]
    target_vol = float(params.get("target_vol", 0.15))
    max_leverage = float(params.get("max_leverage", 2.0))
    out_col = params.get("out_col", "mom_scaled")
    annualizer = math.sqrt(252.0)

    def _lower(frame: pl.DataFrame) -> pl.DataFrame:
        missing = [name for name in (signal_col, vol_col) if name not in frame.columns]
        if missing:
            raise ValueError(f"momentum.vol_scale requires columns {missing}")
        signal = frame.get_column(signal_col).to_numpy().astype(float)
        daily_std = frame.get_column(vol_col).to_numpy().astype(float)
        annualized = daily_std * annualizer
        fallback = np.full(len(frame), max_leverage, dtype=float)
        valid = np.isfinite(annualized) & (annualized > 0.0)
        scaled = fallback.copy()
        scaled[valid] = np.minimum(target_vol / annualized[valid], max_leverage)
        out = scaled * signal
        return frame.with_columns(pl.Series(out_col, out))

    return cast(pl.LazyFrame | pl.DataFrame, _apply_eager(lf, _lower))


def lower_residual_ols_polars(
    ir: dict[str, Any],
    lf: pl.LazyFrame | pl.DataFrame,
    *,
    group_by: Any = None,
) -> pl.LazyFrame | pl.DataFrame:
    import numpy as np
    import polars as pl

    from pysrc.preprocessor.graph.backends.polars import _apply_eager

    params = ir["params"]
    asset_ret_col = params["asset_ret_col"]
    factor_ret_cols = list(params["factor_ret_cols"])
    window = int(params["window"])
    out_col = params.get("out_col", "residual_ols")

    def _lower(frame: pl.DataFrame) -> pl.DataFrame:
        missing = [name for name in (asset_ret_col, *factor_ret_cols) if name not in frame.columns]
        if missing:
            raise ValueError(f"momentum.residual_ols requires columns {missing}")
        y_all = frame.get_column(asset_ret_col).to_numpy().astype(float)
        x_all = np.column_stack(
            [frame.get_column(name).to_numpy().astype(float) for name in factor_ret_cols]
        )
        residuals = np.full(len(frame), np.nan, dtype=float)
        for idx in range(window - 1, len(frame)):
            start = idx - window + 1
            y_window = y_all[start : idx + 1]
            x_window = x_all[start : idx + 1]
            mask = np.isfinite(y_window) & np.isfinite(x_window).all(axis=1)
            if mask.sum() <= x_window.shape[1]:
                continue
            y_valid = y_window[mask]
            x_valid = x_window[mask]
            design = np.column_stack([np.ones(len(x_valid), dtype=float), x_valid])
            beta, *_ = np.linalg.lstsq(design, y_valid, rcond=None)
            current_x = np.concatenate(([1.0], x_all[idx]))
            if not np.isfinite(current_x).all():
                continue
            residuals[idx] = y_all[idx] - float(current_x @ beta)
        return frame.with_columns(pl.Series(out_col, residuals))

    return cast(pl.LazyFrame | pl.DataFrame, _apply_eager(lf, _lower))


def lower_residual_kf_polars(
    ir: dict[str, Any],
    lf: pl.LazyFrame | pl.DataFrame | None,
    *,
    group_by: Any = None,
) -> pl.LazyFrame | pl.DataFrame:
    raise NotImplementedError(
        "momentum.residual_kf Polars lowering: Kalman residualization deferred to Phase III. See OI-MOM-005."
    )


def lower_industry_score_polars(
    ir: dict[str, Any],
    lf: pl.LazyFrame | pl.DataFrame,
    *,
    group_by: Any = None,
) -> pl.LazyFrame | pl.DataFrame:
    raise NotImplementedError(
        "momentum.industry_score Polars lowering is a Phase II stub. See OI-MOM-004."
    )


lower_rsi_polars = _delegate_to_backend("technical_rsi_polars")
lower_sma_polars = _delegate_to_backend("technical_sma_polars")
lower_ema_polars = _delegate_to_backend("technical_ema_polars")
lower_macd_line_signal_polars = _delegate_to_backend("technical_macd_line_signal_polars")
lower_bollinger_polars = _delegate_to_backend("technical_bollinger_polars")
lower_atr_polars = _delegate_to_backend("technical_atr_polars")
lower_obv_polars = _delegate_to_backend("technical_obv_polars")
lower_vwap_polars = _delegate_to_backend("technical_vwap_polars")
lower_zscore_roll_polars = _delegate_to_backend("scaling_zscore_roll_polars")
lower_rolling_std_polars = _delegate_to_backend("stats_rolling_std_polars")

register_polars_lowering("technical.RSI", lower_rsi_polars)
register_polars_lowering("technical.SMA", lower_sma_polars)
register_polars_lowering("technical.EMA", lower_ema_polars)
register_polars_lowering("technical.MACD_line_signal", lower_macd_line_signal_polars)
register_polars_lowering("technical.Bollinger", lower_bollinger_polars)
register_polars_lowering("technical.ATR", lower_atr_polars)
register_polars_lowering("technical.OBV", lower_obv_polars)
register_polars_lowering("technical.VWAP", lower_vwap_polars)
register_polars_lowering("scaling.zscore_roll", lower_zscore_roll_polars)
register_polars_lowering("stats.rolling_std", lower_rolling_std_polars)
register_polars_lowering("pairs.beta", lower_pairs_beta_polars)
register_polars_lowering("pairs.spread", lower_pairs_spread_polars)
register_polars_lowering("stats.half_life", lower_half_life_polars)
register_polars_lowering("stats.rolling_vol", lower_rolling_vol_polars)
register_polars_lowering("momentum.xsec_rank", lower_xsec_rank_polars)
register_polars_lowering("momentum.vol_scale", lower_vol_scale_polars)
register_polars_lowering("momentum.residual_ols", lower_residual_ols_polars)
register_polars_lowering("momentum.residual_kf", lower_residual_kf_polars)
register_polars_lowering("momentum.industry_score", lower_industry_score_polars)


# --------- FEATURE OPS (Phase 0 wiring) ---------


class FeatureReturns(ElementwiseOp):
    """Compute daily returns. Output column: ``returns``."""

    NAME = "feature.returns"

    def validate_params(self) -> None:
        self.params.setdefault("column", "close")

    def _compute_requires(self) -> set[str]:
        return {self.params["column"]}

    def _compute_provides(self) -> set[str]:
        return {"returns"}


class FeatureSMA(RollingOp):
    """Compute simple moving average. Output column: ``sma_{window}``."""

    NAME = "feature.sma"

    def validate_params(self) -> None:
        self.params.setdefault("column", "close")
        self.params.setdefault("window", 20)

    def _compute_requires(self) -> set[str]:
        return {self.params["column"]}

    def _compute_provides(self) -> set[str]:
        return {f"sma_{self.params['window']}"}


class FeatureRSI(RollingOp):
    """Compute Wilder RSI. Output column: ``rsi_{window}``."""

    NAME = "feature.rsi"

    def validate_params(self) -> None:
        self.params.setdefault("column", "close")
        self.params.setdefault("window", 14)

    def _compute_requires(self) -> set[str]:
        return {self.params["column"]}

    def _compute_provides(self) -> set[str]:
        return {f"rsi_{self.params['window']}"}


class DataLoadCSV(ElementwiseOp):
    """Load OHLCV data from CSV. Input to executor is None; produces a DataFrame from path."""

    NAME = "data.load_csv"

    def validate_params(self) -> None:
        if "path" not in self.params:
            raise ValueError("DataLoadCSV requires a 'path' parameter")
        self.params.setdefault("try_parse_dates", True)

    def _compute_requires(self) -> set[str]:
        return set()

    def _compute_provides(self) -> set[str]:
        return set()


__all__ = [
    "RSI",
    "SMA",
    "EMA",
    "MACDLineSignal",
    "Bollinger",
    "ATR",
    "OBV",
    "VWAP",
    "RollingStd",
    "Lags",
    "ZScore",
    "RobustScaler",
    "SentimentLexicon",
    "PairBeta",
    "PairSpread",
    "HalfLife",
    "RollingZ",
    "RollingVol",
    "XSecRank",
    "VolScale",
    "ResidualOLS",
    "ResidualKF",
    "IndustryScore",
    "FeatureReturns",
    "FeatureSMA",
    "FeatureRSI",
    "DataLoadCSV",
    "lower_pairs_beta_polars",
    "lower_pairs_spread_polars",
    "lower_half_life_polars",
    "lower_rolling_vol_polars",
    "lower_xsec_rank_polars",
    "lower_vol_scale_polars",
    "lower_residual_ols_polars",
    "lower_residual_kf_polars",
    "lower_industry_score_polars",
]

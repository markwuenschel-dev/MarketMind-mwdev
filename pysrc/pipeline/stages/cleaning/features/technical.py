from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict

from pysrc.core.errors import DataValidationError
from pysrc.core.validation import validate_dataframe
from pysrc.pipeline.stages.cleaning.core.base import CleaningStep
from pysrc.pipeline.stages.cleaning.core.contracts import (
    CleaningPipelineState,
    CleaningRuntimeContext,
)
from pysrc.pipeline.stages.cleaning.core.registry import register_cleaning_step


def _count_numeric_rows(df: pl.DataFrame) -> int:
    return df.height if df.select(pl.selectors.numeric()).columns else 0


@dataclass
class _IncrementalRSIState:
    window: int
    gains: deque[float]
    losses: deque[float]
    prev_price: float | None = None

    @classmethod
    def create(cls, window: int) -> _IncrementalRSIState:
        return cls(window=window, gains=deque(maxlen=window), losses=deque(maxlen=window))

    def update(self, price: float) -> float:
        if self.prev_price is None:
            self.prev_price = price
            return np.nan
        delta = price - self.prev_price
        self.gains.append(max(delta, 0.0))
        self.losses.append(max(-delta, 0.0))
        self.prev_price = price
        if len(self.gains) < self.window:
            return np.nan
        avg_gain = float(np.mean(self.gains))
        avg_loss = float(np.mean(self.losses))
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))


@dataclass
class _IncrementalMACDState:
    fast: int
    slow: int
    signal: int
    ema_fast: float | None = None
    ema_slow: float | None = None
    macd_signal: float | None = None

    def update(self, price: float) -> tuple[float, float]:
        if self.ema_fast is None or self.ema_slow is None or self.macd_signal is None:
            self.ema_fast = price
            self.ema_slow = price
            self.macd_signal = 0.0
            return 0.0, 0.0
        alpha_fast = 2.0 / (self.fast + 1.0)
        alpha_slow = 2.0 / (self.slow + 1.0)
        alpha_signal = 2.0 / (self.signal + 1.0)
        self.ema_fast = price * alpha_fast + self.ema_fast * (1.0 - alpha_fast)
        self.ema_slow = price * alpha_slow + self.ema_slow * (1.0 - alpha_slow)
        macd = self.ema_fast - self.ema_slow
        self.macd_signal = macd * alpha_signal + self.macd_signal * (1.0 - alpha_signal)
        return macd, self.macd_signal


class RSIParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    window: int = 14
    output_column: str = "rsi"
    close_column: str = "close"
    fillna_method: str = "ffill"


@register_cleaning_step(
    step_type="feature.technical.rsi",
    version="1",
    params_model=RSIParams,
    stateful=True,
)
class RSINormalizerStep(CleaningStep):
    def _apply(
        self,
        df: pl.DataFrame,
        state: CleaningPipelineState,
        context: CleaningRuntimeContext,
    ):
        validate_dataframe(df)
        close_column = self.params.close_column
        if close_column not in df.columns:
            raise DataValidationError(
                "Missing close column for RSI", details={"column": close_column}
            )
        output_column = self.params.output_column
        if context.streaming:
            rsi_state = state.step_state.get(self.spec.step_id)
            if rsi_state is None:
                rsi_state = _IncrementalRSIState.create(self.params.window)
                state.step_state[self.spec.step_id] = rsi_state
            values = [rsi_state.update(float(price)) for price in df[close_column].to_list()]
            cleaned = df.with_columns(pl.Series(output_column, values))
        else:
            delta = pl.col(close_column).diff(1)
            gain = (
                pl.when(delta > 0)
                .then(delta)
                .otherwise(0.0)
                .rolling_mean(self.params.window, min_samples=1)
            )
            loss = (
                pl.when(delta < 0)
                .then(-delta)
                .otherwise(0.0)
                .rolling_mean(self.params.window, min_samples=1)
            )
            rsi = (100 - (100 / (1 + (gain / loss)))).alias(output_column)
            cleaned = df.with_columns(rsi)
        if self.params.fillna_method == "ffill":
            cleaned = cleaned.with_columns(pl.col(output_column).forward_fill())
        elif self.params.fillna_method == "zero":
            cleaned = cleaned.with_columns(pl.col(output_column).fill_null(0.0))
        return self._result(
            cleaned,
            state,
            metrics={"indicator": output_column, "window": self.params.window},
            mutation=self._additive_mutation(df.height, cleaned.height, added_columns=1),
        )


class MACDParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    fast: int = 12
    slow: int = 26
    signal: int = 9
    close_column: str = "close"
    macd_column: str = "macd"
    signal_column: str = "macd_signal"


@register_cleaning_step(
    step_type="feature.technical.macd",
    version="1",
    params_model=MACDParams,
    stateful=True,
)
class MACDNormalizerStep(CleaningStep):
    def _apply(
        self,
        df: pl.DataFrame,
        state: CleaningPipelineState,
        context: CleaningRuntimeContext,
    ):
        close_column = self.params.close_column
        if close_column not in df.columns:
            raise DataValidationError(
                "Missing close column for MACD", details={"column": close_column}
            )
        if context.streaming:
            macd_state = state.step_state.get(self.spec.step_id)
            if macd_state is None:
                macd_state = _IncrementalMACDState(
                    fast=self.params.fast,
                    slow=self.params.slow,
                    signal=self.params.signal,
                )
                state.step_state[self.spec.step_id] = macd_state
            macd_values = [macd_state.update(float(price)) for price in df[close_column].to_list()]
            cleaned = df.with_columns(
                pl.Series(self.params.macd_column, [value[0] for value in macd_values]),
                pl.Series(self.params.signal_column, [value[1] for value in macd_values]),
            )
        else:
            ema_fast = pl.col(close_column).ewm_mean(span=self.params.fast, adjust=False)
            ema_slow = pl.col(close_column).ewm_mean(span=self.params.slow, adjust=False)
            macd_expr = (ema_fast - ema_slow).alias(self.params.macd_column)
            signal_expr = (
                (ema_fast - ema_slow)
                .ewm_mean(span=self.params.signal, adjust=False)
                .alias(self.params.signal_column)
            )
            cleaned = df.with_columns(macd_expr, signal_expr)
        return self._result(
            cleaned,
            state,
            metrics={
                "indicator": self.params.macd_column,
                "fast": self.params.fast,
                "slow": self.params.slow,
            },
            mutation=self._additive_mutation(df.height, cleaned.height, added_columns=2),
        )


class ATRParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    window: int = 14
    output_column: str = "atr"
    high_column: str = "high"
    low_column: str = "low"
    close_column: str = "close"


@register_cleaning_step(
    step_type="feature.technical.atr",
    version="1",
    params_model=ATRParams,
)
class ATRNormalizerStep(CleaningStep):
    def _apply(
        self,
        df: pl.DataFrame,
        state: CleaningPipelineState,
        context: CleaningRuntimeContext,
    ):
        del context
        for column in (self.params.high_column, self.params.low_column, self.params.close_column):
            if column not in df.columns:
                raise DataValidationError("Missing ATR input column", details={"column": column})
        tr1 = pl.col(self.params.high_column) - pl.col(self.params.low_column)
        tr2 = (pl.col(self.params.high_column) - pl.col(self.params.close_column).shift(1)).abs()
        tr3 = (pl.col(self.params.low_column) - pl.col(self.params.close_column).shift(1)).abs()
        cleaned = df.with_columns(
            pl.max_horizontal(tr1, tr2, tr3)
            .rolling_mean(self.params.window, min_samples=1)
            .alias(self.params.output_column)
        )
        return self._result(
            cleaned,
            state,
            metrics={"indicator": self.params.output_column},
            mutation=self._additive_mutation(df.height, cleaned.height, added_columns=1),
        )


class VWAPParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    output_column: str = "vwap"
    high_column: str = "high"
    low_column: str = "low"
    close_column: str = "close"
    volume_column: str = "volume"


@register_cleaning_step(
    step_type="feature.technical.vwap",
    version="1",
    params_model=VWAPParams,
)
class VWAPNormalizerStep(CleaningStep):
    def _apply(
        self,
        df: pl.DataFrame,
        state: CleaningPipelineState,
        context: CleaningRuntimeContext,
    ):
        del context
        for column in (
            self.params.high_column,
            self.params.low_column,
            self.params.close_column,
            self.params.volume_column,
        ):
            if column not in df.columns:
                raise DataValidationError("Missing VWAP input column", details={"column": column})
        typical_price = (
            pl.col(self.params.high_column)
            + pl.col(self.params.low_column)
            + pl.col(self.params.close_column)
        ) / 3.0
        cleaned = df.with_columns(
            (
                (typical_price * pl.col(self.params.volume_column)).cum_sum()
                / pl.col(self.params.volume_column).cum_sum()
            ).alias(self.params.output_column)
        )
        return self._result(
            cleaned,
            state,
            metrics={"indicator": self.params.output_column},
            mutation=self._additive_mutation(df.height, cleaned.height, added_columns=1),
        )

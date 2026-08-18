from __future__ import annotations

import polars as pl
from pydantic import BaseModel, ConfigDict

from pysrc.core.errors import DataValidationError
from pysrc.data.calendars.holidays import HOLIDAYS
from pysrc.pipeline.stages.cleaning.core.base import CleaningStep
from pysrc.pipeline.stages.cleaning.core.contracts import (
    CleaningPipelineState,
    CleaningRuntimeContext,
)
from pysrc.pipeline.stages.cleaning.core.registry import register_cleaning_step


class TimeZoneParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    target_tz: str = "UTC"
    timestamp_col: str = "timestamp"


@register_cleaning_step(
    step_type="feature.calendar.timezone",
    version="1",
    params_model=TimeZoneParams,
)
class TimeZoneNormalizerStep(CleaningStep):
    def _apply(
        self,
        df: pl.DataFrame,
        state: CleaningPipelineState,
        context: CleaningRuntimeContext,
    ):
        del context
        if self.params.timestamp_col not in df.columns:
            raise DataValidationError(
                f"Missing timestamp column {self.params.timestamp_col!r}",
                details={"timestamp_col": self.params.timestamp_col},
            )
        cleaned = df.with_columns(
            pl.col(self.params.timestamp_col)
            .cast(pl.Datetime)
            .dt.replace_time_zone(self.params.target_tz)
            .alias(self.params.timestamp_col)
        )
        return self._result(
            cleaned,
            state,
            metrics={"target_tz": self.params.target_tz},
            mutation=self._cell_mutation(
                df.height,
                cleaned.height,
                rows_with_mutations=cleaned.height,
                cells_mutated=cleaned.height,
            ),
        )


class GlobalCalendarParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    countries: tuple[str, ...] = ()
    day_of_week: bool = True
    is_holiday: bool = True
    timestamp_col: str = "timestamp"


@register_cleaning_step(
    step_type="feature.calendar.global_calendar",
    version="1",
    params_model=GlobalCalendarParams,
)
class GlobalCalendarNormalizerStep(CleaningStep):
    def _apply(
        self,
        df: pl.DataFrame,
        state: CleaningPipelineState,
        context: CleaningRuntimeContext,
    ):
        del context
        if self.params.timestamp_col not in df.columns:
            raise DataValidationError(
                f"Missing timestamp column {self.params.timestamp_col!r}",
                details={"timestamp_col": self.params.timestamp_col},
            )
        timestamp_expr = pl.col(self.params.timestamp_col).cast(pl.Datetime)
        cleaned = df
        warnings: list[str] = []
        if self.params.day_of_week:
            cleaned = cleaned.with_columns(timestamp_expr.dt.weekday().alias("day_of_week"))
        if self.params.is_holiday:
            for country in self.params.countries:
                holidays = HOLIDAYS.get(country, [])
                if not holidays:
                    warnings.append(f"missing_holiday_calendar:{country}")
                    continue
                cleaned = cleaned.with_columns(
                    timestamp_expr.dt.strftime("%Y-%m-%d")
                    .is_in(holidays)
                    .alias(f"is_holiday_{country}")
                )
        return self._result(
            cleaned,
            state,
            warnings=warnings,
            metrics={"calendar_features": len(cleaned.columns) - len(df.columns)},
            mutation=self._additive_mutation(
                df.height,
                cleaned.height,
                added_columns=max(len(cleaned.columns) - len(df.columns), 0),
            ),
        )

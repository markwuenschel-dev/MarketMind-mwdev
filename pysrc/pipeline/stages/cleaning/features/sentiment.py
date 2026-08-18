from __future__ import annotations

import polars as pl
from pydantic import BaseModel, ConfigDict

from pysrc.core.errors import DataValidationError
from pysrc.pipeline.stages.cleaning.core.base import CleaningStep
from pysrc.pipeline.stages.cleaning.core.contracts import (
    CleaningPipelineState,
    CleaningRuntimeContext,
)
from pysrc.pipeline.stages.cleaning.core.providers import GovernedColumns
from pysrc.pipeline.stages.cleaning.core.registry import register_cleaning_step


class SentimentParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_key: str
    output_column: str = "sentiment"
    text_column: str = "text"


class _ProviderSentimentStep(CleaningStep):
    def _apply(
        self,
        df: pl.DataFrame,
        state: CleaningPipelineState,
        context: CleaningRuntimeContext,
    ):
        if self.params.text_column not in df.columns:
            raise DataValidationError(
                "Sentiment extraction requires a text column on governed paths",
                details={"text_column": self.params.text_column},
            )
        provider = context.provider(self.params.provider_key)
        governed_columns: GovernedColumns = provider.materialize(
            df,
            context=context,
            params=self.params,
        )
        if governed_columns.frame.height != df.height:
            raise DataValidationError("Sentiment provider must return row-aligned columns")
        lineage = dict(governed_columns.lineage)
        columns_frame = governed_columns.frame
        if self.params.output_column not in columns_frame.columns:
            raise DataValidationError(
                "Sentiment provider did not emit the configured output column",
                details={
                    "output_column": self.params.output_column,
                    "columns": columns_frame.columns,
                },
            )
        if context.governance_mode.value == "governed" and "model_identity" not in lineage:
            raise DataValidationError(
                "Governed sentiment providers must emit model provenance",
                details={"provider": self.params.provider_key, "lineage": lineage},
            )
        cleaned = df.hstack(columns_frame.select(self.params.output_column))
        return self._result(
            cleaned,
            state,
            warnings=list(governed_columns.warnings),
            provider_lineage={self.params.provider_key: lineage},
            metrics={
                "provider": self.params.provider_key,
                "output_column": self.params.output_column,
            },
            mutation=self._additive_mutation(
                df.height,
                cleaned.height,
                added_columns=1,
            ),
        )


class VaderSentimentParams(SentimentParams):
    provider_key: str = "sentiment.vader"


@register_cleaning_step(
    step_type="feature.sentiment.vader",
    version="1",
    params_model=VaderSentimentParams,
    provider_requirements=("sentiment.vader",),
)
class VaderSentimentStep(_ProviderSentimentStep):
    pass


class FinbertSentimentParams(SentimentParams):
    provider_key: str = "sentiment.finbert"


@register_cleaning_step(
    step_type="feature.sentiment.finbert",
    version="1",
    params_model=FinbertSentimentParams,
    provider_requirements=("sentiment.finbert",),
)
class FinbertSentimentStep(_ProviderSentimentStep):
    pass

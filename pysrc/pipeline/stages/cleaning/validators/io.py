from __future__ import annotations

from abc import abstractmethod
from pathlib import Path

import polars as pl
from pydantic import BaseModel, ConfigDict

from pysrc.core.errors import FileFormatError
from pysrc.core.validation import validate_dataframe, validate_file_data
from pysrc.pipeline.stages.cleaning.core.base import CleaningStep
from pysrc.pipeline.stages.cleaning.core.contracts import (
    CleaningPipelineState,
    CleaningRuntimeContext,
)
from pysrc.pipeline.stages.cleaning.core.registry import register_cleaning_step


class FSInterface:
    @abstractmethod
    def get_size(self, path: str) -> int: ...


class LocalFS(FSInterface):
    def get_size(self, path: str) -> int:
        return Path(path).stat().st_size


class IOValidationParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    file_path: str
    format: str | None = None
    max_size_bytes: int = 100 * 1024 * 1024
    fs: FSInterface | None = None


@register_cleaning_step(
    step_type="validate.io",
    version="1",
    params_model=IOValidationParams,
)
class IOValidationStep(CleaningStep):
    def _apply(
        self,
        df: pl.DataFrame,
        state: CleaningPipelineState,
        context: CleaningRuntimeContext,
    ):
        del context
        file_path = self.params.file_path
        file_format = (self.params.format or Path(file_path).suffix[1:].lower()).lower()
        if file_format not in {"csv", "parquet", "json"}:
            raise FileFormatError(f"Unsupported file format: {file_format}")
        fs = self.params.fs or LocalFS()
        file_size = fs.get_size(file_path)
        if file_size > self.params.max_size_bytes:
            raise FileFormatError(
                f"File too large: {file_size} bytes > {self.params.max_size_bytes} bytes"
            )
        validate_file_data(df, file_format)
        validate_dataframe(df)
        return self._result(
            df,
            state,
            metrics={"file_format": file_format, "file_size_bytes": file_size},
            mutation=self._cell_mutation(df.height, df.height),
        )

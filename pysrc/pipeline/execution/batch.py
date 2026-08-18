# py/pipeline/execution/batch.py


from pysrc.core.errors import DataValidationError
from pysrc.core.runtime.optional_imports import optional_import
from pysrc.core.validation import validate_dataframe
from pysrc.ops.mm_logkit import get_logger
from pysrc.pipeline.core.pipeline_core_base import PipelineConfigError, PipelineStep
from pysrc.pipeline.core.pipeline_core_context import PipelineContext

logger = get_logger(__name__)
pl = optional_import("polars")
pd = optional_import("pandas")
dd = optional_import("dask.dataframe")


class BatchPipeline:
    def __init__(self, steps: list[PipelineStep], *, default_cfg: dict | None = None):
        self.steps = steps
        # local fallback used when ctx has no .pipeline_config attr
        self._default_cfg = default_cfg or {"streaming": True}

    def run(
        self,
        data,
        *,
        ctx: PipelineContext | None = None,
        distributed: str | None = None,
        collect: bool = True,
    ):
        try:
            ctx = ctx or PipelineContext()
            # Pandas + optional Dask path only if input is pandas
            if pd is not None and isinstance(data, pd.DataFrame):
                validate_dataframe(data)
                if distributed == "dask":
                    if dd is None:
                        logger.warning("Dask not available; falling back to single-node processing")
                    else:
                        logger.info("Using Dask for distributed batch processing")
                        ddf = dd.from_pandas(data, npartitions=8)

                        def _apply(df_: "pd.DataFrame") -> "pd.DataFrame":
                            out = df_
                            for s in self.steps:
                                if hasattr(s, "apply_batch_pandas"):
                                    out = s.apply_batch_pandas(out, ctx)
                                else:  # convert through Polars for this step
                                    pl_df = pl.from_pandas(out).lazy()
                                    streaming_flag = getattr(
                                        getattr(ctx, "pipeline_config", None),
                                        "get",
                                        lambda *_: self._default_cfg["streaming"],
                                    )("streaming", True)
                                    out = (
                                        s.apply_batch(pl_df, ctx)
                                        .collect(streaming=streaming_flag)
                                        .to_pandas()
                                    )
                            return out

                        return ddf.map_partitions(_apply).compute()
                # local pandas
                out = data
                for s in self.steps:
                    if hasattr(s, "apply_batch_pandas"):
                        out = s.apply_batch_pandas(out, ctx)
                    else:
                        pl_df = pl.from_pandas(out).lazy()
                        streaming_flag = getattr(
                            getattr(ctx, "pipeline_config", None),
                            "get",
                            lambda *_: self._default_cfg["streaming"],
                        )("streaming", True)
                        out = (
                            s.apply_batch(pl_df, ctx).collect(streaming=streaming_flag).to_pandas()
                        )
                return out

            # Polars-first
            lf = data.lazy() if isinstance(data, pl.DataFrame) else data
            if not isinstance(lf, pl.LazyFrame):
                raise PipelineConfigError(
                    "BatchPipeline expects Polars DataFrame/LazyFrame or pandas DataFrame"
                )
            if distributed == "dask":
                logger.warning(
                    "Dask requested but input is Polars; falling back to single-node processing"
                )
            for s in self.steps:
                lf = s.apply_batch(lf, ctx)
            streaming_flag = getattr(
                getattr(ctx, "pipeline_config", None),
                "get",
                lambda *_: self._default_cfg["streaming"],
            )("streaming", True)
            result = lf.collect(streaming=streaming_flag) if collect else lf
            try:
                validate_dataframe(result) if collect else None
            except DataValidationError as e:
                logger.warning(
                    f"Post-collect validation failed: {str(e)}; returning unvalidated result",
                    error_type="validation_warning",
                )
            return result
        except DataValidationError as e:
            logger.error(
                f"Batch pipeline validation failed: {str(e)}",
                error_type="validation_error",
                details=e.details,
            )
            raise
        except Exception as e:
            logger.error(
                f"Batch pipeline failed: {str(e)}",
                error_type="execution_error",
                severity="critical",
            )
            raise

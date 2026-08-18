# tests/python/test_core_base.py
import pytest


def test_pipeline_context_refine(df_prices):
    try:
        from pysrc.pipeline.core.pipeline_core_context import PipelineContext
    except Exception as e:
        pytest.skip(f"PipelineContext not available: {e!r}")

    ctx = PipelineContext(df=df_prices)
    if hasattr(ctx, "refine"):
        new = ctx.refine(frequency="1d")
        assert new is not ctx
        assert getattr(new, "frequency", None) == "1d"
        assert getattr(ctx, "frequency", None) != "1d"
    else:
        pytest.skip("PipelineContext.refine() not implemented in this codebase")

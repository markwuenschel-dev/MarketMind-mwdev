# utils/io_gpu.py
from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from typing import Any, Literal

from pysrc.core.runtime.optional_imports import optional_import
from pysrc.ops.mm_logkit import get_logger
from pysrc.preprocessor.utils.cuda_runtime import capabilities
from pysrc.preprocessor.utils.errors import OOMRetry
from pysrc.preprocessor.utils.nvtx import nvtx_range

logger = get_logger(__name__)
cudf = optional_import("cudf")
polars = optional_import("polars")

NVCOMP_CODECS = {"zstd", "snappy", "lz4"}
Frame = Any


@dataclass
class ParquetOptions:
    columns: list[str] | None = None
    compression: Literal["snappy", "zstd", "lz4", "none"] | None = "snappy"
    filters: Any | None = None  # backend-specific predicate
    engine: Literal["cudf", "polars"] | None = None


@nvtx_range("read_parquet_gpu")
def read_parquet_gpu(path: str | list[str], opts: ParquetOptions | None = None) -> Frame:
    opts = opts or ParquetOptions()
    caps = capabilities()
    engine = opts.engine or ("polars" if (polars and caps.has_polars_gpu) else "cudf")
    try:
        if engine == "cudf" and cudf:
            kwargs = {}
            if opts.columns:
                kwargs["columns"] = opts.columns
            if opts.filters:
                kwargs["filters"] = opts.filters
            return cudf.read_parquet(path, **kwargs)
        if engine == "polars" and polars:
            scan = polars.scan_parquet(path)
            if opts.columns:
                scan = scan.select(opts.columns)
            if opts.filters is not None:
                scan = scan.filter(opts.filters)
            try:
                eng = polars.GpuEngine(uvm=True)  # 2025 UVM
                return scan.collect(engine=eng)
            except AttributeError:
                return scan.collect(engine="streaming")
            except AttributeError:
                return scan.collect(engine="streaming")
        raise RuntimeError("No GPU reader available.")
    except MemoryError:
        raise OOMRetry(
            "Read OOM: consider filtering columns or reading in chunks",
            retry_hint={"strategy": "reduce_columns"},
        )


@nvtx_range("write_parquet_gpu")
def write_parquet_gpu(
    df: Frame,
    path: str,
    opts: ParquetOptions | None = None,
    **kwargs: Any,
) -> None:
    opts = opts or ParquetOptions()
    engine = opts.engine or ("polars" if (polars and isinstance(df, polars.DataFrame)) else "cudf")
    compression = opts.compression if (opts.compression in NVCOMP_CODECS) else "snappy"
    tmp_path = f"{path}.tmp"
    try:
        if engine == "cudf" and cudf:
            df.to_parquet(tmp_path, compression=compression, **kwargs)
        elif engine == "polars" and polars:
            df.write_parquet(tmp_path, compression=compression, **kwargs)
        else:
            raise RuntimeError("No GPU writer available.")
        os.replace(tmp_path, path)  # atomic commit
    except MemoryError:
        if os.path.exists(tmp_path):
            with contextlib.suppress(Exception):
                os.unlink(tmp_path)
        raise OOMRetry("Write OOM", {"backoff": 2})
    except Exception:
        if os.path.exists(tmp_path):
            with contextlib.suppress(Exception):
                os.unlink(tmp_path)
        raise

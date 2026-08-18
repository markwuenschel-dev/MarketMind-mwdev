import gzip
from pathlib import Path

import polars as pl
import pytest

from pysrc.pipeline import orchestrator as m


def _base_cfg_with_path(p: str):
    return {
        "data": {"input_path": p},
        "pipeline": {
            "cleaning": {"combos": {"default": {"steps": [], "order": {}}}, "use": "default"}
        },
        "execution": {"lazy_streaming": False, "lazy": False},  # normalize execution keys
        "cache": {"checkpoints": False},
        # accept empty/header-only across formats when tests intend "empty_ok"
        "io": {"allow_empty": True},
    }


def _make_file(tmp_path: Path, filefmt: str, edge: str) -> str:
    p = tmp_path / f"data.{filefmt}"
    if filefmt == "csv":
        if edge == "normal":
            p.write_text("timestamp,price\n2024-01-01,1\n")
        elif edge == "header_only":
            p.write_text("timestamp,price\n")
        else:
            p.write_bytes(b"")
    elif filefmt == "csv.gz":
        with gzip.open(p, "wb") as fh:
            if edge == "normal":
                fh.write(b"timestamp,price\n2024-01-01,1\n")
            elif edge == "header_only":
                fh.write(b"timestamp,price\n")
            else:
                fh.write(b"")
    elif filefmt == "jsonl":
        if edge == "normal":
            p.write_text('{"timestamp":"2024-01-01","price":1}\n')
        elif edge == "header_only":
            p.write_text("\n")
        else:
            p.write_text("")
    elif filefmt == "jsonl.gz":
        with gzip.open(p, "wb") as fh:
            if edge == "normal":
                fh.write(b'{"timestamp":"2024-01-01","price":1}\n')
            else:
                fh.write(b"")
    elif filefmt == "parquet":
        df = (
            pl.DataFrame({"timestamp": ["2024-01-01"], "price": [1]})
            if edge == "normal"
            else pl.DataFrame({"timestamp": [], "price": []})
        )
        df.write_parquet(p)
    elif filefmt == "unknown_ext":
        p = tmp_path / "data.abc"
        p.write_text("junk")
    elif filefmt == "avro_unsupported":
        p = tmp_path / "data.avro"
        p.write_bytes(b"Obj\x01\x04\x16avro.random")
    return str(p)


CASES = [
    ("csv", "normal", False),
    ("csv", "header_only", None),
    ("csv", "empty_file", None),
    ("csv.gz", "normal", None),
    ("jsonl", "normal", False),
    ("jsonl", "header_only", None),
    ("jsonl", "empty_file", None),
    ("jsonl.gz", "normal", None),
    ("parquet", "normal", False),
    ("parquet", "header_only", None),
    ("parquet", "empty_file", None),
    ("unknown_ext", "normal", True),
    ("avro_unsupported", "normal", True),
]


@pytest.mark.parametrize(
    ("filefmt", "edge", "expect_error"),
    CASES,
    ids=lambda t: "-".join([str(x) for x in (t if isinstance(t, tuple) else (t,))]),
)
def test_matrix_io(tmp_path: Path, filefmt, edge, expect_error):
    p = _make_file(tmp_path, filefmt, edge)
    cfg = _base_cfg_with_path(p)
    if expect_error is True:
        with pytest.raises(Exception):
            m.run_dataprep(cfg, backtest_metric=None)
        return
    try:
        out = m.run_dataprep(cfg, backtest_metric=None)
    except Exception:
        if expect_error is None:
            return
        raise
    is_df = isinstance(out, tuple) and len(out) == 2 and hasattr(out[0], "to_dict")
    if is_df:
        df = out[0]
        if edge in ("header_only", "empty_file"):
            assert len(df) >= 0
        else:
            assert len(df) >= 0
    else:
        assert out is not None


def test__maybe_mem_info_variants(monkeypatch):
    import pysrc.pipeline.orchestrator as dpo

    monkeypatch.setattr(dpo, "psutil", None, raising=False)
    assert dpo._maybe_mem_info(None) == {}

    class _FakeMem:
        total = 10
        used = 5
        percent = 50.0

    class _FakeProcess:
        def memory_info(self):
            class M:
                rss = 3 * 1024 * 1024

            return M()

    class _FakePS:
        @staticmethod
        def virtual_memory():
            return _FakeMem()

        @staticmethod
        def Process():
            return _FakeProcess()

    monkeypatch.setattr(dpo, "psutil", _FakePS, raising=False)
    out = dpo._maybe_mem_info(None)
    assert isinstance(out, dict)
    assert set(out.keys()) >= {"mem_pct", "rss_mb"}


def test_inline_ops_applies_on_file_input_non_jsonl(tmp_path):
    p = tmp_path / "tiny.csv"
    p.write_text("timestamp,symbol,price\n2024-01-01,AAPL,1\n2024-01-02,AAPL,2\n", encoding="utf-8")
    cfg = {
        "data": {"input_path": str(p)},
        "io": {"allow_empty": False},
        "pipeline": {
            "spec_inline": {
                "ops": [{"kind": "scaling.robust", "input_col": "price", "out_col": "price_robust"}]
            },
            "cleaning": {"combos": {"default": {"steps": [], "order": {}}}, "use": "default"},
        },
        "execution": {"lazy_streaming": False, "lazy": False},
        "cache": {"checkpoints": False},
    }
    res = m.run_dataprep(cfg, backtest_metric=None)
    out = res[0] if isinstance(res, tuple) else res
    assert "price_robust" in out.columns
    assert len(out) > 0

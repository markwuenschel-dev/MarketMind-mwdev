# tests/python/plugins/torture_plugin.py
from __future__ import annotations

import hashlib
import itertools
import os
import random
from dataclasses import dataclass
from pathlib import Path

import pytest

try:
    import yaml  # optional, only if you use a manifest
except Exception:  # pragma: no cover
    yaml = None


@dataclass(frozen=True)
class TortureCase:
    id: str
    relpath: tuple[str, ...]  # ("file.csv",) or ("file.csv","sidecar.csv")
    expect: str  # "ok_frame" | "empty_ok" | "should_raise" | "crossfile_consistency" | "stream_ok"
    read_kwargs: dict[str, object]  # e.g. {"sep": ";", "encoding": "latin1"}
    fmt: str | None = None  # e.g., "jsonl"
    checks: tuple[str, ...] = ()
    raises: tuple[type, ...] = (Exception,)
    match: str | None = None
    marks: tuple[pytest.MarkDecorator, ...] = ()


# ----------------------- pytest hooks (options) -----------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    g = parser.getgroup("torture")
    g.addoption(
        "--torture-data-dir",
        action="store",
        default=None,
        help="Path to data fixtures; defaults to auto-discovered tests/.../data",
    )
    g.addoption(
        "--torture-max",
        action="store",
        type=int,
        default=int(os.getenv("TORTURE_MAX", "9999")),
        help="Max number of discovered cases to run (post- shuffle)",
    )
    g.addoption(
        "--torture-seed",
        action="store",
        type=int,
        default=None,
        help="Random seed for case shuffling (default: deterministic from session)",
    )
    g.addoption(
        "--torture-shuffle",
        action="store_true",
        default=False,
        help="Shuffle discovered cases before capping by --torture-max",
    )
    g.addoption(
        "--torture-backends",
        action="store",
        default="pandas,polars",
        help="Comma-separated backends to test",
    )
    g.addoption(
        "--torture-optimize",
        action="store",
        default="0,1",
        help="Comma-separated optimize flags (0/1)",
    )
    g.addoption(
        "--torture-manifest",
        action="store",
        default=None,
        help="Optional YAML manifest with per-fixture expectations",
    )


# ----------------------- discovery infra -----------------------


def _auto_data_dir(start: Path) -> Path:
    """
    Resolve the torture fixtures directory.

    Prefer the nearest ``data/`` directory that contains ``torture_manifest.yml``
    (``tests/python/data`` in-repo). A bare ``/workspace/data`` (or repo ``data/``)
    must not win over the canonical test fixtures when both exist (e.g. in CI).
    """
    here = start.resolve().parent
    while True:
        cand = here / "data"
        if cand.is_dir() and (cand / "torture_manifest.yml").exists():
            return cand
        if here.parent == here:
            break
        here = here.parent
    for up in (4, 3, 2, 1):
        cand = start.parents[up] / "data"
        if cand.exists():
            return cand
    cand = start.parent / "data"
    cand.mkdir(parents=True, exist_ok=True)
    return cand


def _sha(path: Path | None) -> str:
    if path is None or path.is_dir():
        return "<DIR>"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_manifest(manifest_path: str | None) -> dict[str, dict]:
    if not manifest_path:
        default_manifest = Path(__file__).resolve().parents[1] / "data" / "torture_manifest.yml"
        manifest_path = str(default_manifest) if default_manifest.exists() else None
    if not manifest_path:
        return {}
    if not yaml:
        pytest.skip("pyyaml not installed; cannot read torture manifest")
    p = Path(manifest_path)
    if not p.exists():
        pytest.skip(f"manifest not found: {manifest_path}")
    data = yaml.safe_load(p.read_text()) or {}
    return data if isinstance(data, dict) else {}


def _infer_from_name(fp: Path) -> dict:
    """Heuristics when manifest is absent."""
    name = fp.name.lower()
    d: dict[str, object] = {}
    if name.endswith(".gz"):
        d["expect"] = "ok_frame"
    if "zero_byte" in name or "header_only" in name:
        d["expect"] = "empty_ok"
    if "semicolon" in name:
        d["read_kwargs"] = {"sep": ";"}
    if "latin1" in name:
        d["read_kwargs"] = {"encoding": "latin1"}
    if "bom" in name:
        d["checks"] = ["headers_trimmed"]
    if "malformed" in name:
        d["expect"] = "should_raise"
        d["match"] = "parse|quote|malformed"
    if "unsupported" in name:
        d["expect"] = "should_raise"
    if fp.suffix.lower() in {".yaml", ".yml", ".md"}:
        d["expect"] = "should_raise"
    if "jsonl" in name and name.endswith(".jsonl"):
        d["fmt"] = "jsonl"
        d["expect"] = "ok_frame"
    if "unsorted_dupe" in name:
        d["checks"] = ["sorted_ts", "dedup_ts_symbol"]
    if "irregular_freq" in name:
        d["checks"] = ["sorted_ts"]
    if "stale" in name:
        d["checks"] = ["staleness_flag_or_metric"]
    if "timezones" in name:
        d["checks"] = ["tz_normalized_utc", "sorted_ts"]
    if "corp_actions" in name:
        d["checks"] = ["corp_actions_flags_or_adjust"]
    if "schema_drift" in name:
        d["checks"] = ["unknown_ok_or_strict_errors"]
    if "types_mixed" in name:
        d["checks"] = ["numeric_parse_price"]
    if "overflow" in name:
        d["checks"] = ["finite_prices", "no_negative_low"]
    if "csv_with_blank" in name:
        d.setdefault("expect", "ok_frame")
    return d


def _discover_cases(data_dir: Path, manifest: dict[str, dict]) -> list[TortureCase]:
    cases: list[TortureCase] = []
    sidecar_only = {
        str(meta["sidecar"])
        for meta in manifest.values()
        if isinstance(meta, dict) and isinstance(meta.get("sidecar"), str)
    }

    # Stream directory: validate separately in async test
    stream = data_dir / "micro_batches_stream"
    if stream.exists() and stream.is_dir():
        cases.append(
            TortureCase(
                id="micro_batches_stream",
                relpath=("micro_batches_stream/",),
                expect="stream_ok",
                read_kwargs={},
            )
        )

    # Single-file fixtures
    for fp in sorted(data_dir.glob("*")):
        if fp.is_dir():
            continue
        if fp.name in sidecar_only:
            continue
        meta = manifest.get(fp.name) or _infer_from_name(fp)
        if isinstance(meta, dict) and meta.get("sidecar"):
            continue
        expect = str(meta.get("expect", "ok_frame"))
        read_kwargs = dict(meta.get("read_kwargs", {}))
        fmt = meta.get("format")
        checks = tuple(meta.get("checks", ()))
        raises = tuple(meta.get("raises", (Exception,)))
        match = meta.get("match")
        marks: list[pytest.MarkDecorator] = []

        # Some default xfails if not implemented yet (edit as you implement)
        if fp.suffix == ".jsonl" and fmt == "jsonl":
            marks.append(pytest.mark.xfail(reason="JSONL ingestion not wired yet"))
        if "staleness_flag_or_metric" in checks:
            marks.append(pytest.mark.xfail(reason="Staleness metric/flag TBD"))
        if "tz_normalized_utc" in checks:
            marks.append(pytest.mark.xfail(reason="UTC normalization not implemented"))
        if "corp_actions_flags_or_adjust" in checks:
            marks.append(pytest.mark.xfail(reason="Corporate actions policy TBD"))

        case = TortureCase(
            id=fp.stem.replace(".", "_"),
            relpath=(fp.name,),
            expect=expect,
            read_kwargs=read_kwargs,
            fmt=fmt,
            checks=checks,
            raises=raises,
            match=match,
            marks=tuple(marks),
        )
        cases.append(case)

    # Cross-file / sidecar pairs from manifest (optional)
    for key, meta in manifest.items():
        if isinstance(meta, dict) and meta.get("sidecar"):
            main = key
            side = meta["sidecar"]
            case = TortureCase(
                id=f"{Path(main).stem}+{Path(side).stem}",
                relpath=(main, side),
                expect=str(meta.get("expect", "crossfile_consistency")),
                read_kwargs=dict(meta.get("read_kwargs", {})),
                fmt=meta.get("format"),
                checks=tuple(meta.get("checks", ())),
                raises=tuple(meta.get("raises", (Exception,))),
                match=meta.get("match"),
            )
            cases.append(case)

    return cases


# ----------------------- grid + paramization -----------------------


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "robust: intensive robustness suite")
    config.addinivalue_line("markers", "combinatoric: combinatorial parameter grid")
    config.addinivalue_line("markers", "stream: streaming-only tests")
    config.addinivalue_line("markers", "slow: may be slow")
    config.addinivalue_line("markers", "net: allows network access")


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "torture_case" not in metafunc.fixturenames:
        return

    start = Path(metafunc.definition.fspath)
    opts = metafunc.config.option
    data_dir = Path(opts.torture_data_dir) if opts.torture_data_dir else _auto_data_dir(start)
    manifest = _read_manifest(opts.torture_manifest)

    cases = _discover_cases(data_dir, manifest)

    # Shuffle/cap deterministically
    seed = opts.torture_seed
    r = random.Random(seed if seed is not None else 1337)
    if opts.torture_shuffle:
        r.shuffle(cases)
    maxn = int(opts.torture_max or 9999)
    cases = cases[:maxn]

    # Dynamic grid from CLI
    backends = [b.strip() for b in opts.torture_backends.split(",") if b.strip()]
    optimizes = [bool(int(x)) for x in opts.torture_optimize.split(",") if x.strip()]

    params = []
    for case, backend, optimize in itertools.product(cases, backends, optimizes):
        case_id = f"{case.id}-backend={backend}|optimize={optimize}"
        params.append(
            pytest.param(
                case,
                backend,
                optimize,
                id=case_id,
                marks=list(case.marks),
            )
        )
    metafunc.parametrize("torture_case,backend,optimize", params)

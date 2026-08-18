# matrix.py
from __future__ import annotations

import contextlib
import functools
import inspect
import os
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product
from typing import Any


class MatrixError(Exception): ...


class MatrixValidationError(MatrixError): ...


class MatrixProbeError(MatrixError): ...


pytest: Any
try:
    import pytest as _pytest
except ImportError:  # pragma: no cover
    pytest = None
else:
    pytest = _pytest

_FAIL_COUNTS: dict[str, int] = {}
_SKIP_CACHE: dict[str, bool] = {}


def _as_tuple_dict(d: Mapping[str, Any], keys: Sequence[str]) -> tuple[Any, ...]:
    return tuple(d[k] for k in keys)


def _short_id(v: Any) -> str:
    if isinstance(v, bool):
        return "t" if v else "f"
    if v is None:
        return "none"
    s = str(v).strip()
    s = re.sub(r"\s+", "", s)
    s = s.replace("/", "_").replace("\\", "_")
    return s[:32] if len(s) > 32 else s


def _combo_id_from_names(names: Sequence[str], case: Mapping[str, Any]) -> str:
    return ",".join(f"{k}={_short_id(case.get(k))}" for k in names)


def _parse_env_filter(s: str) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    if not s:
        return out
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k.strip()] = v.strip()
        else:
            out[tok] = None
    return out


def _matches_kv_filter(case: Mapping[str, Any], envf: Mapping[str, str | None]) -> bool:
    if not envf:
        return True
    for k, v in envf.items():
        if k not in case:
            return False
        if v is not None and str(case.get(k)) != v:
            return False
    return True


def _cartesian_from_grid(
    grid: Mapping[str, Sequence[Any]], names: Sequence[str]
) -> list[dict[str, Any]]:
    vals = [list(grid[k]) for k in names]
    return [dict(zip(names, tup, strict=False)) for tup in product(*vals)]


def _apply_constraints(
    case: Mapping[str, Any], constraints: Iterable[Callable[[Mapping[str, Any]], bool]] | None
) -> bool:
    if not constraints:
        return True
    for c in constraints:
        try:
            if not bool(c(case)):
                return False
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            raise MatrixValidationError(f"constraint callable raised for case {case}") from e
    return True


def _probe_case(
    case: Mapping[str, Any],
    probe: Callable[[Mapping[str, Any]], Mapping[str, Callable[[], Any]]] | None,
    timeout_s: float,
    workers: int,
) -> Mapping[str, Any]:
    if probe is None:
        return {}
    try:
        fn_map = probe(case) or {}
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        raise MatrixProbeError("probe() callable failed") from e
    if not fn_map:
        return {}
    out: dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        futs = {ex.submit(cb): name for name, cb in fn_map.items()}
        for fut in as_completed(futs, timeout=max(0.0, float(timeout_s)) * max(1, len(futs))):
            name = futs[fut]
            try:
                out[name] = fut.result(timeout=max(0.0, float(timeout_s)))
            except (KeyboardInterrupt, SystemExit):
                raise
            except TimeoutError:
                out[name] = MatrixProbeError(f"probe '{name}' timed out")
            except Exception:
                out[name] = MatrixProbeError(f"probe '{name}' failed")
    return out


def param_matrix(
    *,
    grid: Mapping[str, Sequence[Any]] | None = None,
    ids: Sequence[str] | Mapping[str, Mapping[Any, str]] | None = None,
    constraints: Sequence[Callable[[Mapping[str, Any]], bool]] | None = None,
    learn: bool = True,
    min_fail_skip: int = 3,
    probe: Callable[[Mapping[str, Any]], Mapping[str, Callable[[], Any]]] | None = None,
    probe_timeout_s: float = float(os.getenv("PYTEST_PROBE_TIMEOUT_S", "1.0")),
    probe_workers: int = int(os.getenv("PYTEST_PROBE_THREADS", "8")),
    environ_filter: str = os.getenv("PYTEST_MATRIX", ""),
    opts: Mapping[str, Any] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    o_cases = (opts or {}).get("cases")
    o_idfn = (opts or {}).get("idfn")
    o_skip_if = (opts or {}).get("skip_if")
    o_xfail_if = (opts or {}).get("xfail_if")
    o_marks = (opts or {}).get("marks")
    o_indirect = (opts or {}).get("indirect")
    o_order = (opts or {}).get("order")
    o_sort_axes = bool((opts or {}).get("sort_axes", False))
    o_strict = bool((opts or {}).get("strict_selection", False))
    o_probe_log = bool((opts or {}).get("probe_log", False))

    kv_filter = _parse_env_filter(environ_filter)
    regex_filter: re.Pattern[str] | None = None
    if environ_filter and not any("=" in t for t in environ_filter.split(",")):
        try:
            regex_filter = re.compile(environ_filter)
        except re.error as e:
            if pytest is not None:
                raise pytest.UsageError(f"Invalid PYTEST_MATRIX regex: {e}") from e
            raise MatrixValidationError("Invalid PYTEST_MATRIX regex") from e

    def _decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        sig_params = list(inspect.signature(func).parameters.keys())
        g_in = grid() if callable(grid) else (grid or {})
        g_in = {k: (v() if callable(v) else v) for k, v in g_in.items()}

        if o_cases:
            candidate_names = list(o_cases[0].keys()) if len(o_cases) > 0 else []
        else:
            candidate_names = list(g_in.keys())

        if o_order:
            missing = [n for n in candidate_names if n not in o_order]
            if missing:
                msg = f"'order' missing axes: {missing}"
                if pytest is not None:
                    raise pytest.UsageError(msg)
                raise MatrixValidationError(msg)
            candidate_names = [n for n in o_order if n in candidate_names]
        elif o_sort_axes:
            candidate_names = sorted(candidate_names)

        names: list[str] = [k for k in candidate_names if k in sig_params]
        if not names:
            return func

        if o_cases:
            exp = set(names)
            bad = [i for i, c in enumerate(o_cases) if not exp.issubset(set(c.keys()))]
            if bad:
                msg = f"All cases must provide keys {sorted(exp)}; missing in indices {bad[:10]}"
                if pytest is not None:
                    raise pytest.UsageError(msg)
                raise MatrixValidationError(msg)

        raw: list[dict[str, Any]]
        if o_cases:
            raw = [dict(c) for c in o_cases]
        else:
            raw = _cartesian_from_grid({k: g_in[k] for k in names}, names)

        combos: list[dict[str, Any]] = []
        for c in raw:
            if not _apply_constraints(c, constraints):
                continue
            if kv_filter and not _matches_kv_filter(c, kv_filter):
                continue
            if regex_filter is not None and not regex_filter.search(_combo_id_from_names(names, c)):
                continue
            combos.append(c)

        if probe and combos:
            if o_probe_log:
                with contextlib.suppress(Exception):
                    print(
                        f"[matrix] probe before: {len(combos)} combos; sample={_combo_id_from_names(names, combos[0])}"
                    )
            pruned: list[dict[str, Any]] = []
            for c in combos:
                try:
                    probe_map = _probe_case(c, probe, probe_timeout_s, probe_workers)
                except MatrixProbeError:
                    continue
                if any(isinstance(v, MatrixProbeError) for v in probe_map.values()):
                    continue
                pruned.append(c)
            combos = pruned
            if o_probe_log:
                try:
                    sample = _combo_id_from_names(names, combos[0]) if combos else "none"
                    print(f"[matrix] probe after: {len(combos)} combos; sample={sample}")
                except Exception:
                    pass

        if not combos:
            if pytest is None:
                return func
            if o_strict:
                raise pytest.UsageError("No matrix combinations matched selection/pruning")
            pytest.skip("No matrix combinations selected by environment filter/constraints/probe")
            return func  # pragma: no cover

        built: list[Any] = []
        id_counts: dict[str, int] = {}
        any_param_ids = False

        for idx, c in enumerate(combos):
            mk_list: list[Any] = []
            key = _combo_id_from_names(names, c)

            if learn and _SKIP_CACHE.get(key, False):
                if pytest is not None:
                    mk_list.append(pytest.mark.skip(reason="learned-skip"))
                else:
                    continue

            if callable(o_skip_if):
                try:
                    reason = o_skip_if(c)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception as e:
                    msg = f"skip_if callable failed for case {c}"
                    if pytest is not None:
                        raise pytest.UsageError(msg) from e
                    raise MatrixValidationError(msg) from e
                if reason:
                    if pytest is not None:
                        mk_list.append(pytest.mark.skip(reason=str(reason)))
                    else:
                        continue

            if callable(o_xfail_if):
                try:
                    reason = o_xfail_if(c)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception as e:
                    msg = f"xfail_if callable failed for case {c}"
                    if pytest is not None:
                        raise pytest.UsageError(msg) from e
                    raise MatrixValidationError(msg) from e
                if reason and pytest is not None:
                    mk_list.append(pytest.mark.xfail(reason=str(reason)))

            if callable(o_marks):
                try:
                    extra = o_marks(c)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception as e:
                    msg = f"marks callable failed for case {c}"
                    if pytest is not None:
                        raise pytest.UsageError(msg) from e
                    raise MatrixValidationError(msg) from e
                if extra:
                    if isinstance(extra, (list, tuple)):
                        mk_list.extend(list(extra))
                    else:
                        mk_list.append(extra)

            pid: str | None = None
            if callable(o_idfn):
                try:
                    pid = str(o_idfn(c))
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception as e:
                    msg = f"idfn callable failed for case {c}"
                    if pytest is not None:
                        raise pytest.UsageError(msg) from e
                    raise MatrixValidationError(msg) from e
            if pid is None and ids and isinstance(ids, Sequence) and len(ids) == len(combos):
                pid = str(ids[idx])
            if pid is None:
                if isinstance(ids, Mapping):
                    parts: list[str] = []
                    for k in names:
                        v = c[k]
                        m = ids.get(k) if isinstance(ids, Mapping) else None
                        parts.append(
                            str((m or {}).get(v, v))
                            if isinstance(m, Mapping)
                            else f"{k}={_short_id(v)}"
                        )
                    pid = "-".join(parts)
                else:
                    pid = _combo_id_from_names(names, c)

            cnt = id_counts.get(pid, 0)
            id_counts[p] = cnt + 1 if (p := pid) else 0  # keep stable counter
            if cnt > 0:
                pid = f"{pid}~{cnt + 1}"

            vals = _as_tuple_dict(c, names)
            if pytest is not None:
                built.append(pytest.param(*vals, id=pid, marks=mk_list))
                any_param_ids = True
            else:
                built.append(vals)

        target = func
        if learn:

            @functools.wraps(func)
            def wrapped(*a: Any, **kw: Any) -> Any:
                case_key = _combo_id_from_names(names, {n: kw.get(n) for n in names})
                try:
                    return func(*a, **kw)
                except (KeyboardInterrupt, SystemExit):
                    raise
                except AssertionError:
                    _FAIL_COUNTS[case_key] = _FAIL_COUNTS.get(case_key, 0) + 1
                    if _FAIL_COUNTS[case_key] >= int(min_fail_skip):
                        _SKIP_CACHE[case_key] = True
                    raise

            target = wrapped

        final_ids = None if any_param_ids else ([str(x) for x in (ids or [])] or None)
        if pytest is None:  # pragma: no cover
            return target

        # Only include 'indirect' when caller actually provided it (avoid None)
        parametrize_kwargs: dict[str, Any] = {"ids": final_ids}
        if o_indirect is not None:
            parametrize_kwargs["indirect"] = o_indirect

        return pytest.mark.parametrize(",".join(names), built, **parametrize_kwargs)(target)

    return _decorator


def matrix(*args: Any, **kwargs: Any):
    # Support bare @matrix and configured @matrix(...)
    if args and callable(args[0]) and not kwargs:
        return param_matrix()(args[0])

    legacy_keys = {
        "grid",
        "ids",
        "constraints",
        "learn",
        "min_fail_skip",
        "probe",
        "probe_timeout_s",
        "probe_workers",
        "environ_filter",
    }
    opt_keys = {
        "cases",
        "idfn",
        "skip_if",
        "xfail_if",
        "marks",
        "indirect",
        "order",
        "sort_axes",
        "strict_selection",
        "probe_log",
    }

    # Extract opts
    opts: dict[str, Any] = dict(kwargs.pop("opts", {}) or {})
    for k in list(kwargs.keys()):
        if k in opt_keys:
            opts[k] = kwargs.pop(k)

    # Pull out legacy (non-grid) so they don't get mistaken for axes
    legacy: dict[str, Any] = {}
    explicit_grid = kwargs.pop("grid", None)
    for k in list(kwargs.keys()):
        if k in legacy_keys and k != "grid":
            legacy[k] = kwargs.pop(k)

    # Remaining kwargs at this point are axis definitions; merge with explicit grid
    merged_grid: dict[str, Sequence[Any]] = {}
    for k, v in kwargs.items():
        merged_grid[k] = v
    if isinstance(explicit_grid, Mapping):
        for k, v in explicit_grid.items():
            merged_grid[k] = v

    return param_matrix(
        grid=merged_grid or None,
        ids=legacy.get("ids"),
        constraints=legacy.get("constraints"),
        learn=legacy.get("learn", True),
        min_fail_skip=legacy.get("min_fail_skip", 3),
        probe=legacy.get("probe"),
        probe_timeout_s=legacy.get(
            "probe_timeout_s", float(os.getenv("PYTEST_PROBE_TIMEOUT_S", "1.0"))
        ),
        probe_workers=legacy.get("probe_workers", int(os.getenv("PYTEST_PROBE_THREADS", "8"))),
        environ_filter=legacy.get("environ_filter", os.getenv("PYTEST_MATRIX", "")),
        opts=opts or None,
    )

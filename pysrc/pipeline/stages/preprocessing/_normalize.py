from __future__ import annotations

from typing import Any


def canonical_op(op_id: str, aliases: dict[str, str]) -> str:
    return aliases.get(op_id, op_id)


def normalize_common_keys(o: dict[str, Any]) -> dict[str, Any]:
    o = dict(o)
    if "by" not in o and "group_key" in o:
        o["by"] = o.pop("group_key")
    if "columns" not in o and "cols" in o:
        o["columns"] = o.pop("cols")
    if "text_col" not in o and "text" in o:
        o["text_col"] = o.pop("text")
    if "ts_col" not in o:
        if "timestamp_col" in o:
            o["ts_col"] = o.pop("timestamp_col")
        elif "timestamp" in o:
            o["ts_col"] = o.pop("timestamp")
    o.pop("enabled", None)
    return o


def normalize_input(o: dict[str, Any]) -> dict[str, Any]:
    if "input" not in o:
        for k in ("col", "column", "source"):
            if k in o:
                o["input"] = o.pop(k)
                break
    return o


def normalize_clip(o: dict[str, Any]) -> dict[str, Any]:
    if "lower" not in o and "clip_min" in o:
        o["lower"] = o.pop("clip_min")
    if "upper" not in o and "clip_max" in o:
        o["upper"] = o.pop("clip_max")
    return o


def ensure_list(o: dict[str, Any], key: str) -> dict[str, Any]:
    if key in o and isinstance(o[key], int):
        o[key] = [o[key]]
    return o


def map_fill_strategy(val: str) -> str:
    v = str(val).lower()
    if v in ("forward", "ffill"):
        return "ffill"
    if v in ("backward", "bfill"):
        return "bfill"
    return "none"


def normalize_lag(o: dict[str, Any]) -> dict[str, Any]:
    o = ensure_list(o, "lags")
    if "fill_strategy" in o and "fill" not in o:
        o["fill"] = map_fill_strategy(o.pop("fill_strategy"))
    if "column" not in o and "base_col" in o:
        o["column"] = o.pop("base_col")
    if "prefix" not in o and "lag_prefix" in o:
        o["prefix"] = o.pop("lag_prefix")
    return o


def normalize_bucket(o: dict[str, Any]) -> dict[str, Any]:
    if "unit" not in o and "granularity" in o:
        o["unit"] = o.pop("granularity")
    if "size" not in o and "every" in o:
        o["size"] = o.pop("every")
    if "label" not in o and "name" in o:
        o["label"] = o.pop("name")
    return o


def assign_device_hint(o: dict[str, Any]) -> dict[str, Any]:
    if "use_gpu" in o and "device" not in o:
        o["device"] = "gpu" if o.pop("use_gpu") else "cpu"
    return o

"""Shared JSON conversion and recursive string-value helpers.

This module has no filesystem or Agent side effects.
It can be reused by deterministic tools, run tracing, and artifact policies.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np

UnknownValuePolicy = Literal["preserve", "string"]


def to_jsonable(
    value: Any,
    *,
    unknown: UnknownValuePolicy = "preserve",
) -> Any:
    """Convert nested values into JSON-compatible Python primitives."""

    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {
            str(key): to_jsonable(item, unknown=unknown)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item, unknown=unknown) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist(), unknown=unknown)
    if isinstance(value, np.floating):
        result = float(value)
        return result if math.isfinite(result) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return str(value) if unknown == "string" else value


def flatten_string_leaves(value: Any) -> list[str]:
    """Return every string leaf from nested mappings and sequences."""

    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        result: list[str] = []
        for item in value.values():
            result.extend(flatten_string_leaves(item))
        return result
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(flatten_string_leaves(item))
        return result
    return []


__all__ = ["UnknownValuePolicy", "flatten_string_leaves", "to_jsonable"]

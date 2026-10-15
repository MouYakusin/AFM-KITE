"""Shared persistence helpers for deterministic AFM workflows.

These helpers remain domain-light so operation modules can reuse them.
They provide validated JSON loading, atomic writes, safe filenames, and provenance attachment.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..contracts import validate_artifact_contract
from ..data_io.paths import resolve_readable_file
from .provenance import attach_current_provenance
from ..serialization import to_jsonable


def safe_stem(
    value: Any,
    *,
    fallback: str = "artifact",
    preserve_separators: bool = True,
    normalize_micro: bool = False,
) -> str:
    """Return a filesystem-safe artifact stem with an explicit fallback."""

    text = str(value).strip()
    if normalize_micro:
        text = text.replace("μ", "u")
    pattern = r"[^A-Za-z0-9_.-]+" if preserve_separators else r"[^A-Za-z0-9]+"
    cleaned = re.sub(pattern, "_", text).strip("_")
    return cleaned or fallback


def read_json_file(
    path: str | Path,
    *,
    expected_schema: str | Collection[str] | None = None,
    required_keys: Sequence[str] = (),
) -> tuple[Path, dict[str, Any]]:
    """Resolve, read, and optionally validate one JSON-object artifact."""

    resolved = resolve_readable_file(path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {resolved}")
    validate_artifact_contract(
        payload,
        expected_schema=expected_schema,
        required_keys=required_keys,
    )
    return resolved, payload


def write_json_file(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write one JSON artifact and attach active invocation provenance."""

    return write_text_file(
        path,
        json.dumps(
            to_jsonable(attach_current_provenance(payload)),
            ensure_ascii=False,
            indent=2,
        ),
    )


def atomic_write_path(path: str | Path, writer: Any) -> Path:
    """Write through a same-directory temporary path and atomically replace target."""

    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(
        f".{resolved.stem}.{uuid4().hex}.tmp{resolved.suffix}"
    )
    try:
        writer(temporary)
        os.replace(temporary, resolved)
    finally:
        temporary.unlink(missing_ok=True)
    return resolved


def write_text_file(
    path: str | Path,
    text: str,
    *,
    encoding: str = "utf-8",
) -> Path:
    """Atomically write a text artifact."""

    return atomic_write_path(
        path,
        lambda temporary: temporary.write_text(text, encoding=encoding),
    )


__all__ = [
    "atomic_write_path",
    "read_json_file",
    "safe_stem",
    "write_json_file",
    "write_text_file",
]

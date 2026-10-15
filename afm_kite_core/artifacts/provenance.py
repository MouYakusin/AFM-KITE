"""Invocation identity and artifact provenance helpers.

The public tool boundary creates one context for every operation.
Stage implementations remain unaware of that context.
Shared JSON writers attach it to persisted artifacts automatically.
"""

from __future__ import annotations

import uuid
from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Optional

from .._version import __version__
from ..serialization import to_jsonable
from .identity import file_content_identity

PROVENANCE_SCHEMA = "afm.provenance/v3"
INVOCATION_MANIFEST_SCHEMA = "afm.invocation-manifest/v3"
INVOCATION_ID_LENGTH = 16
_PATH_PARAMETER_SUFFIXES = ("_path", "_paths")
_CURRENT_PROVENANCE: ContextVar[Optional[dict[str, Any]]] = ContextVar(
    "afm_analysis_provenance",
    default=None,
)


def _stable_value(value: Any) -> Any:
    """Return a JSON-safe value with deterministic mapping key order."""

    converted = to_jsonable(value, unknown="string")
    if isinstance(converted, Mapping):
        return {
            str(key): _stable_value(converted[key])
            for key in sorted(converted, key=str)
        }
    if isinstance(converted, list):
        return [_stable_value(item) for item in converted]
    return converted


def describe_file(path: str | Path) -> dict[str, Any]:
    """Return a filesystem metadata record for one file-like input."""

    requested = str(path)
    resolved = Path(path).expanduser().resolve()
    record: dict[str, Any] = {
        "path": str(resolved),
        "requested_path": requested,
    }
    try:
        resolved.stat()
    except OSError as exc:
        record.update(
            {
                "exists": False,
                "identity": f"unavailable:{requested}",
                "error_type": type(exc).__name__,
            }
        )
        return record

    if not resolved.is_file():
        record.update(
            {
                "exists": True,
                "kind": "directory" if resolved.is_dir() else "other",
                "identity": f"non-file:{resolved}",
            }
        )
        return record

    size_bytes, content_digest = file_content_identity(resolved)
    record.update(
        {
            "exists": True,
            "kind": "file",
            "size_bytes": size_bytes,
            "content_digest": content_digest,
        }
    )
    return record


def _iter_parameter_paths(value: Any) -> Iterator[tuple[Optional[int], str]]:
    if isinstance(value, (str, Path)):
        if str(value).strip():
            yield None, str(value)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            if isinstance(item, (str, Path)) and str(item).strip():
                yield index, str(item)


def build_invocation_provenance(
    tool: str,
    arguments: Mapping[str, Any],
) -> dict[str, Any]:
    """Build provenance metadata and a fresh invocation identifier."""

    parameters: OrderedDict[str, Any] = OrderedDict()
    inputs: list[dict[str, Any]] = []
    for name in sorted(arguments):
        value = arguments[name]
        if name == "output_dir":
            continue
        if name.endswith(_PATH_PARAMETER_SUFFIXES):
            for index, path in _iter_parameter_paths(value):
                record = describe_file(path)
                record["parameter"] = name
                if index is not None:
                    record["index"] = index
                inputs.append(record)
            continue
        parameters[name] = _stable_value(value)

    invocation_id = uuid.uuid4().hex[:INVOCATION_ID_LENGTH]
    return {
        "schema": PROVENANCE_SCHEMA,
        "invocation_id": invocation_id,
        "tool": tool,
        "implementation": {
            "package": "afm-analysis-core",
            "version": __version__,
        },
        "parameters": parameters,
        "inputs": inputs,
    }


def invocation_output_dir(
    output_dir: str | Path,
    invocation_id: str,
) -> Path:
    """Route one operation below its invocation directory."""

    base = Path(output_dir).expanduser()
    return base if base.name == invocation_id else base / invocation_id


@contextmanager
def invocation_provenance(provenance: Mapping[str, Any]) -> Iterator[None]:
    """Expose one invocation's provenance to shared artifact writers."""

    value = _stable_value(provenance)
    token = _CURRENT_PROVENANCE.set(value)
    try:
        yield
    finally:
        _CURRENT_PROVENANCE.reset(token)


def current_provenance() -> Optional[dict[str, Any]]:
    """Return a detached copy of the active invocation provenance, if any."""

    value = _CURRENT_PROVENANCE.get()
    return dict(value) if value is not None else None


def attach_current_provenance(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a payload and attach the active provenance block when available."""

    result = dict(payload)
    provenance = current_provenance()
    if provenance is not None:
        result["provenance"] = provenance
    return result


__all__ = [
    "INVOCATION_ID_LENGTH",
    "INVOCATION_MANIFEST_SCHEMA",
    "PROVENANCE_SCHEMA",
    "attach_current_provenance",
    "build_invocation_provenance",
    "current_provenance",
    "describe_file",
    "invocation_output_dir",
    "invocation_provenance",
]

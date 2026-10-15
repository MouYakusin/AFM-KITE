"""Shared result handling for public AFM analysis tools.

The public tool layer always returns Python dictionaries.
Deterministic stage implementations return mappings and let exceptions propagate here.
JSON input is handled only at serialization boundaries.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Callable

from ..artifacts.provenance import (
    INVOCATION_MANIFEST_SCHEMA,
    build_invocation_provenance,
    describe_file,
    invocation_output_dir,
    invocation_provenance,
)
from ..serialization import to_jsonable

JsonDict = dict[str, Any]
_PATH_KEYS = ("path", "paths", "dir", "directory", "artifact", "artifacts")


def _parse_payload(value: Any) -> JsonDict:
    if not isinstance(value, dict):
        raise TypeError("Analysis result must be a Python dictionary")
    return to_jsonable(value, unknown="string")


def _looks_like_artifact(key: str, value: Any) -> bool:
    return any(key.lower().endswith(suffix) for suffix in _PATH_KEYS) and isinstance(
        value,
        (str, list, tuple, dict),
    )


def _extract_artifacts(payload: JsonDict) -> JsonDict:
    existing = payload.get("artifacts")
    if isinstance(existing, dict):
        return existing
    return {
        key: value
        for key, value in payload.items()
        if value not in (None, "", [], {}) and _looks_like_artifact(key, value)
    }


def _extract_summary(payload: JsonDict) -> JsonDict:
    existing = payload.get("summary")
    if isinstance(existing, dict):
        return existing
    return {
        key: value
        for key, value in payload.items()
        if key not in {"status", "step", "message", "artifacts"}
        and not _looks_like_artifact(key, value)
        and (value is None or isinstance(value, (str, int, float, bool)))
    }


def _extract_result(payload: JsonDict) -> JsonDict:
    """Keep domain values without repeating public envelope fields or artifacts."""

    envelope_fields = {
        "artifacts",
        "error",
        "message",
        "provenance",
        "status",
        "step",
        "summary",
    }
    return {
        key: value
        for key, value in payload.items()
        if key not in envelope_fields and not _looks_like_artifact(key, value)
    }


def error_result(
    step: str,
    error: Exception | str,
    *,
    provenance: JsonDict | None = None,
) -> JsonDict:
    """Return one public error result without raising into the Agent runtime."""
    result: JsonDict = {
        "status": "error",
        "step": step,
        "message": str(error),
        "artifacts": {},
        "summary": {},
    }
    if provenance is not None:
        result["invocation_id"] = provenance["invocation_id"]
        result["provenance"] = provenance
    return result


def normalise_result(
    step: str,
    value: Any,
    *,
    provenance: JsonDict | None = None,
) -> JsonDict:
    """Convert an operation result to the stable public tool contract."""
    payload = _parse_payload(value)
    status = payload.get("status")
    if status not in {"ok", "error"}:
        return error_result(
            step,
            "Invalid analysis status {!r}; expected 'ok' or 'error'.".format(status),
            provenance=provenance,
        )

    result: JsonDict = {
        "status": status,
        "step": step,
        "artifacts": _extract_artifacts(payload),
        "summary": _extract_summary(payload),
        "result": _extract_result(payload),
    }
    if provenance is not None:
        result["invocation_id"] = provenance["invocation_id"]
        result["provenance"] = provenance
    message = payload.get("message") or payload.get("error")
    if message:
        result["message"] = str(message)
    elif status == "error":
        result["message"] = "Analysis reported status 'error' without a failure message."
    return result


def _write_invocation_manifest(
    output_dir: Path,
    provenance: JsonDict,
    status: str,
) -> Path:
    """Persist a file inventory linking every generated file to its inputs."""

    from ..artifacts.writers import write_json_file

    manifest_path = output_dir / "invocation.json"
    outputs = []
    if output_dir.is_dir():
        for path in sorted(output_dir.rglob("*")):
            if not path.is_file() or path == manifest_path or path.name.startswith("."):
                continue
            record = describe_file(path)
            record["relative_path"] = path.relative_to(output_dir).as_posix()
            record.pop("requested_path", None)
            outputs.append(record)
    write_json_file(
        manifest_path,
        {
            "status": status,
            "schema": INVOCATION_MANIFEST_SCHEMA,
            "output_dir": str(output_dir.resolve()),
            "provenance": provenance,
            "outputs": outputs,
        },
    )
    return manifest_path


def invoke_operation(
    step: str,
    operation: Callable[..., Any],
    /,
    *args: Any,
    provenance_arguments: JsonDict | None = None,
    **kwargs: Any,
) -> JsonDict:
    """Call one concrete operation and expose its result through the public contract."""
    provenance: JsonDict | None = None
    try:
        signature = inspect.signature(operation)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        provenance = build_invocation_provenance(
            step,
            provenance_arguments
            if provenance_arguments is not None
            else bound.arguments,
        )

        routed_output_dir: Path | None = None
        if "output_dir" in bound.arguments and str(bound.arguments["output_dir"]).strip():
            routed_output_dir = invocation_output_dir(
                bound.arguments["output_dir"],
                provenance["invocation_id"],
            )
            bound.arguments["output_dir"] = str(routed_output_dir)

        with invocation_provenance(provenance):
            value = operation(*bound.args, **bound.kwargs)
            result = normalise_result(step, value, provenance=provenance)
            if routed_output_dir is not None and result["status"] == "ok":
                manifest_path = _write_invocation_manifest(
                    routed_output_dir,
                    provenance,
                    result["status"],
                )
                manifest_value = str(manifest_path.resolve())
                result["artifacts"]["invocation_manifest_path"] = manifest_value
            return result
    except Exception as exc:
        return error_result(step, exc, provenance=provenance)


__all__ = ["JsonDict", "error_result", "invoke_operation", "normalise_result"]

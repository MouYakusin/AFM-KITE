"""Dependency-light Agent message parsing and tool-call tracing."""

from __future__ import annotations

import json
import posixpath
from typing import Any, Callable, Mapping, Sequence

from ..serialization import flatten_string_leaves, to_jsonable

InputClassifier = Callable[[Any], Sequence[str]]
FILESYSTEM_READ_TOOL_NAMES = frozenset({"ls", "read_file", "glob", "grep"})


def as_text(value: Any) -> str:
    """Return message content as readable text across common content shapes."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(as_text(item) for item in value)
    if isinstance(value, Mapping):
        return str(value.get("text", value.get("content", value)))
    return str(value)


def message_value(value: Any, name: str, default: Any = None) -> Any:
    """Read one field from either a mapping or a message-like object."""
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def json_safe(value: Any) -> Any:
    """Convert nested Agent values into JSON-serialisable primitives."""
    return to_jsonable(value, unknown="string")


def flatten_paths(value: Any) -> list[str]:
    """Flatten the string leaves of an artifact mapping or sequence."""
    return flatten_string_leaves(value)


def physical_input_ids_for_value(
    value: Any,
    inputs: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Return declared input IDs whose physical workspace paths occur in a value."""
    leaves = flatten_string_leaves(json_safe(value))
    candidate_paths = []
    for leaf in leaves:
        candidate = str(leaf).strip()
        if not candidate.startswith("/"):
            continue
        candidate_paths.append(posixpath.normpath(candidate))

    found = []
    for input_info in inputs:
        input_id = str(input_info.get("input_id", "")).strip()
        workspace_path = str(input_info.get("workspace_path", "")).strip()
        if not input_id or not workspace_path.startswith("/inputs/"):
            continue
        physical_path = posixpath.normpath(workspace_path)
        if any(
            candidate == physical_path
            or physical_path.startswith(candidate.rstrip("/") + "/")
            for candidate in candidate_paths
            if candidate == "/inputs" or candidate.startswith("/inputs/")
        ):
            found.append(input_id)
    return found


def parse_tool_payload(message: Any) -> dict[str, Any]:
    """Parse a mapping-shaped tool response from one message."""
    content = message_value(message, "content", {})
    if isinstance(content, Mapping):
        return dict(content)
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


def build_tool_trace(
    messages: Sequence[Any],
    classify_inputs: InputClassifier | None = None,
    protected_inputs: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Record tool arguments, statuses, artifacts, and optional input ownership."""
    classifier = classify_inputs or (lambda value: ())
    trace: list[dict[str, Any]] = []
    calls_by_id: dict[str, dict[str, Any]] = {}
    pending_by_name: dict[str, list[dict[str, Any]]] = {}

    for message in messages:
        for call in message_value(message, "tool_calls", []) or []:
            name = message_value(call, "name", "")
            if not name:
                continue
            arguments = message_value(
                call,
                "args",
                message_value(call, "arguments", {}),
            )
            protected_input_ids = physical_input_ids_for_value(
                arguments,
                protected_inputs,
            )
            entry: dict[str, Any] = {
                "sequence": len(trace) + 1,
                "tool": str(name),
                "call_id": str(message_value(call, "id", "")),
                "arguments": json_safe(arguments),
                "input_ids": sorted(
                    set(classifier(arguments)) | set(protected_input_ids)
                ),
                "status": "not_returned",
                "artifacts": [],
                "_protected_input_ids": protected_input_ids,
            }
            trace.append(entry)
            if entry["call_id"]:
                calls_by_id[entry["call_id"]] = entry
            pending_by_name.setdefault(entry["tool"], []).append(entry)

        call_id = str(message_value(message, "tool_call_id", ""))
        response_name = str(message_value(message, "name", ""))
        entry = calls_by_id.get(call_id)
        if entry is None and response_name:
            entry = next(
                (
                    item
                    for item in pending_by_name.get(response_name, [])
                    if item["status"] == "not_returned"
                ),
                None,
            )
        if entry is None:
            continue

        payload = parse_tool_payload(message)
        if not payload:
            continue
        artifacts = sorted(set(flatten_paths(payload.get("artifacts", {}))))
        entry["status"] = str(payload.get("status", "unknown"))
        entry["artifacts"] = artifacts
        entry["input_ids"] = sorted(
            set(entry["input_ids"]) | set(classifier(artifacts))
        )
        if payload.get("message"):
            entry["message"] = str(payload["message"])
        for name in ("error_code", "recoverable", "details"):
            if name in payload:
                entry[name] = json_safe(payload[name])

    for entry in trace:
        protected_input_ids = entry.pop("_protected_input_ids", [])
        if (
            protected_input_ids
            and entry["tool"] in FILESYSTEM_READ_TOOL_NAMES
            and entry["status"] in {"not_returned", "unknown", "error", "denied"}
        ):
            entry["status"] = "blocked"
            entry["policy_decision"] = "deny"
            entry["execution_started"] = False
            entry["error_code"] = "physical_input_filesystem_read_denied"
            entry["recoverable"] = True
            entry.setdefault(
                "message",
                "Built-in filesystem reads are denied under /inputs; use data_ingest.",
            )
    return trace

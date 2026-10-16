"""Experiment-specific persistence, tracing, artifact indexing, and audits."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from afm_kite_core.runtime.messages import (
    as_text,
    build_tool_trace,
    flatten_paths,
    json_safe,
    message_value,
    parse_tool_payload,
)
from afm_kite_core.tools import TOOL_FUNCTIONS

from .generic_compute import GENERIC_COMPUTE_TOOL_NAME
from .protocol import AGENT_VARIANTS

__all__ = [
    "as_text",
    "append_jsonl",
    "called_tools",
    "dependency_isolation_audit",
    "dependency_observation",
    "file_manifest",
    "flatten_paths",
    "input_ids_for_value",
    "json_safe",
    "message_value",
    "operational_success",
    "parse_tool_payload",
    "quality_run_success",
    "result_artifacts",
    "tool_call_counts",
    "tool_trace",
    "write_audit_index",
    "write_json",
]


RUN_OUTPUT_ROOT = "/outputs"
RUN_SCRATCH_ROOT = "/scratch"
DEDICATED_AFM_TOOL_NAMES = frozenset(function.__name__ for function in TOOL_FUNCTIONS)


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def called_tools(messages: Sequence[Any]) -> list[str]:
    return [entry["tool"] for entry in build_tool_trace(messages)]


def tool_call_counts(tool_calls: Sequence[str]) -> dict[str, int]:
    dedicated_afm = sum(name in DEDICATED_AFM_TOOL_NAMES for name in tool_calls)
    generic_compute = sum(name == GENERIC_COMPUTE_TOOL_NAME for name in tool_calls)
    filesystem_or_other = len(tool_calls) - dedicated_afm - generic_compute
    return {
        "generic_compute": generic_compute,
        "dedicated_afm": dedicated_afm,
        "filesystem_or_other": filesystem_or_other,
        "total": len(tool_calls),
    }


def dependency_observation(answer: str) -> dict[str, bool]:
    lowered = answer.lower()
    return {
        "reported_not_determined": "not determined" in lowered,
        "reported_missing_capability": bool(
            re.search(
                r"(missing|unavailable|required|without).{0,40}(tool|capabilit|workflow)",
                lowered,
            )
        ),
    }


def operational_success(record: Mapping[str, Any]) -> bool:
    """Return runner, artifact, and dependency-isolation compliance."""
    return bool(
        record.get("error") is None
        and record.get("artifact_compliance", {}).get("compliant", False)
        and record.get("dependency_isolation", {}).get("compliant", False)
    )


def quality_run_success(record: Mapping[str, Any]) -> bool:
    """Return whether one attempt fills an experiment-series success slot.

    Scientific correctness is intentionally excluded. Retrying based on the outcome
    score would select favorable answers, so success only requires operational
    completion and a non-empty final answer.
    """

    return bool(operational_success(record) and str(record.get("answer", "")).strip())


def _changed_paths(before: Mapping[str, str], after: Mapping[str, str]) -> list[str]:
    return sorted(path for path in set(before) | set(after) if before.get(path) != after.get(path))


def dependency_isolation_audit(
    workspace: Path,
    method: str,
    tool_calls: Sequence[str],
    trace: Sequence[Mapping[str, Any]],
    before: Mapping[str, str],
    after: Mapping[str, str],
) -> dict[str, Any]:
    """Detect Deterministic AFM Tool leakage into Tools-disabled conditions."""
    variant = AGENT_VARIANTS[method]
    violations = []
    unexpected_calls = sorted(set(tool_calls) & DEDICATED_AFM_TOOL_NAMES)
    if not variant.tool_bundle and unexpected_calls:
        violations.append(
            "Registered Deterministic AFM Tools were called: {}".format(", ".join(unexpected_calls))
        )

    code_suffixes = {".py", ".sh", ".bash", ".zsh"}
    forbidden_patterns = (
        re.compile(r"\b(?:import|from)\s+afmdata_tools\b"),
        re.compile(r"\b(?:import|from)\s+afm_kite_core\.engine\b"),
        re.compile(r"\b(?:import|from)\s+afm_kite_core\.(?:tools|algorithms|data_io)\b"),
        re.compile(r"afm_kite_core[/\\\\]engine[/\\\\]afmdata_tools"),
        re.compile(r"afm_kite_core[/\\\\](?:tools|algorithms|data_io)"),
    )
    if not variant.tool_bundle:
        for entry in trace:
            if entry.get("tool") != GENERIC_COMPUTE_TOOL_NAME:
                continue
            code = str(entry.get("arguments", {}).get("code", ""))
            if any(pattern.search(code) for pattern in forbidden_patterns):
                violations.append("Dedicated AFM implementation referenced by generic_compute")
        for relative in _changed_paths(before, after):
            if not relative.startswith(("outputs/", "scratch/")):
                continue
            path = workspace / relative
            if not path.is_file() or path.suffix.lower() not in code_suffixes:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
            if any(pattern.search(text) for pattern in forbidden_patterns):
                violations.append(
                    "Dedicated AFM implementation referenced by {}".format(relative)
                )
    return {
        "compliant": not violations,
        "tool_bundle": variant.tool_bundle,
        "violations": violations,
    }


def input_ids_for_value(
    value: Any,
    inputs: Sequence[Mapping[str, str]],
    workspace: Path,
) -> list[str]:
    try:
        text = json.dumps(json_safe(value), ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(value)
    found = []
    for input_info in inputs:
        input_id = input_info["input_id"]
        candidates = (
            input_info["source_file"],
            input_info["workspace_path"],
            input_info["workspace_path"].lstrip("/"),
            str(workspace / input_info["workspace_path"].lstrip("/")),
            input_info["artifact_dir"],
            input_info["artifact_dir"].lstrip("/"),
            str(workspace / input_info["artifact_dir"].lstrip("/")),
        )
        if any(candidate and candidate in text for candidate in candidates):
            found.append(input_id)
    return found


def tool_trace(
    messages: Sequence[Any],
    inputs: Sequence[Mapping[str, str]],
    workspace: Path,
) -> list[dict[str, Any]]:
    """Record only tool inputs, statuses, and paths needed for human auditing."""
    return build_tool_trace(
        messages,
        classify_inputs=lambda value: input_ids_for_value(
            value,
            inputs,
            workspace,
        ),
        protected_inputs=inputs,
    )


def result_artifacts(
    workspace: Path,
    inputs: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    per_input = {input_info["input_id"]: [] for input_info in inputs}
    cross_input: list[str] = []
    unclassified: list[str] = []
    outputs_dir_name = RUN_OUTPUT_ROOT.lstrip("/")
    outputs_root = workspace / outputs_dir_name
    if not outputs_root.is_dir():
        return {
            "per_input": per_input,
            "cross_input": cross_input,
            "unclassified": unclassified,
        }
    for path in sorted(outputs_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(workspace).as_posix()
        parts = path.relative_to(workspace).parts
        if (
            len(parts) >= 4
            and parts[:2] == (outputs_dir_name, "by_input")
            and parts[2] in per_input
        ):
            per_input[parts[2]].append(relative)
        elif len(parts) >= 3 and parts[:2] == (outputs_dir_name, "cross_input"):
            cross_input.append(relative)
        else:
            unclassified.append(relative)
    return {
        "per_input": per_input,
        "cross_input": cross_input,
        "unclassified": unclassified,
    }


def markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def write_audit_index(
    workspace: Path,
    case: Any,
    inputs: Sequence[Mapping[str, str]],
    trace: Sequence[Mapping[str, Any]],
    artifact_compliance: Mapping[str, Any],
    dependency_isolation: Mapping[str, Any],
) -> None:
    artifacts = result_artifacts(workspace, inputs)
    lines = [
        "# Audit index",
        "",
        "This index is for human review. It does not certify scientific correctness or completion.",
        "",
        "## Expected input coverage",
        "",
        "| Input ID | Source file | Workspace input | Per-input result artifacts |",
        "|---|---|---|---|",
    ]
    for input_info in inputs:
        paths = artifacts["per_input"].get(input_info["input_id"], [])
        lines.append(
            "| {input_id} | {source_file} | {workspace_path} | {paths} |".format(
                **input_info,
                paths=markdown_cell("<br>".join(paths) if paths else "None"),
            )
        )
    lines.extend(["", "## Cross-input result artifacts", ""])
    lines.extend(
        ("- `{}`".format(path) for path in artifacts["cross_input"])
        if artifacts["cross_input"]
        else ["None"]
    )
    lines.extend(["", "## Unclassified result artifacts", ""])
    lines.extend(
        ("- `{}`".format(path) for path in artifacts["unclassified"])
        if artifacts["unclassified"]
        else ["None"]
    )
    lines.extend(["", "## Artifact policy compliance", ""])
    lines.append("Compliant: `{}`".format(bool(artifact_compliance.get("compliant"))))
    for violation in artifact_compliance.get("violations", []):
        lines.append("- `{path}`: {change}".format(**violation))
    lines.extend(["", "## Dependency isolation", ""])
    lines.append("Compliant: `{}`".format(bool(dependency_isolation.get("compliant"))))
    for violation in dependency_isolation.get("violations", []):
        lines.append(f"- {violation}")
    lines.extend(
        [
            "",
            "## Tool trace summary",
            "",
            "| # | Tool | Input IDs inferred from paths | Status | Returned artifacts |",
            "|---:|---|---|---|---|",
        ]
    )
    if trace:
        for entry in trace:
            lines.append(
                "| {sequence} | {tool} | {input_ids} | {status} | {artifacts} |".format(
                    sequence=entry["sequence"],
                    tool=markdown_cell(entry["tool"]),
                    input_ids=markdown_cell(
                        ", ".join(entry.get("input_ids", [])) or "Not inferred"
                    ),
                    status=markdown_cell(entry.get("status", "unknown")),
                    artifacts=markdown_cell(
                        "<br>".join(entry.get("artifacts", [])) or "None"
                    ),
                )
            )
    else:
        lines.append("| — | No tool calls captured | — | — | — |")
    lines.extend(
        [
            "",
            "## Human review checklist",
            "",
            "- [ ] Every input ID has a result or an objective `Not determined` "
            "outcome in `answer.md`.",
            "- [ ] Every per-input result has a corresponding artifact or a "
            "traceable completed stage.",
            "- [ ] Each cross-input result identifies all contributing input IDs "
            "in its artifact and in `answer.md`.",
            "- [ ] Unclassified artifacts are reviewed before treating the run as complete.",
            "- [ ] Tool errors have an objective limitation or a documented recovery path.",
            "",
            "## Required reporting items",
            "",
        ]
    )
    lines.extend(f"- {metric}" for metric in case.required_metrics)
    (workspace / "audit_index.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def file_manifest(
    workspace: Path,
    inputs: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    entries = []
    for path in sorted(workspace.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        relative = path.relative_to(workspace).as_posix()
        category = "other"
        if relative.startswith("inputs/"):
            category = "input"
        elif relative.startswith(RUN_OUTPUT_ROOT.lstrip("/") + "/"):
            category = "result"
        elif relative.startswith(RUN_SCRATCH_ROOT.lstrip("/") + "/"):
            category = "scratch"
        elif relative.startswith("audit_") or relative == "tool_trace.json":
            category = "audit"
        entries.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "category": category,
                "input_ids": input_ids_for_value(relative, inputs, workspace),
            }
        )
    return entries

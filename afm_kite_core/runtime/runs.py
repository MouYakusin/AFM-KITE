"""AFM-KITE LLM Agent Run recording, finalisation, and execution lifecycle."""

from __future__ import annotations

import json
import signal
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

from .messages import as_text, build_tool_trace, json_safe, message_value
from .usage import TokenUsageCallbackHandler
from .workspace import (
    DEFAULT_OUTPUT_ROOT,
    RunLayout,
    create_run_layout,
    invoke_agent,
    resolve_input_paths,
    snapshot_run_resources,
    validate_skill_source,
)


@dataclass(frozen=True)
class RunOutcome:
    """Successful analysis invocation result and its persisted Run location."""

    answer: str
    layout: RunLayout


AgentFactory = Callable[[RunLayout, Sequence[Mapping[str, Any]]], Any]


class RunTimeoutError(TimeoutError):
    """Raised when one AFM-KITE LLM Agent Run exceeds its safety ceiling."""


@contextmanager
def run_hard_timeout(seconds: float):
    """Interrupt one synchronous AFM-KITE LLM Agent Run at a wall-clock deadline."""
    if seconds <= 0:
        raise ValueError("Run hard timeout must be positive")
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Run hard timeout requires execution on the main thread")
    if not hasattr(signal, "setitimer"):
        raise RuntimeError("Run hard timeout requires signal.setitimer support")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_delay, previous_interval = signal.setitimer(signal.ITIMER_REAL, 0)
    started = time.monotonic()

    def raise_timeout(_signum: int, _frame: Any) -> None:
        raise RunTimeoutError("AFM-KITE LLM Agent Run exceeded its execution limit")

    signal.signal(signal.SIGALRM, raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_delay > 0:
            elapsed = time.monotonic() - started
            signal.setitimer(
                signal.ITIMER_REAL,
                max(previous_delay - elapsed, 1e-6),
                previous_interval,
            )


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
    )


def _file_manifest(layout: RunLayout) -> list[dict[str, Any]]:
    entries = []
    for path in sorted(layout.root.rglob("*")):
        if not path.is_file() or path == layout.manifest_path:
            continue
        relative = layout.relative_path(path)
        category = "other"
        if relative.startswith("inputs/"):
            category = "input"
        elif relative.startswith("outputs/") or relative == "answer.md":
            category = "result"
        elif relative.startswith("scratch/"):
            category = "scratch"
        elif relative in {"tool_trace.json", "model_trace.json"}:
            category = "audit"
        elif relative == "run.json":
            category = "metadata"
        entries.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "category": category,
            }
        )
    return entries


def _write_manifest(
    layout: RunLayout,
    status: str,
    error: Mapping[str, Any] | None = None,
) -> None:
    _write_json(
        layout.manifest_path,
        {
            "run_id": layout.run_id,
            "status": status,
            "request_file": layout.relative_path(layout.request_path),
            "input_manifest": layout.relative_path(layout.input_manifest_path),
            "answer_file": "answer.md",
            "artifact_dir": "outputs",
            "scratch_dir": "scratch",
            "run_record": "run.json",
            "tool_trace": "tool_trace.json",
            "model_trace": "model_trace.json",
            "files": _file_manifest(layout),
            "error": (
                {"type": error.get("type"), "message": error.get("message")}
                if error is not None
                else None
            ),
        },
    )


def _model_metadata(
    model: Any,
    default_name: str,
    default_thinking_mode: str,
    default_temperature: float,
    default_max_retries: int,
) -> dict[str, Any]:
    name = getattr(model, "model_name", getattr(model, "model", default_name))
    extra_body = getattr(model, "extra_body", {})
    thinking = extra_body.get("thinking", {}) if isinstance(extra_body, Mapping) else {}
    thinking_mode = (
        str(thinking.get("type", default_thinking_mode))
        if isinstance(thinking, Mapping)
        else default_thinking_mode
    )
    temperature = getattr(model, "temperature", default_temperature)
    max_retries = getattr(model, "max_retries", default_max_retries)
    return {
        "class": type(model).__name__,
        "name": str(name),
        "thinking_mode": thinking_mode,
        "temperature": json_safe(temperature),
        "max_retries": json_safe(max_retries),
    }


def _finalize_run(
    *,
    layout: RunLayout,
    model: Any,
    model_name: str,
    model_thinking_mode: str,
    model_temperature: float,
    model_max_retries: int,
    capabilities: Mapping[str, Any],
    prompt: str,
    effective_prompt: str,
    inputs: Sequence[Mapping[str, Any]],
    answer: str,
    messages: Sequence[Any],
    token_usage: Mapping[str, Any],
    model_trace: Sequence[Mapping[str, Any]],
    skills: Sequence[str],
    recursion_limit: int,
    max_model_calls: int,
    max_tool_calls: int,
    run_timeout_seconds: float,
    tool_timeout_seconds: float,
    status: str,
    error: Mapping[str, Any] | None,
) -> None:
    trace = build_tool_trace(messages, protected_inputs=inputs)
    answer_text = answer + ("" if not answer or answer.endswith("\n") else "\n")
    _atomic_write_text(layout.answer_path, answer_text)
    _write_json(layout.tool_trace_path, trace)
    _write_json(layout.model_trace_path, model_trace)
    _write_json(
        layout.run_record_path,
        {
            "run_id": layout.run_id,
            "status": status,
            "model": _model_metadata(
                model,
                model_name,
                model_thinking_mode,
                model_temperature,
                model_max_retries,
            ),
            "capabilities": json_safe(capabilities),
            "skills": list(skills),
            "limits": {
                "recursion": recursion_limit,
                "model_calls": max_model_calls,
                "tool_calls": max_tool_calls,
                "run_timeout_seconds": run_timeout_seconds,
                "tool_timeout_seconds": tool_timeout_seconds,
            },
            "prompt": prompt,
            "effective_prompt": effective_prompt,
            "inputs": [dict(input_record) for input_record in inputs],
            "answer": answer,
            "tool_calls": [entry["tool"] for entry in trace],
            "tool_call_count": len(trace),
            "token_usage": json_safe(token_usage),
            "model_trace": layout.relative_path(layout.model_trace_path),
            "workspace": "/",
            "run_root": layout.host_virtual_root,
            "artifact_dir": layout.virtual_output_root,
            "scratch_dir": layout.virtual_scratch_root,
            "error": dict(error) if error is not None else None,
        },
    )
    _write_manifest(
        layout,
        status=status,
        error=error,
    )


def _build_effective_prompt(
    prompt: str,
    inputs: Sequence[Mapping[str, Any]],
) -> str:
    prepared = "\n".join(
        "- {input_id}: {workspace_path} (source label: {source_file})".format(
            **input_record
        )
        for input_record in inputs
    )
    return (
        "The host prepared the following user-supplied input files inside this Run "
        "sandbox. Use the workspace_path values for all file operations; source labels "
        "are metadata only. Do not search outside /inputs for additional project data.\n"
        f"{prepared}\n\n"
        "User request:\n"
        f"{prompt}"
    )


def execute_run(
    *,
    model: Any,
    model_name: str,
    model_thinking_mode: str,
    model_temperature: float,
    model_max_retries: int,
    capabilities: Mapping[str, Any],
    prompt: str,
    agent_factory: AgentFactory,
    recursion_limit: int,
    max_model_calls: int,
    max_tool_calls: int,
    run_timeout_seconds: float,
    tool_timeout_seconds: float,
    skills: Sequence[str],
    input_paths: Sequence[Path],
    project_root: Path,
    skill_source: Path | None,
    output_root: str = DEFAULT_OUTPUT_ROOT,
    run_id: str | None = None,
) -> RunOutcome:
    """Execute and persist one isolated Agent Run."""
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    project_root = project_root.resolve()
    resolved_inputs = resolve_input_paths(project_root, input_paths)
    resolved_skill_source = validate_skill_source(skill_source)
    layout = create_run_layout(project_root, output_root, run_id=run_id)
    inputs = snapshot_run_resources(layout, resolved_inputs, resolved_skill_source)
    request_text = prompt + ("" if prompt.endswith("\n") else "\n")
    _atomic_write_text(layout.request_path, request_text)
    _write_json(layout.input_manifest_path, {"inputs": inputs})
    effective_prompt = _build_effective_prompt(prompt, inputs)
    _write_manifest(layout, status="running")

    answer = ""
    messages: Sequence[Any] = []
    usage_tracker = TokenUsageCallbackHandler()

    def capture_state(state: Any) -> None:
        nonlocal messages
        if not isinstance(state, Mapping):
            return
        state_messages = state.get("messages", [])
        if isinstance(state_messages, Sequence) and not isinstance(
            state_messages, (str, bytes)
        ):
            messages = state_messages

    try:
        with run_hard_timeout(run_timeout_seconds):
            result = invoke_agent(
                agent_factory(layout, inputs),
                effective_prompt,
                recursion_limit,
                layout,
                state_observer=capture_state,
                callbacks=[usage_tracker],
            )
        capture_state(result)
        if messages:
            answer = as_text(message_value(messages[-1], "content", ""))
    except BaseException as exc:
        status = "interrupted" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed"
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(limit=6),
        }
        usage_tracker.finalize_pending(f"Agent Run ended with {type(exc).__name__}")
        _finalize_run(
            layout=layout,
            model=model,
            model_name=model_name,
            model_thinking_mode=model_thinking_mode,
            model_temperature=model_temperature,
            model_max_retries=model_max_retries,
            capabilities=capabilities,
            prompt=prompt,
            effective_prompt=effective_prompt,
            inputs=inputs,
            answer=answer,
            messages=messages,
            token_usage=usage_tracker.summary(),
            model_trace=usage_tracker.calls,
            skills=skills,
            recursion_limit=recursion_limit,
            max_model_calls=max_model_calls,
            max_tool_calls=max_tool_calls,
            run_timeout_seconds=run_timeout_seconds,
            tool_timeout_seconds=tool_timeout_seconds,
            status=status,
            error=error,
        )
        raise

    usage_tracker.finalize_pending("Agent Run returned before a model callback completed")
    _finalize_run(
        layout=layout,
        model=model,
        model_name=model_name,
        model_thinking_mode=model_thinking_mode,
        model_temperature=model_temperature,
        model_max_retries=model_max_retries,
        capabilities=capabilities,
        prompt=prompt,
        effective_prompt=effective_prompt,
        inputs=inputs,
        answer=answer,
        messages=messages,
        token_usage=usage_tracker.summary(),
        model_trace=usage_tracker.calls,
        skills=skills,
        recursion_limit=recursion_limit,
        max_model_calls=max_model_calls,
        max_tool_calls=max_tool_calls,
        run_timeout_seconds=run_timeout_seconds,
        tool_timeout_seconds=tool_timeout_seconds,
        status="completed",
        error=None,
    )
    return RunOutcome(answer=answer, layout=layout)

"""Single-run lifecycle, timeout handling, auditing, and persistence."""

from __future__ import annotations

import signal
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from afm_kite_core.artifacts import (
    ARTIFACT_POLICY_VERSION,
    agent_workspace_guard,
    audit_workspace_changes,
    snapshot_workspace,
)
from afm_kite_core.runtime.usage import TokenUsageCallbackHandler
from afm_kite_core.runtime.workspace import invoke_agent_with_observer

from .agent import (
    MAX_MODEL_CALLS,
    MAX_TOOL_CALLS,
    RUN_HARD_TIMEOUT_SECONDS,
    SKILL_BUNDLE_PATH,
    build_evaluation_agent_variant,
    effective_variant_capabilities,
    factor_assignment,
    model_metadata,
)
from .artifacts import (
    RUN_OUTPUT_ROOT,
    RUN_SCRATCH_ROOT,
    as_text,
    called_tools,
    dependency_isolation_audit,
    dependency_observation,
    file_manifest,
    quality_run_success,
    result_artifacts,
    tool_call_counts,
    tool_trace,
    write_audit_index,
    write_json,
)
from .protocol import AGENT_VARIANTS, FACTOR_SCHEMA_VERSION, experiment_prompt_sections
from .resources import PROJECT_ROOT, Case, prepare_workspace, rel, verify_batch_snapshot


class RunHardTimeoutError(TimeoutError):
    """Raised when one experiment run exceeds its safety ceiling."""


@contextmanager
def run_hard_timeout(seconds: float):
    """Interrupt the core phases of one synchronous experiment run at a deadline."""
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
        raise RunHardTimeoutError("Experiment Run exceeded its execution limit")

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


def make_prompt(case: Case, inputs: Sequence[Mapping[str, str]], template: str) -> str:
    try:
        return template.format(
            inputs="\n".join(
                "- {input_id}: {workspace_path} "
                "(source: {source_file}; per-input artifacts: {artifact_dir})".format(
                    **input_info
                )
                for input_info in inputs
            ),
            task=case.task,
            required_metrics="\n".join(f"- {metric}" for metric in case.required_metrics),
            artifact_dir=RUN_OUTPUT_ROOT,
        )
    except KeyError as exc:
        raise ValueError(
            "common_template only supports {inputs}, {task}, {required_metrics}, "
            "and {artifact_dir}"
        ) from exc


def _snapshot_reference(
    resource_snapshot: Mapping[str, Any] | None,
    *,
    include_locations: bool,
) -> dict[str, Any] | None:
    if resource_snapshot is None:
        return None
    reference = {
        "schema": resource_snapshot.get("schema"),
        "input_content_digest": resource_snapshot.get("input_content_digest"),
    }
    if include_locations:
        reference.update(
            {
                "root": resource_snapshot.get("root"),
                "manifest": resource_snapshot.get("manifest"),
            }
        )
    return reference


def _write_input_inventory(
    workspace: Path,
    run_name: str,
    case: Case,
    inputs: Sequence[Mapping[str, str]],
    resource_snapshot: Mapping[str, Any] | None,
    *,
    complete: bool,
) -> None:
    inventory: dict[str, Any] = {
        "run_name": run_name,
        "case_id": case.case_id,
        "required_reporting_items": list(case.required_metrics),
        "inputs": list(inputs),
    }
    if complete:
        inventory.update(
            {
                "batch_snapshot": _snapshot_reference(
                    resource_snapshot,
                    include_locations=False,
                ),
                "per_input_artifact_root": f"{RUN_OUTPUT_ROOT}/by_input/<input_id>/",
                "cross_input_artifact_root": f"{RUN_OUTPUT_ROOT}/cross_input/",
            }
        )
    write_json(workspace / "audit_input_inventory.json", inventory)


def _artifact_compliance(
    workspace: Path,
    inputs: Sequence[Mapping[str, str]],
    before_state: Mapping[str, str],
    after_state: Mapping[str, str],
) -> dict[str, Any]:
    if not before_state:
        return {
            "compliant": False,
            "evaluated": False,
            "allowed_roots": [RUN_OUTPUT_ROOT, RUN_SCRATCH_ROOT],
            "violations": [{"path": "", "change": "agent_not_started"}],
        }
    compliance = audit_workspace_changes(before_state, after_state)
    unclassified = result_artifacts(workspace, inputs)["unclassified"]
    compliance["violations"].extend(
        {"path": path, "change": "unclassified_output"} for path in unclassified
    )
    compliance["compliant"] = not compliance["violations"]
    compliance["evaluated"] = True
    return compliance


def _final_answer(messages: Sequence[Any]) -> str:
    if not messages:
        return ""
    final_message = messages[-1]
    content = (
        final_message.get("content", "")
        if isinstance(final_message, Mapping)
        else getattr(final_message, "content", "")
    )
    return as_text(content)


def run_case(
    model: Any,
    run_number: int,
    method: str,
    case: Case,
    repetition: int,
    output_dir: Path,
    recursion_limit: int,
    template: str,
    system_prompt: str,
    attempt: int | None = None,
    experiment_series_id: str = "",
    random_selection: Mapping[str, Any] | None = None,
    resource_root: Path = PROJECT_ROOT,
    resource_snapshot: Mapping[str, Any] | None = None,
    agent_builder: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Execute, audit, and persist one isolated experiment attempt."""
    started_at = datetime.now(timezone.utc)
    started_counter = time.perf_counter()
    attempt_number = int(attempt if attempt is not None else repetition)
    run_name = (
        f"{run_number:03d}-{method}-{case.case_id}-success-{repetition:02d}"
        f"-attempt-{attempt_number:02d}"
    )
    workspace = output_dir / "runs" / run_name
    workspace.mkdir(parents=True)
    variant = AGENT_VARIANTS[method]
    prompt = ""
    inputs: list[dict[str, str]] = []
    messages: Sequence[Any] = []
    error = None
    timed_out = False
    usage_tracker = TokenUsageCallbackHandler()
    before_state: dict[str, str] = {}
    after_state: dict[str, str] = {}

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
        with run_hard_timeout(RUN_HARD_TIMEOUT_SECONDS):
            if resource_snapshot is not None:
                verify_batch_snapshot(resource_root, resource_snapshot)
            inputs = prepare_workspace(
                case,
                workspace,
                variant.skill_bundle,
                resource_root=resource_root,
            )
            _write_input_inventory(
                workspace,
                run_name,
                case,
                inputs,
                resource_snapshot,
                complete=True,
            )
            prompt = make_prompt(case, inputs, template)
            builder = agent_builder or build_evaluation_agent_variant
            agent = builder(model, method, workspace, system_prompt, inputs)
            unhashed_roots = (
                (SKILL_BUNDLE_PATH.strip("/"),)
                if variant.skill_bundle
                else ()
            )
            before_state = snapshot_workspace(
                workspace,
                metadata_only_roots=unhashed_roots,
            )
            with agent_workspace_guard(workspace):
                result = invoke_agent_with_observer(
                    agent,
                    prompt,
                    recursion_limit,
                    state_observer=capture_state,
                    callbacks=[usage_tracker],
                )
            capture_state(result)
    except Exception as exc:
        timed_out = isinstance(exc, TimeoutError)
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(limit=6),
        }

    usage_tracker.finalize_pending(
        (
            "Experiment Run returned before a model callback completed"
            if error is None
            else f"Experiment Run ended with {error['type']}"
        )
    )
    model_trace = usage_tracker.calls
    tool_calls = called_tools(messages)
    trace = tool_trace(messages, inputs, workspace)
    answer = _final_answer(messages) if error is None else ""

    if before_state:
        after_state = snapshot_workspace(
            workspace,
            metadata_only_roots=(
                (SKILL_BUNDLE_PATH.strip("/"),)
                if variant.skill_bundle
                else ()
            ),
        )
    artifact_compliance = _artifact_compliance(
        workspace,
        inputs,
        before_state,
        after_state,
    )
    dependency_isolation = dependency_isolation_audit(
        workspace,
        method,
        tool_calls,
        trace,
        before_state,
        after_state,
    )

    (workspace / "answer.md").write_text(answer, encoding="utf-8")
    if not (workspace / "audit_input_inventory.json").exists():
        _write_input_inventory(
            workspace,
            run_name,
            case,
            inputs,
            resource_snapshot,
            complete=False,
        )
    write_json(workspace / "tool_trace.json", trace)
    write_json(workspace / "model_trace.json", model_trace)
    write_audit_index(
        workspace,
        case,
        inputs,
        trace,
        artifact_compliance,
        dependency_isolation,
    )

    finished_at = datetime.now(timezone.utc)
    elapsed_seconds = round(max(time.perf_counter() - started_counter, 0.0), 6)

    record = {
        "run_number": run_number,
        "run_name": run_name,
        "status": "completed" if error is None else "failed",
        "started_at_utc": started_at.isoformat(timespec="milliseconds").replace(
            "+00:00", "Z"
        ),
        "finished_at_utc": finished_at.isoformat(timespec="milliseconds").replace(
            "+00:00", "Z"
        ),
        "elapsed_seconds": elapsed_seconds,
        "timed_out": timed_out,
        "method": method,
        "experiment_series_id": experiment_series_id,
        "factor_schema_version": FACTOR_SCHEMA_VERSION,
        "factor_assignment": factor_assignment(method),
        "effective_capabilities": effective_variant_capabilities(method),
        "prompt_sections": list(experiment_prompt_sections(method)),
        "loaded_skill_paths": [SKILL_BUNDLE_PATH] if variant.skill_bundle else [],
        "model": model_metadata(model),
        "limits": {
            "recursion": recursion_limit,
            "model_calls": MAX_MODEL_CALLS,
            "tool_calls": MAX_TOOL_CALLS,
            "total_tokens": None,
        },
        "case": asdict(case),
        "inputs": inputs,
        "batch_snapshot": _snapshot_reference(resource_snapshot, include_locations=True),
        "repetition": repetition,
        "attempt": attempt_number,
        "random_selection": dict(random_selection or {}),
        "prompt": prompt,
        "answer": answer,
        "tool_calls": tool_calls,
        "tool_call_counts": tool_call_counts(tool_calls),
        "token_usage": usage_tracker.summary(),
        "dependency_observation": dependency_observation(answer),
        "dependency_isolation": dependency_isolation,
        "artifact_compliance": artifact_compliance,
        "artifact_policy_version": ARTIFACT_POLICY_VERSION,
        "workspace": rel(workspace),
        "artifact_dir": rel(workspace / RUN_OUTPUT_ROOT.lstrip("/")),
        "audit": {
            "input_inventory": rel(workspace / "audit_input_inventory.json"),
            "tool_trace": rel(workspace / "tool_trace.json"),
            "model_trace": rel(workspace / "model_trace.json"),
            "index": rel(workspace / "audit_index.md"),
        },
        "error": error,
    }
    record["quality_run_success"] = quality_run_success(record)
    write_json(workspace / "run.json", record)
    write_json(
        workspace / "manifest.json",
        {
            "run_name": run_name,
            "inputs": inputs,
            "files": file_manifest(workspace, inputs),
        },
    )
    return record

"""Stable batch summaries and post-hoc table records."""

from __future__ import annotations

import math
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Callable, Mapping, Sequence

from afm_kite_core.artifacts import ARTIFACT_POLICY_VERSION
from afm_kite_core.artifacts.identity import file_content_identity

from .agent import (
    MAX_MODEL_CALLS,
    MAX_TOOL_CALLS,
    MODEL_MAX_RETRIES,
    MODEL_NAME,
    MODEL_TEMPERATURE,
    MODEL_THINKING_MODE,
    RECURSION_LIMIT,
    SHARED_CAPABILITY_METADATA,
    SKILL_BUNDLE_PATH,
    effective_variant_capabilities,
    factor_assignment,
)
from .artifacts import operational_success
from .protocol import (
    AGENT_VARIANTS,
    FACTOR_DEFINITIONS,
    FACTOR_SCHEMA_VERSION,
    PROMPT_CONTRACT_VERSION,
    build_experiment_system_prompt,
    experiment_prompt_sections,
)
from .resources import PROJECT_ROOT, Case
from .scheduling import CellKey, cell_key

EXPERIMENTS_METHODS = tuple(AGENT_VARIANTS)


def _elapsed_seconds(record: Mapping[str, Any]) -> float | None:
    value = record.get("elapsed_seconds")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    seconds = float(value)
    if seconds < 0 or not math.isfinite(seconds):
        return None
    return seconds


def _record_timed_out(record: Mapping[str, Any]) -> bool:
    explicit = record.get("timed_out")
    if isinstance(explicit, bool):
        return explicit
    error = record.get("error")
    if not isinstance(error, Mapping):
        return False
    error_text = " ".join(
        (str(error.get("type", "")), str(error.get("message", "")))
    ).casefold()
    return "timeout" in error_text or "timed out" in error_text


def _duration_statistics(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = [
        seconds
        for item in items
        if (seconds := _elapsed_seconds(item)) is not None
    ]
    if not values:
        return {
            "count": 0,
            "mean_seconds": None,
            "median_seconds": None,
            "std_seconds": None,
            "min_seconds": None,
            "max_seconds": None,
        }
    return {
        "count": len(values),
        "mean_seconds": round(mean(values), 3),
        "median_seconds": round(median(values), 3),
        "std_seconds": round(pstdev(values), 3),
        "min_seconds": round(min(values), 3),
        "max_seconds": round(max(values), 3),
    }


def _input_identity(cases: Sequence[Case]) -> list[dict[str, Any]]:
    identities = []
    input_paths = sorted({input_path for case in cases for input_path in case.inputs})
    for input_path in input_paths:
        size_bytes, digest = file_content_identity(PROJECT_ROOT / input_path)
        identities.append(
            {
                "path": input_path,
                "size_bytes": size_bytes,
                "blake2b": digest,
            }
        )
    return identities


def build_plan(
    args: Any,
    output_dir: Path,
    discovered_cases: Sequence[Case],
    cases: Sequence[Case],
    system_prompts: Mapping[str, str],
    next_random_draw: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the stable, auditable batch configuration document."""
    return {
        "experiment_name": output_dir.name,
        "factor_schema_version": FACTOR_SCHEMA_VERSION,
        "prompt_contract_version": PROMPT_CONTRACT_VERSION,
        "artifact_policy_version": ARTIFACT_POLICY_VERSION,
        "design": "2x2 component-ablation design with constrained-random order",
        "experiment_series_id": args.experiment_series_id,
        "experiment_series_id_source": args.experiment_series_id_source,
        "factor_definitions": FACTOR_DEFINITIONS,
        "shared_capabilities": SHARED_CAPABILITY_METADATA,
        "group": args.experiment,
        "case_selection": args.case_selection,
        "auto_resume": args.auto_resume,
        "available_case_count": len(discovered_cases),
        "successful_runs_required_per_cell": args.successful_runs,
        "max_attempts_per_cell": args.max_attempts_per_cell,
        "runs_per_batch": args.runs_per_batch,
        "random_seed": args.random_seed,
        "randomization": {
            "method_sampling": "uniform_pending",
            "case_policy": "sequential",
            "draw_unit": "one case-method attempt",
            "candidate_rule": (
                "successes below target and attempts below per-cell limit"
            ),
            "next_draw": dict(next_random_draw) if next_random_draw else None,
        },
        "methods": args.methods,
        "factor_assignments": {
            method: factor_assignment(method) for method in args.methods
        },
        "method_configuration": {
            method: {
                "effective_capabilities": effective_variant_capabilities(method),
                "prompt_sections": list(experiment_prompt_sections(method)),
                "loaded_skill_paths": (
                    [SKILL_BUNDLE_PATH] if AGENT_VARIANTS[method].skill_bundle else []
                ),
            }
            for method in args.methods
        },
        "model": {
            "name": MODEL_NAME,
            "thinking_mode": MODEL_THINKING_MODE,
            "temperature": MODEL_TEMPERATURE,
            "max_retries": MODEL_MAX_RETRIES,
        },
        "limits": {
            "recursion": RECURSION_LIMIT,
            "model_calls": MAX_MODEL_CALLS,
            "tool_calls": MAX_TOOL_CALLS,
            "total_tokens": None,
        },
        "cases": [
            {
                "case_id": case.case_id,
                "group": case.group,
                "inputs": list(case.inputs),
                "task": case.task,
                "required_metrics": list(case.required_metrics),
            }
            for case in cases
        ],
        "input_identity": _input_identity(cases),
        "minimum_planned_invocations": 1 if next_random_draw else 0,
        "maximum_planned_invocations": args.runs_per_batch,
        "maximum_series_invocations": (
            len(cases) * len(args.methods) * args.max_attempts_per_cell
        ),
        "system_prompts": {
            method: build_experiment_system_prompt(system_prompts, method)
            for method in args.methods
        },
    }


def result_for_table(record: Mapping[str, Any], batch_id: str) -> dict[str, Any]:
    """Return one method-explicit record for post-hoc table extraction."""
    error = record.get("error")
    error_summary = None
    if isinstance(error, Mapping):
        error_summary = {
            "type": str(error.get("type", "")),
            "message": str(error.get("message", "")),
        }
    workspace = str(record.get("workspace", ""))
    workspace_path = workspace.rstrip("/")
    return {
        "schema": "afm.quality-table-source/v3",
        "experiment_series_id": str(record.get("experiment_series_id", "")),
        "batch_id": batch_id,
        "run_number": int(record.get("run_number", 0)),
        "run_name": str(record.get("run_name", "")),
        "method": str(record.get("method", "")),
        "factor_assignment": dict(record.get("factor_assignment", {})),
        "quality_run_success": bool(record.get("quality_run_success")),
        "run_status": str(record.get("status", "")),
        "started_at_utc": record.get("started_at_utc"),
        "finished_at_utc": record.get("finished_at_utc"),
        "elapsed_seconds": _elapsed_seconds(record),
        "timed_out": _record_timed_out(record),
        "successful_repetition": int(record.get("repetition", 0)),
        "attempt": int(record.get("attempt", 0)),
        "random_selection": dict(record.get("random_selection", {})),
        "case": dict(record.get("case", {})),
        "inputs": list(record.get("inputs", [])),
        "answer": str(record.get("answer", "")),
        "error": error_summary,
        "tool_call_counts": dict(record.get("tool_call_counts", {})),
        "workspace": workspace,
        "answer_path": f"{workspace_path}/answer.md",
        "run_record_path": f"{workspace_path}/run.json",
    }


def _mean_value(
    items: Sequence[Mapping[str, Any]],
    value: Callable[[Mapping[str, Any]], int | float],
) -> float:
    return round(mean(value(item) for item in items), 3)


def summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"total_runs": len(records), "methods": {}}
    for method in EXPERIMENTS_METHODS:
        items = [item for item in records if item["method"] == method]
        if not items:
            continue
        cache_hit_tokens = sum(
            int(item.get("token_usage", {}).get("prompt_cache_hit_tokens", 0))
            for item in items
        )
        cache_miss_tokens = sum(
            int(item.get("token_usage", {}).get("prompt_cache_miss_tokens", 0))
            for item in items
        )
        cache_denominator = cache_hit_tokens + cache_miss_tokens
        timed_out_items = [item for item in items if _record_timed_out(item)]
        quality_success_items = [
            item for item in items if bool(item.get("quality_run_success"))
        ]
        non_timeout_quality_failure_items = [
            item
            for item in items
            if not bool(item.get("quality_run_success"))
            and not _record_timed_out(item)
        ]
        runs_with_timing = sum(_elapsed_seconds(item) is not None for item in items)
        result["methods"][method] = {
            "attempts": len(items),
            "runs": len(items),
            "successful_quality_runs": sum(
                bool(item.get("quality_run_success")) for item in items
            ),
            "invocation_successful_runs": sum(item["error"] is None for item in items),
            "artifact_compliant_runs": sum(
                bool(item.get("artifact_compliance", {}).get("compliant")) for item in items
            ),
            "operationally_successful_runs": sum(operational_success(item) for item in items),
            "mean_tool_calls": _mean_value(items, lambda item: len(item["tool_calls"])),
            "mean_generic_compute_calls": _mean_value(
                items,
                lambda item: item.get("tool_call_counts", {}).get("generic_compute", 0),
            ),
            "mean_dedicated_afm_tool_calls": _mean_value(
                items,
                lambda item: item.get("tool_call_counts", {}).get("dedicated_afm", 0),
            ),
            "mean_filesystem_or_other_tool_calls": _mean_value(
                items,
                lambda item: item.get("tool_call_counts", {}).get("filesystem_or_other", 0),
            ),
            "mean_model_calls": _mean_value(
                items,
                lambda item: item.get("token_usage", {}).get("model_call_count", 0),
            ),
            "mean_input_tokens": _mean_value(
                items,
                lambda item: item.get("token_usage", {}).get("input_tokens", 0),
            ),
            "mean_output_tokens": _mean_value(
                items,
                lambda item: item.get("token_usage", {}).get("output_tokens", 0),
            ),
            "mean_total_tokens": _mean_value(
                items,
                lambda item: item.get("token_usage", {}).get("total_tokens", 0),
            ),
            "prompt_cache_hit_tokens": cache_hit_tokens,
            "prompt_cache_miss_tokens": cache_miss_tokens,
            "prompt_cache_hit_rate": (
                round(cache_hit_tokens / cache_denominator, 6)
                if cache_denominator
                else None
            ),
            "runs_with_complete_token_usage": sum(
                item.get("token_usage", {}).get("model_call_count", 0) > 0
                and item.get("token_usage", {}).get("missing_usage_call_count", 0) == 0
                for item in items
            ),
            "timing": {
                "unit": "seconds",
                "recorded_runs": runs_with_timing,
                "missing_runs": len(items) - runs_with_timing,
                "timed_out_runs": len(timed_out_items),
                "all_attempts": _duration_statistics(items),
                "quality_successes": _duration_statistics(quality_success_items),
                "non_timeout_quality_failures": _duration_statistics(
                    non_timeout_quality_failure_items
                ),
                "timeouts": _duration_statistics(timed_out_items),
            },
        }
    return result


def quality_run_completion(
    cases: Sequence[Case],
    methods: Sequence[str],
    required_successes: int,
    max_attempts_per_cell: int,
    attempts_by_cell: Mapping[CellKey, int],
    successes_by_cell: Mapping[CellKey, int],
) -> list[dict[str, Any]]:
    """Describe whether every case-method cell filled its requested success slots."""
    completion = []
    for case in cases:
        for method in methods:
            key = cell_key(case.case_id, str(method))
            successful_runs = successes_by_cell.get(key, 0)
            attempts = attempts_by_cell.get(key, 0)
            complete = successful_runs >= required_successes
            attempt_limit_reached = attempts >= max_attempts_per_cell and not complete
            completion.append(
                {
                    "case_id": case.case_id,
                    "method": method,
                    "successful_runs": successful_runs,
                    "required_successful_runs": required_successes,
                    "attempts": attempts,
                    "max_attempts": max_attempts_per_cell,
                    "attempt_limit_reached": attempt_limit_reached,
                    "status": (
                        "complete"
                        if complete
                        else "attempt_limit_reached"
                        if attempt_limit_reached
                        else "pending"
                    ),
                    "complete": complete,
                }
            )
    return completion

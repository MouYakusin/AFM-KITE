"""Cross-batch experiment-series discovery and resume compatibility checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SERIES_COMPATIBILITY_FIELDS = (
    "factor_schema_version",
    "prompt_contract_version",
    "artifact_policy_version",
    "group",
    "successful_runs_required_per_cell",
    "max_attempts_per_cell",
    "random_seed",
    "methods",
    "model",
    "limits",
    "system_prompts",
)


def _canonical_json_value(value: Any) -> Any:
    """Normalize JSON-compatible containers before compatibility comparison."""
    return json.loads(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Cannot read experiment-series record {path}"
        ) from exc
    if not isinstance(value, dict):
        raise RuntimeError(
            f"Experiment-series record must contain an object: {path}"
        )
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RuntimeError(f"Cannot read experiment Runs {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Invalid JSON in experiment Runs {path}:{line_number}"
            ) from exc
        if not isinstance(value, dict):
            raise RuntimeError(
                f"Experiment Run must contain an object: {path}:{line_number}"
            )
        records.append(value)
    return records


def _cases_by_id(plan: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result = {}
    for value in plan.get("cases", []):
        if isinstance(value, Mapping) and value.get("case_id"):
            result[str(value["case_id"])] = value
    return result


def _inputs_by_path(plan: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result = {}
    for value in plan.get("input_identity", []):
        if isinstance(value, Mapping) and value.get("path"):
            result[str(value["path"])] = value
    return result


def validate_series_compatibility(
    current_plan: Mapping[str, Any],
    prior_plan: Mapping[str, Any],
    prior_path: Path,
) -> None:
    """Reject batches that cannot safely contribute to one resumed series."""
    mismatches = [
        field
        for field in SERIES_COMPATIBILITY_FIELDS
        if current_plan.get(field) != prior_plan.get(field)
    ]
    current_cases = _cases_by_id(current_plan)
    prior_cases = _cases_by_id(prior_plan)
    for case_id in sorted(set(current_cases) & set(prior_cases)):
        if _canonical_json_value(current_cases[case_id]) != _canonical_json_value(
            prior_cases[case_id]
        ):
            mismatches.append(f"case:{case_id}")
    current_inputs = _inputs_by_path(current_plan)
    prior_inputs = _inputs_by_path(prior_plan)
    for input_path in sorted(set(current_inputs) & set(prior_inputs)):
        if _canonical_json_value(
            current_inputs[input_path]
        ) != _canonical_json_value(prior_inputs[input_path]):
            mismatches.append(f"input:{input_path}")
    if mismatches:
        joined = ", ".join(mismatches)
        raise ValueError(
            "Experiment series ID "
            f"{current_plan.get('experiment_series_id')!r} is incompatible with "
            f"{prior_path.parent.name}; mismatched fields: {joined}"
        )


def load_resume_records(
    outputs_root: Path,
    experiment_series_id: str,
    current_plan: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Load compatible Run records from one earlier experiment series."""
    records: list[dict[str, Any]] = []
    batch_ids: list[str] = []
    if not outputs_root.is_dir():
        return records, batch_ids
    config_paths = sorted(
        path
        for path in outputs_root.glob("*/config.json")
        if path.parent.is_dir()
    )
    for config_path in config_paths:
        prior_plan = _read_json(config_path)
        if (
            str(prior_plan.get("experiment_series_id", ""))
            != experiment_series_id
        ):
            continue
        validate_series_compatibility(current_plan, prior_plan, config_path)
        runs_path = config_path.parent / "runs.jsonl"
        batch_ids.append(config_path.parent.name)
        if not runs_path.is_file():
            continue
        for record in _read_jsonl(runs_path):
            if (
                str(record.get("experiment_series_id", ""))
                == experiment_series_id
            ):
                records.append(record)
    return records, batch_ids


def series_state(
    experiment_series_id: str,
    prior_batch_ids: Sequence[str],
    prior_record_count: int,
    current_batch_id: str,
    next_draw_index: int,
    random_draws: Sequence[Mapping[str, Any]],
    cell_completion: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the auditable series-resume state stored with the current batch."""
    return {
        "schema": "afm.experiment-series-state/v1",
        "experiment_series_id": experiment_series_id,
        "prior_batches": list(prior_batch_ids),
        "prior_record_count": prior_record_count,
        "current_batch": current_batch_id,
        "next_draw_index": next_draw_index,
        "random_draws_in_current_batch": [dict(item) for item in random_draws],
        "cell_completion": [dict(item) for item in cell_completion],
        "all_cells_complete": all(
            bool(item.get("complete")) for item in cell_completion
        ),
    }

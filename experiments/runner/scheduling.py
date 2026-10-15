"""Reproducible constrained-random experiment scheduling."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Iterable, Mapping, Sequence

from .resources import Case

CellKey = tuple[str, str]


def cell_key(case_id: str, method: str) -> CellKey:
    return str(case_id), str(method)


def execution_progress(
    records: Iterable[Mapping[str, Any]],
) -> tuple[dict[CellKey, int], dict[CellKey, int], int]:
    """Summarize attempts, successes, and the next random draw index."""
    attempts: dict[CellKey, int] = {}
    successes: dict[CellKey, int] = {}
    next_draw_index = 1
    for record in records:
        case = record.get("case", {})
        if not isinstance(case, Mapping):
            continue
        try:
            key = cell_key(str(case["case_id"]), str(record["method"]))
        except KeyError:
            continue
        attempts[key] = attempts.get(key, 0) + 1
        if bool(record.get("quality_run_success")):
            successes[key] = successes.get(key, 0) + 1
        selection = record.get("random_selection", {})
        if isinstance(selection, Mapping):
            try:
                next_draw_index = max(
                    next_draw_index,
                    int(selection.get("draw_index", 0)) + 1,
                )
            except (TypeError, ValueError):
                pass
    return attempts, successes, next_draw_index


def eligible_methods(
    case_id: str,
    methods: Sequence[str],
    target_successes: int,
    max_attempts_per_cell: int,
    attempts_by_cell: Mapping[CellKey, int],
    successes_by_cell: Mapping[CellKey, int],
) -> list[str]:
    """Return methods that still need successes and retain attempt capacity."""
    return [
        str(method)
        for method in methods
        if successes_by_cell.get(cell_key(case_id, str(method)), 0) < target_successes
        and attempts_by_cell.get(cell_key(case_id, str(method)), 0)
        < max_attempts_per_cell
    ]


def _draw_index(
    experiment_series_id: str,
    random_seed: int,
    case_id: str,
    draw_index: int,
    eligible: Sequence[str],
) -> int:
    population_size = len(eligible)
    value_range = 1 << 128
    rejection_limit = value_range - (value_range % population_size)
    nonce = 0
    while True:
        payload = json.dumps(
            {
                "experiment_series_id": experiment_series_id,
                "random_seed": random_seed,
                "case_id": case_id,
                "draw_index": draw_index,
                "eligible_methods": list(eligible),
                "nonce": nonce,
            },
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        value = int.from_bytes(
            hashlib.blake2b(payload, digest_size=16).digest(),
            "big",
        )
        if value < rejection_limit:
            return value % population_size
        nonce += 1


def select_random_draw(
    cases: Sequence[Case],
    methods: Sequence[str],
    target_successes: int,
    max_attempts_per_cell: int,
    random_seed: int,
    experiment_series_id: str,
    draw_index: int,
    attempts_by_cell: Mapping[CellKey, int],
    successes_by_cell: Mapping[CellKey, int],
) -> dict[str, Any] | None:
    """Select one method uniformly from the first unfinished case."""
    for case in cases:
        eligible = eligible_methods(
            case.case_id,
            methods,
            target_successes,
            max_attempts_per_cell,
            attempts_by_cell,
            successes_by_cell,
        )
        if not eligible:
            continue
        selected_method = eligible[
            _draw_index(
                experiment_series_id,
                random_seed,
                case.case_id,
                draw_index,
                eligible,
            )
        ]
        key = cell_key(case.case_id, selected_method)
        return {
            "draw_index": draw_index,
            "random_seed": random_seed,
            "case_id": case.case_id,
            "case_policy": "sequential",
            "method_sampling": "uniform_pending",
            "sampling_algorithm": "blake2b-128-rejection-sampling-v1",
            "eligible_methods_before_draw": eligible,
            "selected_method": selected_method,
            "selection_probability": 1.0 / len(eligible),
            "selection_probability_numerator": 1,
            "selection_probability_denominator": len(eligible),
            "successful_repetition": successes_by_cell.get(key, 0) + 1,
            "attempt": attempts_by_cell.get(key, 0) + 1,
        }
    return None


def execute_random_draws(
    cases: Sequence[Case],
    methods: Sequence[str],
    target_successes: int,
    max_attempts_per_cell: int,
    random_seed: int,
    experiment_series_id: str,
    runs_per_batch: int,
    attempt_runner: Callable[[Mapping[str, Any], int], Mapping[str, Any]],
    *,
    initial_attempts: Mapping[CellKey, int] | None = None,
    initial_successes: Mapping[CellKey, int] | None = None,
    initial_draw_index: int = 1,
) -> tuple[
    list[dict[str, Any]],
    dict[CellKey, int],
    dict[CellKey, int],
    int,
    list[dict[str, Any]],
]:
    """Execute up to ``runs_per_batch`` adaptive constrained-random draws."""
    if target_successes < 1:
        raise ValueError("target_successes must be at least 1")
    if max_attempts_per_cell < target_successes:
        raise ValueError("max_attempts_per_cell must be at least target_successes")
    if runs_per_batch < 1:
        raise ValueError("runs_per_batch must be at least 1")
    attempts_by_cell = dict(initial_attempts or {})
    successes_by_cell = dict(initial_successes or {})
    draw_index = max(1, int(initial_draw_index))
    records: list[dict[str, Any]] = []
    draws: list[dict[str, Any]] = []

    for _ in range(runs_per_batch):
        draw = select_random_draw(
            cases,
            methods,
            target_successes,
            max_attempts_per_cell,
            random_seed,
            experiment_series_id,
            draw_index,
            attempts_by_cell,
            successes_by_cell,
        )
        if draw is None:
            break
        method = str(draw["selected_method"])
        key = cell_key(str(draw["case_id"]), method)
        attempt = attempts_by_cell.get(key, 0) + 1
        record = dict(attempt_runner(draw, attempt))
        records.append(record)
        draws.append(dict(draw))
        attempts_by_cell[key] = attempt
        if bool(record.get("quality_run_success")):
            successes_by_cell[key] = successes_by_cell.get(key, 0) + 1
        draw_index += 1

    return records, attempts_by_cell, successes_by_cell, draw_index, draws


def remaining_attempt_capacity(
    cases: Sequence[Case],
    methods: Sequence[str],
    target_successes: int,
    max_attempts_per_cell: int,
    attempts_by_cell: Mapping[CellKey, int],
    successes_by_cell: Mapping[CellKey, int],
) -> int:
    """Return the remaining attempts available to unfinished cells."""
    capacity = 0
    for case in cases:
        for method in methods:
            key = cell_key(case.case_id, str(method))
            if successes_by_cell.get(key, 0) >= target_successes:
                continue
            capacity += max(0, max_attempts_per_cell - attempts_by_cell.get(key, 0))
    return capacity

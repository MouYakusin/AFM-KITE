#!/usr/bin/env python3
"""Run reproducible AFM-KITE component-ablation experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.runner.agent import RECURSION_LIMIT, build_model
from experiments.runner.artifacts import append_jsonl, write_json
from experiments.runner.execution import run_case
from experiments.runner.protocol import (
    AGENT_VARIANTS,
    FACTOR_SCHEMA_VERSION,
    GROUPS,
    PROMPT_CONTRACT_VERSION,
    load_system_prompts,
    load_task_prompts,
)
from experiments.runner.reporting import (
    build_plan,
    quality_run_completion,
    result_for_table,
    summary,
)
from experiments.runner.resources import (
    EXPERIMENT_OUTPUTS,
    Case,
    create_batch_snapshot,
    discover_cases,
    next_output_dir,
    normalise_case_selection,
    select_cases,
)
from experiments.runner.scheduling import (
    CellKey,
    execute_random_draws,
    execution_progress,
    remaining_attempt_capacity,
    select_random_draw,
)
from experiments.runner.studies import load_resume_records, series_state


# ================================ Configuration ================================
EXPERIMENT_GROUP = "wafer"  # Options: grating, hs100_mg, perovskite, wafer, ag_tio2.
EXPERIMENT_CASE: int | str = 2  # Options: "all" or exact case_id.
EXPERIMENT_METHODS = ("basic_agent", "skill_agent", "tool_agent", "skill_tool_agent")  # Options: basic_agent, skill_agent, tool_agent, skill_tool_agent.
# EXPERIMENT_METHODS = ("skill_agent",)  # Options: basic_agent, skill_agent, tool_agent, skill_tool_agent.
EXPERIMENT_SERIES_ID = "W-M4-T3-S7-V2"  # Custom ID shared by resumable batches; empty auto-generates one.
TARGET_SUCCESSFUL_RUNS = 3  # Required successful Runs per case-method cell.
MAX_ATTEMPTS_PER_CELL = 10  # Attempt ceiling for each case-method cell.
RANDOM_SEED = 7  # Seed for reproducible constrained-random method draws.
RUNS_PER_BATCH = 20  # Maximum attempts executed by one script invocation.
AUTO_RESUME = True  # Reuse prior batches with the same experiment series ID.
DRY_RUN = False  # Print the plan without creating files or invoking the model.
# ============================== End Configuration ==============================

def _case_selection(value: str) -> int | str:
    try:
        return normalise_case_selection(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run AFM case-method attempts selected uniformly from unfinished, "
            "non-exhausted methods."
        )
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--experiment",
        choices=GROUPS,
        default=EXPERIMENT_GROUP,
        help="Experiment group.",
    )
    for name in GROUPS:
        group.add_argument(
            f"--{name.replace('_', '-')}",
            dest="experiment",
            action="store_const",
            const=name,
            help=f"Shortcut for --experiment {name}.",
        )

    case = parser.add_mutually_exclusive_group()
    case.add_argument(
        "--case",
        dest="case_selection",
        type=_case_selection,
        help="One-based case number or 'all'.",
    )
    case.add_argument(
        "--case-id",
        dest="case_selection",
        help="Exact discovered case identifier.",
    )
    case.add_argument(
        "--all-cases",
        dest="case_selection",
        action="store_const",
        const="all",
        help="Select every discovered case in the group.",
    )
    parser.set_defaults(case_selection=EXPERIMENT_CASE)

    parser.add_argument(
        "--methods",
        nargs="+",
        choices=tuple(AGENT_VARIANTS),
        default=list(EXPERIMENT_METHODS),
        help="Methods eligible for random selection, in stable candidate order.",
    )
    parser.add_argument(
        "--successful-runs",
        type=int,
        default=TARGET_SUCCESSFUL_RUNS,
        help="Required successful runs for every selected case-method cell.",
    )
    parser.add_argument(
        "--max-attempts-per-cell",
        type=int,
        default=MAX_ATTEMPTS_PER_CELL,
        help="Safety limit for attempts in each case-method cell.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=RANDOM_SEED,
        help="Seed used by the reproducible constrained-random sampler.",
    )
    parser.add_argument(
        "--runs-per-batch",
        type=int,
        default=RUNS_PER_BATCH,
        help="Maximum random draws executed by this invocation.",
    )
    parser.add_argument(
        "--experiment-series-id",
        default=EXPERIMENT_SERIES_ID,
        help=(
            "Cross-batch experiment series ID; empty derives one from the "
            "effective design."
        ),
    )
    parser.add_argument(
        "--no-resume",
        dest="auto_resume",
        action="store_false",
        help="Ignore earlier batches with the same experiment series ID.",
    )
    parser.set_defaults(auto_resume=AUTO_RESUME)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=DRY_RUN,
        help="Print the resolved plan without creating files or invoking the model.",
    )
    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.successful_runs < 1:
        parser.error("--successful-runs must be at least 1")
    if args.max_attempts_per_cell < args.successful_runs:
        parser.error(
            "--max-attempts-per-cell must be at least --successful-runs"
        )
    if args.runs_per_batch < 1:
        parser.error("--runs-per-batch must be at least 1")
    if not args.methods:
        parser.error("EXPERIMENT_METHODS must contain at least one method")
    unknown_methods = [
        method for method in args.methods if method not in AGENT_VARIANTS
    ]
    if unknown_methods:
        parser.error(
            "Unknown method(s) in EXPERIMENT_METHODS: "
            + ", ".join(unknown_methods)
        )
    if len(args.methods) != len(set(args.methods)):
        parser.error("--methods must not contain duplicates")
def _automatic_experiment_series_id(
    args: argparse.Namespace,
    cases: Sequence[Case],
) -> str:
    payload = {
        "factor_schema_version": FACTOR_SCHEMA_VERSION,
        "prompt_contract_version": PROMPT_CONTRACT_VERSION,
        "group": args.experiment,
        "case_ids": [case.case_id for case in cases],
        "methods": list(args.methods),
        "successful_runs": args.successful_runs,
        "max_attempts_per_cell": args.max_attempts_per_cell,
        "random_seed": args.random_seed,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.blake2b(canonical, digest_size=8).hexdigest()
    return f"{args.experiment}-random-{digest}"


def _next_draw(
    args: argparse.Namespace,
    cases: Sequence[Case],
    attempts: Mapping[CellKey, int],
    successes: Mapping[CellKey, int],
    draw_index: int,
) -> dict[str, Any] | None:
    return select_random_draw(
        cases,
        args.methods,
        args.successful_runs,
        args.max_attempts_per_cell,
        args.random_seed,
        args.experiment_series_id,
        draw_index,
        attempts,
        successes,
    )


def _augment_plan(
    plan: dict[str, Any],
    args: argparse.Namespace,
    cases: Sequence[Case],
    attempts: Mapping[CellKey, int],
    successes: Mapping[CellKey, int],
    prior_batch_ids: Sequence[str],
    prior_record_count: int,
) -> None:
    completion = quality_run_completion(
        cases,
        args.methods,
        args.successful_runs,
        args.max_attempts_per_cell,
        attempts,
        successes,
    )
    capacity = remaining_attempt_capacity(
        cases,
        args.methods,
        args.successful_runs,
        args.max_attempts_per_cell,
        attempts,
        successes,
    )
    plan["resume"] = {
        "enabled": bool(args.auto_resume),
        "prior_batches": list(prior_batch_ids),
        "prior_record_count": prior_record_count,
    }
    plan["cell_completion_before_batch"] = completion
    plan["maximum_planned_invocations"] = min(args.runs_per_batch, capacity)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)

    task_prompts = load_task_prompts()
    system_prompts = load_system_prompts()
    discovered_cases = discover_cases(args.experiment, task_prompts)
    try:
        cases = select_cases(discovered_cases, args.case_selection)
    except ValueError as exc:
        parser.error(str(exc))

    output_dir = next_output_dir(args.experiment)
    if args.experiment_series_id:
        args.experiment_series_id_source = "explicit"
    else:
        args.experiment_series_id = _automatic_experiment_series_id(args, cases)
        args.experiment_series_id_source = "automatic"
        if not args.auto_resume:
            args.experiment_series_id = (
                f"{args.experiment_series_id}-{output_dir.name}"
            )
            args.experiment_series_id_source = "automatic_unique_batch"
    base_plan = build_plan(
        args,
        output_dir,
        discovered_cases,
        cases,
        system_prompts,
    )

    prior_records: list[dict[str, Any]] = []
    prior_batch_ids: list[str] = []
    if args.auto_resume:
        try:
            prior_records, prior_batch_ids = load_resume_records(
                EXPERIMENT_OUTPUTS,
                args.experiment_series_id,
                base_plan,
            )
        except (RuntimeError, ValueError) as exc:
            parser.error(str(exc))

    attempts, successes, draw_index = execution_progress(prior_records)
    selected_draw = _next_draw(args, cases, attempts, successes, draw_index)
    plan = build_plan(
        args,
        output_dir,
        discovered_cases,
        cases,
        system_prompts,
        selected_draw,
    )
    _augment_plan(
        plan,
        args,
        cases,
        attempts,
        successes,
        prior_batch_ids,
        len(prior_records),
    )

    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    if selected_draw is None:
        completion = plan["cell_completion_before_batch"]
        if all(bool(item["complete"]) for item in completion):
            print(
                f"Experiment series {args.experiment_series_id} "
                "is already complete."
            )
            return 0
        print(
            f"Experiment series {args.experiment_series_id} cannot continue: "
            "at least one cell reached its attempt limit before the success target.",
            file=sys.stderr,
        )
        return 2

    output_dir.mkdir(parents=True, exist_ok=False)
    snapshot_root, snapshot_record = create_batch_snapshot(
        cases,
        output_dir,
        include_skill=any(AGENT_VARIANTS[method].skill_bundle for method in args.methods),
    )
    plan["batch_snapshot"] = snapshot_record
    write_json(output_dir / "config.json", plan)

    model = build_model()
    batch_records: list[dict[str, Any]] = []
    table_records: list[dict[str, Any]] = []

    def run_attempt(draw: Mapping[str, Any], attempt: int) -> Mapping[str, Any]:
        case_by_id = {case.case_id: case for case in cases}
        case = case_by_id[str(draw["case_id"])]
        method = str(draw["selected_method"])
        run_number = len(prior_records) + len(batch_records) + 1
        repetition = int(draw["successful_repetition"])
        print(
            f"Running draw {draw['draw_index']}: {case.case_id} × {method} "
            f"(success {repetition}/{args.successful_runs}, attempt {attempt})"
        )
        record = run_case(
            model=model,
            run_number=run_number,
            method=method,
            case=case,
            repetition=repetition,
            attempt=attempt,
            output_dir=output_dir,
            recursion_limit=RECURSION_LIMIT,
            template=task_prompts["common_template"],
            system_prompt=plan["system_prompts"][method],
            experiment_series_id=args.experiment_series_id,
            random_selection=draw,
            resource_root=snapshot_root,
            resource_snapshot=snapshot_record,
        )
        batch_records.append(record)
        table_record = result_for_table(record, output_dir.name)
        table_records.append(table_record)
        append_jsonl(output_dir / "runs.jsonl", record)
        append_jsonl(output_dir / "results_for_table.jsonl", table_record)
        return record

    (
        _,
        final_attempts,
        final_successes,
        next_draw_index,
        random_draws,
    ) = execute_random_draws(
        cases,
        args.methods,
        args.successful_runs,
        args.max_attempts_per_cell,
        args.random_seed,
        args.experiment_series_id,
        args.runs_per_batch,
        run_attempt,
        initial_attempts=attempts,
        initial_successes=successes,
        initial_draw_index=draw_index,
    )

    completion = quality_run_completion(
        cases,
        args.methods,
        args.successful_runs,
        args.max_attempts_per_cell,
        final_attempts,
        final_successes,
    )
    next_after_batch = _next_draw(
        args,
        cases,
        final_attempts,
        final_successes,
        next_draw_index,
    )
    all_complete = all(bool(item["complete"]) for item in completion)
    terminal_incomplete = next_after_batch is None and not all_complete

    batch_summary = summary(batch_records)
    batch_summary.update(
        {
            "experiment_series_id": args.experiment_series_id,
            "random_draws": random_draws,
            "quality_run_completion": completion,
            "all_cells_complete": all_complete,
            "series_terminal": next_after_batch is None,
            "series_terminal_incomplete": terminal_incomplete,
            "next_random_draw": next_after_batch,
            "results_for_table": table_records,
        }
    )
    write_json(output_dir / "summary.json", batch_summary)
    write_json(
        output_dir / "series_state.json",
        series_state(
            args.experiment_series_id,
            prior_batch_ids,
            len(prior_records),
            output_dir.name,
            next_draw_index,
            random_draws,
            completion,
        ),
    )

    if all_complete:
        print(
            f"Completed experiment series {args.experiment_series_id}; "
            f"batch: {output_dir}"
        )
    elif terminal_incomplete:
        print(
            f"Experiment series {args.experiment_series_id} stopped before "
            "completion because at least one cell reached its attempt limit.",
            file=sys.stderr,
        )
    else:
        print(
            f"Completed batch {output_dir}; experiment series "
            f"{args.experiment_series_id} remains pending."
        )
    return 2 if terminal_incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())

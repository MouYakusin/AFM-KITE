#!/usr/bin/env python3
"""AFM-KITE LLM Agent entry point."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from afm_kite_core.identity import PRODUCT_NAME, SKILL_DIR_NAME, SKILL_VIRTUAL_PATH
from afm_kite_core.runtime.agent_profile import (
    DETERMINISTIC_AGENT_CAPABILITIES,
    build_execution_guards,
    build_input_read_permissions,
    configure_deterministic_agent_profile,
)
from afm_kite_core.runtime.compatibility import require_agent_python
from afm_kite_core.runtime.runs import RunOutcome, execute_run
from afm_kite_core.runtime.scoping import build_scoped_tools
from afm_kite_core.runtime.workspace import DEFAULT_OUTPUT_ROOT, RunLayout

ROOT = Path(__file__).resolve().parent


# ================================ Configuration ================================
DEFAULT_INPUT_PATHS: tuple[Path, ...] = (
    # ROOT / "data" / "grating" / "grating2.txt",
    # ROOT / "data" / "wafer" / "wafer2.txt",
    # ROOT / "data" / "Ag-TiO2" / "Ag-TiO2-3-before.txt",
    # ROOT / "data" / "Ag-TiO2" / "Ag-TiO2-3-after.txt",
    ROOT / "data" / "HS100-MG" / "HS100-MG1.txt",
)

MODEL_NAME = "deepseek-v4-pro"
MODEL_THINKING_MODE = "enabled"  # Options: "enabled" or "disabled"
MODEL_TEMPERATURE = 0  # Ignored when MODEL_THINKING_MODE is "enabled"
MODEL_TIMEOUT_SECONDS = 120
MODEL_MAX_RETRIES = 2
RUN_TIMEOUT_SECONDS = 1200
TOOL_TIMEOUT_SECONDS = 180
RECURSION_LIMIT = 200
MAX_MODEL_CALLS = 100
MAX_TOOL_CALLS = 200
DEFAULT_SKILL = SKILL_DIR_NAME
# ============================== End Configuration ==============================

SYSTEM_PROMPT = " ".join(
    (
        "## AFM-KITE Runtime Contract",
        "Use only user-supplied data and artifacts deterministically derived from it as measurement evidence; do not fabricate measurements or artifacts, and state objective limits clearly when evidence is unavailable.",
        "Write persistent analysis artifacts under {output_root} and disposable caches or temporary files under {scratch_root}; do not modify supplied inputs or loaded skills.",
        "The host exclusively owns answer.md, run.json, manifest.json, tool_trace.json, and model_trace.json; do not modify these host-managed run records.",
        "Never use `read_file` on a declared physical AFM or KPFM numeric input, even for a preview or partial read; the runtime denies those exact input paths at the filesystem-tool layer, so pass the path directly to the active deterministic ingestion tool.",
        "Use only tools exposed by the active runtime and treat their schemas and structured results as authoritative. Inspect every returned `status` and every `quality.passed` field when present, consume fields and artifact paths already present in the result directly, and do not reopen an artifact only to recover values already returned.",
        "Construct and execute the smallest deterministic sequence that directly supports the requested reporting items. Compute only required metrics and run only preview or diagnostic stages needed to produce or validate them.",
        "Use a loaded Skill's explicit disclosed fallback when it defines one for an unresolved semantic role, and otherwise ask the user for direction when a newly discovered unresolved input, channel, condition role, or recovery branch would change the required analysis.",
        "For a recoverable error, correct arguments from returned `details` and retry once. Otherwise stop only dependent stages and report the objective failure. Do not repeat a successful stage unless required for validation, recovery, or a distinct required entity or parameter set.",
        "Complete every requested reporting item for every required file, channel, region, or condition; treat an item as complete only when a completed workflow directly supports it or an objective, evidence-backed limitation has been established, since intermediate-stage success alone is insufficient;",
        "return the final answer only after every item meets that condition.",
        "An objective limitation requires missing required input, channel, or metadata; a quality diagnostic that rules out measurement or materially reduces result reliability; an unrecoverable applicable-workflow failure; or unavailable required capability.",
        "For requests that produce analysis results, follow an explicit output format specified by the user when present; if none is specified, use exactly these four top-level Markdown headings in this order: `Workflow run`, `Computed results`, `Key artifacts`, and `Limitations`.",
        "For requests that do not produce analysis results, answer directly without forcing the four-heading analysis format.",
        "Under `Workflow run`, list only analysis stages or tools that were actually executed and include the final status of each attempted stage when known; write `None` if no computation was executed.",
        "Under `Computed results`, provide one top-level bullet for every requested reporting item using its exact name, with nested details when an item covers multiple entities.",
        "For every required file, channel, region, or condition, report an evidence-backed result with units when applicable and identify its supporting artifact when one exists; otherwise, write `Not determined` and state the objective reason.",
        "Under `Key artifacts`, list only existing paths returned by tools or created by completed stages, using workspace-virtual paths under the current output root; do not expose host absolute paths or guess artifact locations.",
        "Write `None` if no artifacts were generated.",
        "Under `Limitations`, report only objective limitations related to input data, metadata, channel availability, data quality, method capability, or execution failure.",
        "Write `None` if no material limitation was identified.",
        "Do not add physical interpretation, causal claims, scientific significance, or subjective sample-quality assessment unless the user explicitly requests it.",
        f"Load `{SKILL_VIRTUAL_PATH}SKILL.md` directly without directory discovery.",
        "Use the Skill only to select AFM/KPFM workflows, define measurement semantics and domain parameters, resolve workflow-specific channel and condition roles through explicit conventions and disclosed fallbacks, and apply metadata-backed physical interpretation and scientific validity gates.",
    )
)


def build_model() -> Any:
    require_agent_python()
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI

    load_dotenv(ROOT / "api.env")
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", MODEL_NAME),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=MODEL_TEMPERATURE,
        extra_body={"thinking": {"type": MODEL_THINKING_MODE}},
        timeout=MODEL_TIMEOUT_SECONDS,
        max_retries=MODEL_MAX_RETRIES,
    )


def build_agent(
    model: Any,
    layout: RunLayout,
    inputs: Sequence[Mapping[str, Any]],
) -> Any:
    from deepagents import create_deep_agent
    from deepagents.backends import FilesystemBackend

    configure_deterministic_agent_profile(f"openai:{MODEL_NAME}")
    return create_deep_agent(
        model=model,
        tools=build_scoped_tools(
            layout.agent_root,
            layout.virtual_output_root,
            timeout_seconds=TOOL_TIMEOUT_SECONDS,
        ),
        backend=FilesystemBackend(
            root_dir=str(layout.agent_root),
            virtual_mode=True,
        ),
        permissions=build_input_read_permissions(inputs),
        skills=[SKILL_VIRTUAL_PATH],
        subagents=[],
        middleware=build_execution_guards(
            max_model_calls=MAX_MODEL_CALLS,
            max_tool_calls=MAX_TOOL_CALLS,
        ),
        system_prompt=SYSTEM_PROMPT.format(
            output_root=layout.virtual_output_root,
            scratch_root=layout.virtual_scratch_root,
        ),
    )


def run_agent(
    model: Any,
    user_prompt: str,
    input_paths: Sequence[Path],
    *,
    recursion_limit: int = RECURSION_LIMIT,
    project_root: Path = ROOT,
    output_root: str = DEFAULT_OUTPUT_ROOT,
    run_id: str | None = None,
) -> RunOutcome:
    return execute_run(
        model=model,
        model_name=MODEL_NAME,
        model_thinking_mode=MODEL_THINKING_MODE,
        model_temperature=MODEL_TEMPERATURE,
        model_max_retries=MODEL_MAX_RETRIES,
        capabilities=DETERMINISTIC_AGENT_CAPABILITIES,
        prompt=user_prompt,
        agent_factory=lambda layout, inputs: build_agent(
            model,
            layout,
            inputs,
        ),
        recursion_limit=recursion_limit,
        max_model_calls=MAX_MODEL_CALLS,
        max_tool_calls=MAX_TOOL_CALLS,
        run_timeout_seconds=RUN_TIMEOUT_SECONDS,
        tool_timeout_seconds=TOOL_TIMEOUT_SECONDS,
        skills=[SKILL_VIRTUAL_PATH],
        input_paths=input_paths,
        project_root=project_root,
        skill_source=ROOT / DEFAULT_SKILL,
        output_root=output_root,
        run_id=run_id,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=f"Run the {PRODUCT_NAME} Agent against explicit project input files."
    )
    parser.add_argument(
        "input_paths",
        nargs="*",
        type=Path,
        help=(
            "Input file(s) inside the project. When omitted, DEFAULT_INPUT_PATHS from agent.py is used."
        ),
    )
    arguments = parser.parse_args(argv)
    input_paths = arguments.input_paths or list(DEFAULT_INPUT_PATHS)
    if not input_paths:
        parser.error(
            "Provide at least one input path or configure DEFAULT_INPUT_PATHS in agent.py"
        )
    model = build_model()
    while True:
        try:
            user_prompt = input("User: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_prompt:
            continue
        try:
            outcome = run_agent(model, user_prompt, input_paths)
        except Exception as exc:
            print(f"Run failed: {exc}", file=sys.stderr)
            continue
        print(outcome.answer)
        print(f"Run saved: {outcome.layout.host_virtual_root}", file=sys.stderr)


if __name__ == "__main__":
    main()

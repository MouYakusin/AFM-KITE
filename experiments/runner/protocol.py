"""Define experiment variants and compose experiment-only prompts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class AgentVariant:
    """AFM-KITE components assigned to one ablation condition."""

    skill_bundle: bool
    tool_bundle: bool


AGENT_VARIANTS = {
    "basic_agent": AgentVariant(skill_bundle=False, tool_bundle=False),
    "skill_agent": AgentVariant(skill_bundle=True, tool_bundle=False),
    "tool_agent": AgentVariant(skill_bundle=False, tool_bundle=True),
    "skill_tool_agent": AgentVariant(skill_bundle=True, tool_bundle=True),
}

FACTOR_DEFINITIONS = {
    "skill_bundle": {
        "description": (
            "AFM Analysis Skill snapshot, references, bundle prompt, and context-loading overhead."
        ),
        "components": [
            "skill_snapshot",
            "skill_references",
            "skill_bundle_system_prompt",
            "skill_loading",
        ],
    },
    "tool_bundle": {
        "description": (
            "Deterministic AFM Tools and their necessary callable schemas."
        ),
        "components": [
            "dedicated_afm_tool_registration",
            "dedicated_afm_tool_schemas",
        ],
    },
}

DEPENDENCY_STUDY_METHODS = (
    "basic_agent",
    "skill_agent",
    "tool_agent",
    "skill_tool_agent",
)

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
PROMPTS = EXPERIMENTS_DIR / "prompts.json"
GROUPS = ("grating", "hs100_mg", "perovskite", "wafer", "ag_tio2")
PROMPT_NAMESPACES = {"task_prompts", "system_prompts"}
FACTOR_SCHEMA_VERSION = "constrained-random-cell-v11"
PROMPT_CONTRACT_VERSION = "afm-prompt-contract-v3"


def _join_prompt_parts(
    value: Any,
    label: str,
    separator: str,
) -> str:
    """Validate and join one source-formatted prompt section."""
    if not isinstance(value, list) or not value or not all(
        isinstance(part, str) and part.strip() for part in value
    ):
        raise ValueError(f"{label} must be a non-empty list of strings")
    return separator.join(value)


def _load_prompt_source() -> dict[str, Any]:
    """Load and validate the shared prompt source document."""
    try:
        source = json.loads(PROMPTS.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read {PROMPTS}") from exc
    if not isinstance(source, dict):
        raise ValueError("prompts.json must contain a JSON object")
    actual_namespaces = set(source)
    if actual_namespaces != PROMPT_NAMESPACES:
        missing = sorted(PROMPT_NAMESPACES - actual_namespaces)
        unknown = sorted(actual_namespaces - PROMPT_NAMESPACES)
        raise ValueError(
            f"prompts.json namespaces mismatch; missing={missing}, unknown={unknown}"
        )
    return source


def load_task_prompts() -> dict[str, Any]:
    prompts = _load_prompt_source()["task_prompts"]
    if not isinstance(prompts, dict):
        raise ValueError("prompts.json task_prompts must contain a JSON object")
    prompts["common_template"] = _join_prompt_parts(
        prompts.get("common_template"),
        "prompts.json task_prompts.common_template",
        "",
    )
    if not isinstance(prompts.get("groups"), dict):
        raise ValueError("prompts.json task_prompts requires groups")
    for group in GROUPS:
        group_prompt = prompts["groups"].get(group, {})
        if not isinstance(group_prompt, dict):
            raise ValueError(
                f"prompts.json task_prompts group {group} must be an object"
            )
        group_prompt["task"] = _join_prompt_parts(
            group_prompt.get("task"),
            f"prompts.json task_prompts task for {group}",
            " ",
        )
        required_metrics = group_prompt.get("required_metrics")
        if not isinstance(required_metrics, list) or not required_metrics or not all(
            isinstance(metric, str) and metric.strip() for metric in required_metrics
        ):
            raise ValueError(
                "prompts.json task_prompts requires non-empty "
                f"required_metrics for {group}"
            )
    return prompts


def load_system_prompts() -> dict[str, str]:
    required_keys = {
        "common_system_prompt",
        "skill_bundle_system_prompt",
    }
    prompts = _load_prompt_source()["system_prompts"]
    if not isinstance(prompts, dict):
        raise ValueError("prompts.json system_prompts must contain a JSON object")
    actual_keys = set(prompts)
    if actual_keys != required_keys:
        missing = sorted(required_keys - actual_keys)
        unknown = sorted(actual_keys - required_keys)
        raise ValueError(
            "prompts.json system_prompts keys mismatch; "
            f"missing={missing}, unknown={unknown}"
        )
    return {
        key: _join_prompt_parts(
            prompts[key],
            f"prompts.json system_prompts.{key}",
            "\n\n",
        )
        for key in required_keys
    }


def experiment_prompt_sections(method: str) -> tuple[str, ...]:
    """Return the ordered prompt-section names enabled for one method."""
    variant = AGENT_VARIANTS[method]
    sections = ["common"]
    if variant.skill_bundle:
        sections.append("skill_bundle")
    return tuple(sections)


def build_experiment_system_prompt(
    prompts: Mapping[str, str],
    method: str,
) -> str:
    prompt_keys = {
        "common": "common_system_prompt",
        "skill_bundle": "skill_bundle_system_prompt",
    }
    return "\n\n".join(
        prompts[prompt_keys[section]]
        for section in experiment_prompt_sections(method)
    )

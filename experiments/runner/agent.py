"""Evaluation Agent Variant model configuration and factory."""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from afm_kite_core.artifacts import ArtifactPolicy
from afm_kite_core.identity import SKILL_VIRTUAL_PATH
from afm_kite_core.runtime.agent_profile import (
    DETERMINISTIC_AGENT_CAPABILITIES,
    DETERMINISTIC_MAX_MODEL_CALLS,
    DETERMINISTIC_MAX_TOOL_CALLS,
    DETERMINISTIC_RECURSION_LIMIT,
    DETERMINISTIC_RUN_TIMEOUT_SECONDS,
    DETERMINISTIC_TOOL_TIMEOUT_SECONDS,
    build_execution_guards,
    build_input_read_permissions,
    configure_deterministic_agent_profile,
)
from afm_kite_core.runtime.compatibility import require_agent_python
from afm_kite_core.runtime.scoping import build_scoped_tools

from .artifacts import RUN_OUTPUT_ROOT, json_safe
from .generic_compute import build_generic_compute_tool
from .protocol import AGENT_VARIANTS

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ================================ Configuration ================================
MODEL_NAME = "deepseek-v4-pro"  # Model used by every experiment cell
MODEL_THINKING_MODE = "disabled"  # Options: "enabled" or "disabled"
MODEL_TEMPERATURE = 0  # Ignored when MODEL_THINKING_MODE is "enabled"
MODEL_TIMEOUT_SECONDS = 120  # Timeout for one model request, in seconds
MODEL_MAX_RETRIES = 2  # Maximum retries for one model request
RUN_HARD_TIMEOUT_SECONDS = DETERMINISTIC_RUN_TIMEOUT_SECONDS
TOOL_TIMEOUT_SECONDS = DETERMINISTIC_TOOL_TIMEOUT_SECONDS
RECURSION_LIMIT = DETERMINISTIC_RECURSION_LIMIT
MAX_MODEL_CALLS = DETERMINISTIC_MAX_MODEL_CALLS
MAX_TOOL_CALLS = DETERMINISTIC_MAX_TOOL_CALLS
SKILL_BUNDLE_PATH = SKILL_VIRTUAL_PATH  # Agent-visible Skill path
# ============================== End Configuration ==============================

# These capability flags describe the baseline assigned to every factorial cell.
SHARED_CAPABILITY_METADATA = {
    **DETERMINISTIC_AGENT_CAPABILITIES,
    "generic_compute": True,
    "numpy": True,
    "scipy": True,
    "artifact_policy": True,
    "result_compaction": True,
}


def model_metadata(model: Any) -> dict[str, Any]:
    """Capture the effective experiment model configuration for audit records."""
    extra_body = getattr(model, "extra_body", {})
    thinking = extra_body.get("thinking", {}) if isinstance(extra_body, Mapping) else {}
    thinking_mode = (
        str(thinking.get("type", MODEL_THINKING_MODE))
        if isinstance(thinking, Mapping)
        else MODEL_THINKING_MODE
    )
    return {
        "class": type(model).__name__,
        "name": str(getattr(model, "model_name", getattr(model, "model", MODEL_NAME))),
        "thinking_mode": thinking_mode,
        "temperature": json_safe(getattr(model, "temperature", MODEL_TEMPERATURE)),
        "max_retries": json_safe(getattr(model, "max_retries", MODEL_MAX_RETRIES)),
    }


def factor_assignment(method: str) -> dict[str, bool]:
    """Return the two randomized capability-bundle assignments."""
    return asdict(AGENT_VARIANTS[method])


def effective_variant_capabilities(method: str) -> dict[str, bool]:
    """Return shared and assigned capabilities enabled for one method."""
    variant = AGENT_VARIANTS[method]
    return {
        **SHARED_CAPABILITY_METADATA,
        "skill_bundle": variant.skill_bundle,
        "dedicated_afm_tools": variant.tool_bundle,
    }


def build_model() -> Any:
    """Create the fixed model configuration used by every experiment cell."""
    require_agent_python()
    from dotenv import load_dotenv
    from langchain_openai import ChatOpenAI

    load_dotenv(PROJECT_ROOT / "api.env")
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", MODEL_NAME),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=MODEL_TEMPERATURE,
        extra_body={"thinking": {"type": MODEL_THINKING_MODE}},
        timeout=MODEL_TIMEOUT_SECONDS,
        max_retries=MODEL_MAX_RETRIES,
    )


def build_evaluation_agent_variant(
    model: Any,
    method: str,
    workspace: Path,
    system_prompt: str,
    inputs: Sequence[Mapping[str, str]],
) -> Any:
    """Build one isolated Evaluation Agent Variant for the selected method."""
    from deepagents import create_deep_agent
    from deepagents.backends import FilesystemBackend

    variant = AGENT_VARIANTS[method]
    artifact_policy = ArtifactPolicy.experiment(
        workspace=workspace,
        inputs=inputs,
        output_root=RUN_OUTPUT_ROOT,
    )
    configure_deterministic_agent_profile(f"openai:{MODEL_NAME}")

    # Every method receives the shared restricted NumPy/SciPy baseline.
    tools = [
        build_generic_compute_tool(
            workspace,
            RUN_OUTPUT_ROOT,
            artifact_policy,
            timeout_seconds=TOOL_TIMEOUT_SECONDS,
        )
    ]

    # Dedicated AFM tools are an experimental treatment and remain method-specific.
    if variant.tool_bundle:
        tools.extend(
            build_scoped_tools(
                workspace,
                RUN_OUTPUT_ROOT,
                artifact_policy=artifact_policy,
                timeout_seconds=TOOL_TIMEOUT_SECONDS,
            )
        )

    skills = [SKILL_BUNDLE_PATH] if variant.skill_bundle else []
    return create_deep_agent(
        model=model,
        tools=tools,
        skills=skills,
        subagents=[],
        backend=FilesystemBackend(root_dir=str(workspace), virtual_mode=True),
        permissions=build_input_read_permissions(inputs),
        middleware=build_execution_guards(
            max_model_calls=MAX_MODEL_CALLS,
            max_tool_calls=MAX_TOOL_CALLS,
        ),
        system_prompt=system_prompt,
    )

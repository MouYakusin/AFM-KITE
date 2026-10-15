"""Shared Deep Agents capability restrictions for deterministic AFM runs."""

from __future__ import annotations

from functools import lru_cache
from types import MappingProxyType
from typing import Any, Mapping, Sequence

DETERMINISTIC_RUN_TIMEOUT_SECONDS = 1200
DETERMINISTIC_TOOL_TIMEOUT_SECONDS = 180
DETERMINISTIC_RECURSION_LIMIT = 200
DETERMINISTIC_MAX_MODEL_CALLS = 100
DETERMINISTIC_MAX_TOOL_CALLS = 200

DETERMINISTIC_AGENT_CAPABILITIES: Mapping[str, bool] = MappingProxyType(
    {
        "subagents_enabled": False,
        "filesystem_write_tools_enabled": False,
        "physical_input_filesystem_reads_denied": True,
        "determined_workflows_execute_without_todos": True,
    }
)


def build_execution_guards(
    max_model_calls: int = DETERMINISTIC_MAX_MODEL_CALLS,
    max_tool_calls: int = DETERMINISTIC_MAX_TOOL_CALLS,
) -> list[Any]:
    """Return fresh per-Agent model and tool call limits."""
    from langchain.agents.middleware import (
        ModelCallLimitMiddleware,
        ToolCallLimitMiddleware,
    )

    return [
        ModelCallLimitMiddleware(run_limit=max_model_calls, exit_behavior="error"),
        ToolCallLimitMiddleware(run_limit=max_tool_calls, exit_behavior="error"),
    ]


def build_input_read_permissions(
    inputs: Sequence[Mapping[str, Any]],
) -> list[Any]:
    """Deny built-in filesystem-tool reads anywhere under the input tree."""
    has_inputs = False
    for input_info in inputs:
        workspace_path = str(input_info.get("workspace_path", "")).strip()
        if not workspace_path.startswith("/"):
            raise ValueError(
                "Every Agent input must declare an absolute virtual workspace_path"
            )
        if not workspace_path.startswith("/inputs/"):
            raise ValueError(
                "Every Agent input workspace_path must stay under /inputs"
            )
        has_inputs = True

    if not has_inputs:
        return []

    from deepagents import FilesystemPermission

    return [
        FilesystemPermission(
            operations=["read"],
            paths=["/inputs/**"],
            mode="deny",
        )
    ]


@lru_cache(maxsize=None)
def configure_deterministic_agent_profile(profile_key: str) -> None:
    """Disable Subagents and model-authored files for one exact model profile."""
    from deepagents import GeneralPurposeSubagentProfile, HarnessProfile, register_harness_profile

    register_harness_profile(
        profile_key,
        HarnessProfile(
            excluded_tools=frozenset({"write_file", "edit_file", "write_todos"}),
            general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
        ),
    )

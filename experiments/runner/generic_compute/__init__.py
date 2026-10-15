"""Public interface for the runner's restricted generic-compute capability."""

from .controller import (
    GENERIC_COMPUTE_MAX_INPUT_FILES,
    GENERIC_COMPUTE_MAX_OUTPUT_FILES,
    GENERIC_COMPUTE_TOOL_NAME,
    build_generic_compute_tool,
    generic_compute,
)

__all__ = [
    "GENERIC_COMPUTE_TOOL_NAME",
    "GENERIC_COMPUTE_MAX_INPUT_FILES",
    "GENERIC_COMPUTE_MAX_OUTPUT_FILES",
    "build_generic_compute_tool",
    "generic_compute",
]

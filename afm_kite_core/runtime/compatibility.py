"""Runtime compatibility checks for the optional model-backed Agent stack."""

from __future__ import annotations

import sys
from typing import Sequence

MIN_AGENT_PYTHON = (3, 11)


def require_agent_python(version_info: Sequence[int] | None = None) -> None:
    """Reject Python versions unsupported by the model-backed Agent runtimes."""

    current = tuple((version_info or sys.version_info)[:2])
    if current < MIN_AGENT_PYTHON:
        required = ".".join(str(part) for part in MIN_AGENT_PYTHON)
        detected = ".".join(str(part) for part in current)
        raise RuntimeError(
            f"The AFM-KITE LLM Agent and Evaluation Runner require Python {required}+; "
            f"detected Python {detected}. The deterministic afm_kite_core "
            "package remains supported on Python 3.9+."
        )


__all__ = ["MIN_AGENT_PYTHON", "require_agent_python"]

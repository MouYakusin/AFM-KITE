"""Deterministic execution layer for the AFM-KITE framework.

Agent-specific configuration lives in the project entry points.
This package provides reusable public tools, in-memory algorithms, and AFM data I/O.
It also provides message tracing, Run recording, and artifact safeguards.
"""

from .artifacts import ARTIFACT_POLICY_VERSION, ArtifactPolicy
from ._version import __version__

__all__ = ["ARTIFACT_POLICY_VERSION", "ArtifactPolicy", "__version__"]

"""Filesystem helpers shared by data readers and deterministic tools."""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Iterator

PROJECT_ROOT = Path(os.environ.get("AFMDATA_WORKDIR", Path.cwd())).resolve()
_ACTIVE_WORKSPACE_ROOT: ContextVar[Path | None] = ContextVar(
    "afm_active_workspace_root",
    default=None,
)


@contextmanager
def scoped_workspace(root: str | Path) -> Iterator[None]:
    """Restrict nested artifact reads to ``root`` for one scoped tool call."""

    token = _ACTIVE_WORKSPACE_ROOT.set(Path(root).expanduser().resolve())
    try:
        yield
    finally:
        _ACTIVE_WORKSPACE_ROOT.reset(token)


def ensure_output_dir(output_dir: str | Path) -> Path:
    """Resolve and create an output directory under the active work directory."""

    path = Path(output_dir).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def resolve_readable_file(path: str | Path) -> Path:
    """Resolve a readable file against the current and configured work directories."""

    candidate = Path(path).expanduser()
    candidates = [candidate]
    if not candidate.is_absolute():
        candidates.append(PROJECT_ROOT / candidate)

    for item in candidates:
        if item.exists() and item.is_file():
            resolved = item.resolve()
            workspace_root = _ACTIVE_WORKSPACE_ROOT.get()
            if workspace_root is not None and (
                resolved != workspace_root and workspace_root not in resolved.parents
            ):
                raise PermissionError(
                    f"Artifact path must stay inside {workspace_root}: {resolved}"
                )
            return resolved

    raise FileNotFoundError(
        f"Input file not found: {candidate}. Use an existing AFM txt, TSV, or NPY path."
    )


__all__ = [
    "PROJECT_ROOT",
    "ensure_output_dir",
    "resolve_readable_file",
    "scoped_workspace",
]

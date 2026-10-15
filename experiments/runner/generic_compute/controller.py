"""Restricted NumPy/SciPy capability shared by every experiment method."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from functools import wraps
from pathlib import Path
from typing import Any

from afm_kite_core.artifacts import ArtifactPolicy
from afm_kite_core.artifacts.identity import file_content_identity
from afm_kite_core.runtime.scoping import scope_callable
from afm_kite_core.tools.boundary import invoke_operation

GENERIC_COMPUTE_TOOL_NAME = "generic_compute"
DEFAULT_GENERIC_COMPUTE_TIMEOUT_SECONDS = 180
GENERIC_COMPUTE_MAX_CODE_CHARACTERS = 50_000
GENERIC_COMPUTE_MAX_INPUT_FILES = 32
GENERIC_COMPUTE_MAX_OUTPUT_FILES = 64
GENERIC_COMPUTE_MAX_OUTPUT_FILE_BYTES = 256 * 1024 * 1024
GENERIC_COMPUTE_MAX_TOTAL_OUTPUT_BYTES = 512 * 1024 * 1024
WORKER = Path(__file__).with_name("worker.py")


def _worker_environment() -> dict[str, str]:
    environment = {
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    for name in ("PATH", "SYSTEMROOT"):
        if name in os.environ:
            environment[name] = os.environ[name]
    return environment


def _file_snapshot(output_dir: Path) -> dict[str, tuple[int, str]]:
    if not output_dir.is_dir():
        return {}
    snapshot: dict[str, tuple[int, str]] = {}
    for path in output_dir.rglob("*"):
        if path.is_file():
            snapshot[path.relative_to(output_dir).as_posix()] = file_content_identity(
                path
            )
    return snapshot


def _generated_artifacts(
    output_dir: Path,
    before: dict[str, tuple[int, str]],
) -> tuple[list[str], str | None]:
    after = _file_snapshot(output_dir)
    changed = sorted(
        relative for relative, signature in after.items() if before.get(relative) != signature
    )
    artifacts = [str((output_dir / relative).resolve()) for relative in changed]
    if len(changed) > GENERIC_COMPUTE_MAX_OUTPUT_FILES:
        return artifacts, (
            "generic_compute produced more than "
            f"{GENERIC_COMPUTE_MAX_OUTPUT_FILES} output files"
        )
    sizes = [after[relative][0] for relative in changed]
    if any(size > GENERIC_COMPUTE_MAX_OUTPUT_FILE_BYTES for size in sizes):
        return artifacts, "generic_compute produced an output file above the size limit"
    if sum(sizes) > GENERIC_COMPUTE_MAX_TOTAL_OUTPUT_BYTES:
        return artifacts, "generic_compute exceeded the total output size limit"
    return artifacts, None


def _generic_compute_operation(
    code: str,
    input_paths: list[str],
    output_dir: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    if not isinstance(code, str) or not code.strip():
        return {
            "status": "error",
            "step": GENERIC_COMPUTE_TOOL_NAME,
            "message": "code must be a non-empty string",
            "artifacts": {},
            "summary": {},
        }
    if len(code) > GENERIC_COMPUTE_MAX_CODE_CHARACTERS:
        return {
            "status": "error",
            "step": GENERIC_COMPUTE_TOOL_NAME,
            "message": "code exceeds the generic-compute size limit",
            "artifacts": {},
            "summary": {},
        }
    if not isinstance(input_paths, list) or not input_paths:
        return {
            "status": "error",
            "step": GENERIC_COMPUTE_TOOL_NAME,
            "message": "input_paths must declare at least one source or upstream artifact",
            "artifacts": {},
            "summary": {},
        }
    if len(input_paths) > GENERIC_COMPUTE_MAX_INPUT_FILES:
        return {
            "status": "error",
            "step": GENERIC_COMPUTE_TOOL_NAME,
            "message": (
                "input_paths exceeds the generic-compute limit of "
                f"{GENERIC_COMPUTE_MAX_INPUT_FILES} files"
            ),
            "artifacts": {},
            "summary": {},
        }

    target = Path(output_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    before = _file_snapshot(target)
    request = {
        "code": code,
        "input_paths": [str(Path(path).resolve()) for path in input_paths],
        "output_dir": str(target),
    }
    try:
        completed = subprocess.run(
            [sys.executable, "-I", str(WORKER)],
            input=json.dumps(request),
            text=True,
            capture_output=True,
            cwd=target,
            env=_worker_environment(),
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "step": GENERIC_COMPUTE_TOOL_NAME,
            "message": "generic_compute exceeded its execution limit",
            "error_code": "generic_compute_timeout",
            "artifacts": {},
            "summary": {},
        }

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        payload = {
            "status": "error",
            "message": completed.stderr.strip() or "generic_compute worker returned invalid JSON",
            "summary": {},
        }
    if not isinstance(payload, dict):
        payload = {
            "status": "error",
            "message": "generic_compute worker returned a non-object result",
            "summary": {},
        }

    artifacts, artifact_violation = _generated_artifacts(target, before)
    result: dict[str, Any] = {
        "status": (
            "ok"
            if completed.returncode == 0
            and payload.get("status") == "ok"
            and artifact_violation is None
            else "error"
        ),
        "step": GENERIC_COMPUTE_TOOL_NAME,
        "summary": payload.get("summary", {}),
        "artifacts": {"files": artifacts} if artifacts else {},
        "execution": payload.get("execution", {}),
    }
    if result["status"] == "error":
        result["message"] = str(
            artifact_violation
            or payload.get("message")
            or completed.stderr.strip()
            or "generic_compute worker failed"
        )
        if artifact_violation:
            result["error_code"] = "generic_compute_output_limit"
        if payload.get("error_type"):
            result["error_type"] = str(payload["error_type"])
    return result


def _invoke_generic_compute(
    code: str,
    input_paths: list[str],
    output_dir: str,
    timeout_seconds: float,
) -> dict[str, Any]:
    return invoke_operation(
        "generic-compute",
        _generic_compute_operation,
        code,
        input_paths,
        output_dir,
        timeout_seconds,
    )


def generic_compute(
    code: str,
    input_paths: list[str],
    output_dir: str = "/outputs/data/generic-compute",
) -> dict[str, Any]:
    """Run deterministic NumPy/SciPy code in a restricted worker.

    NumPy, SciPy, ``math``, and ``json`` are preloaded. Conventional imports of
    ``numpy``, ``math``, ``json``, and these SciPy modules are accepted:
    ``fft``, ``interpolate``, ``linalg``, ``ndimage``, ``optimize``, ``signal``,
    ``spatial``, and ``stats``. They resolve to safe preloaded modules.
    The submitted code can access ``input_paths`` and ``output_dir``.
    ``open`` may read declared ``input_paths`` and may write only below
    ``output_dir``. Arbitrary imports, undeclared paths, networking, and shell
    commands are unavailable.
    Store the final JSON-compatible output in ``result``.
    Write persistent artifacts only below ``output_dir``.
    """
    return _invoke_generic_compute(
        code,
        input_paths,
        output_dir,
        DEFAULT_GENERIC_COMPUTE_TIMEOUT_SECONDS,
    )


def build_generic_compute_tool(
    root: Path,
    output_root: str,
    artifact_policy: ArtifactPolicy,
    *,
    timeout_seconds: float,
) -> Any:
    """Build the shared generic-compute StructuredTool under experiment policy."""
    if timeout_seconds <= 0:
        raise ValueError("generic-compute timeout must be positive")

    try:
        from langchain_core.tools import StructuredTool
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("langchain-core is required to build generic_compute") from exc

    @wraps(generic_compute)
    def configured_generic_compute(
        code: str,
        input_paths: list[str],
        output_dir: str = "/outputs/data/generic-compute",
    ) -> dict[str, Any]:
        return _invoke_generic_compute(
            code,
            input_paths,
            output_dir,
            timeout_seconds,
        )

    tool = StructuredTool.from_function(configured_generic_compute)
    return tool.model_copy(
        update={
            "func": scope_callable(
                tool.func,
                root,
                output_root,
                artifact_policy=artifact_policy,
            )
        }
    )


__all__ = [
    "GENERIC_COMPUTE_TOOL_NAME",
    "GENERIC_COMPUTE_MAX_INPUT_FILES",
    "GENERIC_COMPUTE_MAX_OUTPUT_FILES",
    "build_generic_compute_tool",
    "generic_compute",
]

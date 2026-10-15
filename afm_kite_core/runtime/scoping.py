"""Workspace-scoped deterministic tool construction without model dependencies."""

from __future__ import annotations

import inspect
import multiprocessing
import traceback
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Sequence

from ..artifacts import ArtifactPolicy, ArtifactPolicyError, WorkspacePathError
from ..data_io.paths import scoped_workspace
from ..tools import get_tools
from .workspace import DEFAULT_OUTPUT_ROOT

ROOT = Path(__file__).resolve().parents[1]
MODEL_LIST_ITEM_LIMIT = 32


def _tool_process_entry(
    function: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    root: Path,
    connection: Any,
) -> None:
    """Execute one deterministic tool in an isolated, terminable process."""

    try:
        with scoped_workspace(root):
            result = function(*args, **kwargs)
        connection.send(("ok", compact_model_result(result)))
    except BaseException as exc:  # pragma: no cover - exercised through parent result
        connection.send(
            (
                "error",
                type(exc).__name__,
                str(exc),
                traceback.format_exc(limit=8),
            )
        )
    finally:
        connection.close()


def _invoke_with_process_timeout(
    function: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    root: Path,
    timeout_seconds: float,
    step: str,
) -> Any:
    """Run one tool in a process that can be terminated at the shared deadline."""

    context = multiprocessing.get_context("spawn")
    receive_connection, send_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_tool_process_entry,
        args=(function, args, kwargs, root, send_connection),
    )
    process.start()
    send_connection.close()
    try:
        if not receive_connection.poll(timeout_seconds):
            process.terminate()
            process.join()
            return {
                "status": "error",
                "step": step,
                "message": f"{step} exceeded its execution limit",
                "error_code": "tool_timeout",
                "recoverable": False,
                "artifacts": {},
                "summary": {},
            }
        try:
            payload = receive_connection.recv()
        except EOFError:
            payload = (
                "error",
                "ToolWorkerExit",
                f"{step} worker exited without returning a result",
                "",
            )
    finally:
        receive_connection.close()
        process.join(timeout=1.0)
        if process.is_alive():
            process.terminate()
            process.join()

    if payload[0] == "ok":
        return payload[1]
    return {
        "status": "error",
        "step": step,
        "message": f"{payload[1]}: {payload[2]}",
        "error_code": "tool_worker_error",
        "recoverable": False,
        "artifacts": {},
        "summary": {},
        "details": {"traceback": payload[3]},
    }


def _compact_model_value(value: Any) -> tuple[Any, bool]:
    """Remove dense list payloads from Agent context while preserving artifacts."""

    if isinstance(value, dict):
        compacted: dict[Any, Any] = {}
        changed = False
        for key, item in value.items():
            if key == "artifacts":
                compacted[key] = item
                continue
            compacted_item, item_changed = _compact_model_value(item)
            compacted[key] = compacted_item
            changed = changed or item_changed
        return compacted, changed
    if isinstance(value, (list, tuple)):
        if len(value) > MODEL_LIST_ITEM_LIMIT:
            return {
                "omitted_from_model_context": True,
                "item_count": len(value),
            }, True
        compacted_items = []
        changed = False
        for item in value:
            compacted_item, item_changed = _compact_model_value(item)
            compacted_items.append(compacted_item)
            changed = changed or item_changed
        return compacted_items, changed
    return value, False


def compact_model_result(value: Any) -> Any:
    """Return a model-facing copy without changing direct Python results."""

    compacted, changed = _compact_model_value(value)
    if changed and isinstance(compacted, dict):
        compacted["model_context_compacted"] = True
    return compacted


def _map_path(
    value: Any,
    root: Path,
    denied_roots: Sequence[Path] = (),
) -> Any:
    root = root.resolve()
    if isinstance(value, str):
        if value == "":
            return value
        path = Path(value).expanduser()
        resolved = path.resolve()
        if not (path.is_absolute() and (resolved == root or root in resolved.parents)):
            resolved = (root / str(path).lstrip("/")).resolve()
        if resolved != root and root not in resolved.parents:
            raise WorkspacePathError(f"Path must stay inside {root}: {value}")
        resolved_denied_roots = tuple(path.resolve() for path in denied_roots)
        if any(
            resolved == denied_root or denied_root in resolved.parents
            for denied_root in resolved_denied_roots
        ):
            raise WorkspacePathError(f"Path is not available to this Agent: {value}")
        return str(resolved)
    if isinstance(value, list):
        return [_map_path(item, root, denied_roots) for item in value]
    if isinstance(value, tuple):
        return tuple(_map_path(item, root, denied_roots) for item in value)
    return value


def scope_callable(
    function: Callable[..., Any],
    root: Path = ROOT,
    output_root: str = DEFAULT_OUTPUT_ROOT,
    artifact_policy: ArtifactPolicy | None = None,
    timeout_seconds: float | None = None,
    denied_roots: Sequence[Path] = (),
) -> Callable[..., Any]:
    """Wrap one callable so file inputs and outputs remain in its workspace."""
    signature = inspect.signature(function)
    output_parameter = signature.parameters.get("output_dir")
    default_output_dir = (
        output_parameter.default
        if output_parameter is not None
        else inspect.Parameter.empty
    )
    policy = artifact_policy or ArtifactPolicy.analysis(output_root, workspace=root)
    allowed_output_root = Path(_map_path(policy.output_root, root, denied_roots)).resolve()
    step = function.__name__.replace("_", "-")

    @wraps(function)
    def call(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        try:
            for name, value in bound.arguments.items():
                if name == "output_dir":
                    value = policy.resolve_output_dir(
                        str(value),
                        str(default_output_dir),
                        bound.arguments,
                    )
                    mapped_output = Path(_map_path(value, root, denied_roots)).resolve()
                    if (
                        mapped_output != allowed_output_root
                        and allowed_output_root not in mapped_output.parents
                    ):
                        raise WorkspacePathError(
                            f"Output directory must stay inside {policy.output_root}: {value}"
                        )
                    bound.arguments[name] = str(mapped_output)
                elif name.endswith(("_path", "_paths", "_dir")):
                    bound.arguments[name] = _map_path(value, root, denied_roots)
        except (ArtifactPolicyError, WorkspacePathError) as exc:
            return {
                "status": "error",
                "step": step,
                "message": str(exc),
                "error_code": "artifact_policy_violation",
                "recoverable": True,
                "details": {"allowed_roots": policy.allowed_roots(bound.arguments)},
                "artifacts": {},
                "summary": {},
            }
        if timeout_seconds is None:
            with scoped_workspace(root):
                result = function(*bound.args, **bound.kwargs)
        else:
            result = _invoke_with_process_timeout(
                function,
                bound.args,
                bound.kwargs,
                root,
                timeout_seconds,
                step,
            )
        compacted = compact_model_result(result)
        return compacted

    return call


def build_scoped_tools(
    root: Path = ROOT,
    output_root: str = DEFAULT_OUTPUT_ROOT,
    artifact_policy: ArtifactPolicy | None = None,
    timeout_seconds: float | None = None,
    denied_roots: Sequence[Path] = (),
) -> list[Any]:
    """Return registered tools whose callables enforce one artifact policy."""
    return [
        tool.model_copy(
            update={
                "func": scope_callable(
                    tool.func,
                    root,
                    output_root,
                    artifact_policy=artifact_policy,
                    timeout_seconds=timeout_seconds,
                    denied_roots=denied_roots,
                )
            }
        )
        for tool in get_tools()
    ]


__all__ = [
    "MODEL_LIST_ITEM_LIMIT",
    "build_scoped_tools",
    "compact_model_result",
    "scope_callable",
]

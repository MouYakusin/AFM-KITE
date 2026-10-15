"""Dependency-light policies for routing generated analysis artifacts.

AFM-KITE Agent Runs keep the generic ``outputs/data/<stage>`` layout.
Experiment runs classify every default tool output by its contributing input IDs.
This classification separates per-input and cross-input results in audit artifacts.
"""

from __future__ import annotations

import os
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping, Sequence

from ..serialization import flatten_string_leaves
from .identity import file_content_identity

ARTIFACT_POLICY_VERSION = "input-aware-v2"


class ArtifactPolicyError(ValueError):
    """A recoverable output-routing or artifact-ownership violation."""


class WorkspacePathError(ValueError):
    """A recoverable attempt to use a path outside the active workspace."""


def _normalise_path(value: str) -> str:
    """Return one slash-normalised absolute-looking path for comparisons."""
    return PurePosixPath("/" + value.replace("\\", "/").lstrip("/")).as_posix()


def _is_within(path: str, root: str) -> bool:
    path_value = PurePosixPath(_normalise_path(path))
    root_value = PurePosixPath(_normalise_path(root))
    return path_value == root_value or root_value in path_value.parents


@dataclass(frozen=True)
class ArtifactInput:
    """Stable experiment input identity and its permitted artifact subtree."""

    input_id: str
    artifact_dir: str
    identifiers: tuple[str, ...]


@dataclass(frozen=True)
class ArtifactPolicy:
    """Resolve and validate output directories for one agent runtime."""

    output_root: str
    inputs: tuple[ArtifactInput, ...] = ()
    cross_input_root: str = ""
    require_classification: bool = False
    workspace_root: str = ""

    @classmethod
    def analysis(
        cls,
        output_root: str = "/outputs",
        workspace: Path | None = None,
    ) -> "ArtifactPolicy":
        return cls(
            output_root=output_root,
            workspace_root=str(workspace.resolve()) if workspace is not None else "",
        )

    @classmethod
    def experiment(
        cls,
        workspace: Path,
        inputs: Sequence[Mapping[str, str]],
        output_root: str = "/outputs",
    ) -> "ArtifactPolicy":
        workspace = workspace.resolve()
        policy_inputs = []
        for input_info in inputs:
            workspace_path = input_info["workspace_path"]
            artifact_dir = input_info["artifact_dir"]
            identifiers = {
                input_info["input_id"],
                input_info.get("source_file", ""),
                workspace_path,
                workspace_path.lstrip("/"),
                str(workspace / workspace_path.lstrip("/")),
                artifact_dir,
                artifact_dir.lstrip("/"),
                str(workspace / artifact_dir.lstrip("/")),
            }
            policy_inputs.append(
                ArtifactInput(
                    input_id=input_info["input_id"],
                    artifact_dir=artifact_dir,
                    identifiers=tuple(
                        sorted(identifier for identifier in identifiers if identifier)
                    ),
                )
            )
        return cls(
            output_root=output_root,
            inputs=tuple(policy_inputs),
            cross_input_root=output_root.rstrip("/") + "/cross_input",
            require_classification=True,
            workspace_root=str(workspace),
        )

    def resolve_output_dir(
        self,
        requested: str,
        default: str,
        arguments: Mapping[str, Any],
    ) -> str:
        """Resolve a tool output directory and enforce experiment ownership."""
        scope, input_id = self._infer_scope(arguments)
        if requested == default:
            relative_stage = self._default_relative_stage(default)
            if not self.require_classification:
                return self._join(self.output_root, relative_stage)
            if scope == "input" and input_id is not None:
                return self._join(self._input(input_id).artifact_dir, relative_stage)
            if scope == "cross":
                return self._join(self.cross_input_root, relative_stage)
            raise ArtifactPolicyError(
                "Cannot classify the default tool output by input ID; pass an explicit "
                "output_dir under a declared by_input or cross_input artifact root."
            )

        if not self._is_within_root(requested, self.output_root):
            raise ArtifactPolicyError(
                f"Output directory must stay inside {self.output_root}: {requested}"
            )
        if not self.require_classification:
            return requested

        if scope == "input" and input_id is not None:
            expected = self._input(input_id).artifact_dir
            if not self._is_within_root(requested, expected):
                raise ArtifactPolicyError(
                    f"Output for {input_id} must stay inside {expected}: {requested}"
                )
            return requested
        if scope == "cross":
            if not self._is_within_root(requested, self.cross_input_root):
                raise ArtifactPolicyError(
                    "Cross-input output must stay inside {}: {}".format(
                        self.cross_input_root, requested
                    )
                )
            return requested

        classified_roots = [item.artifact_dir for item in self.inputs]
        classified_roots.append(self.cross_input_root)
        if not any(self._is_within_root(requested, root) for root in classified_roots):
            raise ArtifactPolicyError(
                "Experiment output must stay inside a declared by_input or "
                f"cross_input artifact root: {requested}"
            )
        return requested

    def allowed_roots(self, arguments: Mapping[str, Any]) -> list[str]:
        """Return virtual output roots that can recover the current tool call."""
        if not self.require_classification:
            return [self.output_root]
        scope, input_id = self._infer_scope(arguments)
        if scope == "input" and input_id is not None:
            return [self._input(input_id).artifact_dir]
        if scope == "cross":
            return [self.cross_input_root]
        return [item.artifact_dir for item in self.inputs] + [self.cross_input_root]

    def _infer_scope(self, arguments: Mapping[str, Any]) -> tuple[str, str | None]:
        values = []
        for name, value in arguments.items():
            if name != "output_dir":
                values.extend(flatten_string_leaves(value))
        values = [value for value in values if value]

        cross_paths = self._root_paths(self.cross_input_root)
        if any(
            _is_within(value, candidate)
            for value in values
            for candidate in cross_paths
            if candidate
        ):
            return "cross", None

        matched_ids = {
            item.input_id
            for item in self.inputs
            if any(
                _is_within(value, identifier) or _is_within(identifier, value)
                for value in values
                for identifier in item.identifiers
            )
        }
        if len(matched_ids) == 1:
            return "input", next(iter(matched_ids))
        if len(matched_ids) > 1:
            return "cross", None
        return "unknown", None

    def _input(self, input_id: str) -> ArtifactInput:
        for item in self.inputs:
            if item.input_id == input_id:
                return item
        raise KeyError(input_id)

    def _default_relative_stage(self, default: str) -> str:
        parts = list(PurePosixPath(default.replace("\\", "/")).parts)
        if parts and parts[0] == "/":
            parts = parts[1:]
        output_name = PurePosixPath(self.output_root).name
        if parts and parts[0] in {"outputs", output_name}:
            parts = parts[1:]
        if self.require_classification and parts and parts[0] == "data":
            parts = parts[1:]
        return PurePosixPath(*parts).as_posix() if parts else ""

    def _root_paths(self, root: str) -> tuple[str, ...]:
        if not root:
            return ()
        paths = {root, root.lstrip("/")}
        if self.workspace_root:
            paths.add(str(Path(self.workspace_root) / root.lstrip("/")))
        return tuple(sorted(paths))

    def _is_within_root(self, path: str, root: str) -> bool:
        return any(
            _is_within(path, candidate) for candidate in self._root_paths(root)
        )

    @staticmethod
    def _join(root: str, relative: str) -> str:
        root_path = PurePosixPath(_normalise_path(root))
        return (root_path / relative).as_posix() if relative else root_path.as_posix()


def _file_state(path: Path, *, content_hash: bool = True) -> str:
    if path.is_symlink():
        return "symlink:" + os.readlink(path)
    if not content_hash:
        metadata = path.stat()
        return "file-metadata:{}:{}:{}".format(
            metadata.st_size,
            metadata.st_mtime_ns,
            stat.S_IMODE(metadata.st_mode),
        )
    size_bytes, content_digest = file_content_identity(path)
    return f"file:{size_bytes}:{content_digest}"


def snapshot_workspace(
    workspace: Path,
    *,
    metadata_only_roots: Sequence[str] = (),
) -> dict[str, str]:
    """Record workspace state for post-Agent mutation auditing.

    Files below ``metadata_only_roots`` use size, modification time, and mode
    rather than content hashes. Experiment Skill resources use this path so
    their content is neither hashed nor recorded as a digest.
    """
    workspace = workspace.resolve()
    unhashed_roots = tuple(
        root.strip("/") for root in metadata_only_roots if root.strip("/")
    )
    result: dict[str, str] = {}
    for path in sorted(workspace.rglob("*")):
        if path.is_file() or path.is_symlink():
            relative = path.relative_to(workspace).as_posix()
            hash_content = not any(
                relative == root or relative.startswith(root + "/")
                for root in unhashed_roots
            )
            result[relative] = _file_state(path, content_hash=hash_content)
    return result


def audit_workspace_changes(
    before: Mapping[str, str],
    after: Mapping[str, str],
    allowed_roots: Sequence[str] = ("outputs", "scratch"),
) -> dict[str, Any]:
    """Report Agent-created, modified, or deleted files outside writable roots."""
    normalised_roots = tuple(root.strip("/") for root in allowed_roots if root.strip("/"))
    violations = []
    for path in sorted(set(before) | set(after)):
        if before.get(path) == after.get(path):
            continue
        if any(path == root or path.startswith(root + "/") for root in normalised_roots):
            continue
        change = "modified"
        if path not in before:
            change = "created"
        elif path not in after:
            change = "deleted"
        violations.append({"path": path, "change": change})
    return {
        "compliant": not violations,
        "allowed_roots": ["/" + root for root in normalised_roots],
        "violations": violations,
    }


def _is_under_relative_root(path: Path, workspace: Path, roots: Sequence[str]) -> bool:
    relative = path.relative_to(workspace).as_posix()
    return any(relative == root or relative.startswith(root + "/") for root in roots)


@contextmanager
def agent_workspace_guard(
    workspace: Path,
    writable_roots: Sequence[str] = ("outputs", "scratch"),
) -> Iterator[None]:
    """Make all prepared experiment resources except approved roots read-only.

    This boundary protects the built-in filesystem and execution capabilities of Deep Agents.
    Post-run metadata comparison remains authoritative.
    A process owned by the same user may change its own permissions.
    """
    workspace = workspace.resolve()
    roots = tuple(root.strip("/") for root in writable_roots if root.strip("/"))
    if os.name != "posix":
        yield
        return

    original_modes: dict[Path, int] = {}
    prepared_paths = [workspace] + sorted(workspace.rglob("*"), key=lambda item: len(item.parts))
    try:
        for path in prepared_paths:
            if path.is_symlink():
                continue
            original_modes[path] = stat.S_IMODE(path.stat().st_mode)
            if path == workspace:
                path.chmod(0o555)
            elif _is_under_relative_root(path, workspace, roots):
                path.chmod(0o755 if path.is_dir() else 0o644)
            else:
                path.chmod(0o555 if path.is_dir() else 0o444)
        yield
    finally:
        for path in sorted(original_modes, key=lambda item: len(item.parts), reverse=True):
            if path.exists() and not path.is_symlink():
                path.chmod(original_modes[path])

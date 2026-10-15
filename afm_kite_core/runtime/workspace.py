"""Isolated AFM-KITE Agent Run paths, resource snapshots, and permissions."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence
from uuid import uuid4

from ..artifacts import agent_workspace_guard
from ..artifacts.identity import file_content_identity

DEFAULT_OUTPUT_ROOT = "/outputs"
RUN_PREFIX = "runs"
StateObserver = Callable[[Any], None]


@dataclass(frozen=True)
class RunLayout:
    """Host paths and Agent-visible paths owned by one analysis invocation."""

    project_root: Path
    run_id: str
    root: Path

    @property
    def agent_root(self) -> Path:
        return self.root

    @property
    def inputs(self) -> Path:
        return self.root / "inputs"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    @property
    def scratch(self) -> Path:
        return self.root / "scratch"

    @property
    def request_path(self) -> Path:
        return self.inputs / "request.md"

    @property
    def input_manifest_path(self) -> Path:
        return self.inputs / "manifest.json"

    @property
    def answer_path(self) -> Path:
        return self.root / "answer.md"

    @property
    def run_record_path(self) -> Path:
        return self.root / "run.json"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def tool_trace_path(self) -> Path:
        return self.root / "tool_trace.json"

    @property
    def model_trace_path(self) -> Path:
        return self.root / "model_trace.json"

    @property
    def host_virtual_root(self) -> str:
        return self.project_virtual_path(self.root)

    @property
    def virtual_output_root(self) -> str:
        return "/outputs"

    @property
    def virtual_scratch_root(self) -> str:
        return "/scratch"

    @property
    def writable_roots(self) -> tuple[str, str]:
        return ("outputs", "scratch")

    def agent_virtual_path(self, path: Path) -> str:
        relative = path.resolve().relative_to(self.agent_root)
        return "/" + relative.as_posix()

    def project_virtual_path(self, path: Path) -> str:
        relative = path.resolve().relative_to(self.project_root)
        return "/" + relative.as_posix()

    def relative_path(self, path: Path) -> str:
        return path.resolve().relative_to(self.root).as_posix()


def _resolve_project_output_root(project_root: Path, value: str) -> Path:
    project_root = project_root.resolve()
    if not value or not value.strip() or value.strip() in {"/", "."}:
        raise ValueError("output_root must name a directory inside the project")
    supplied = Path(value).expanduser()
    supplied_resolved = supplied.resolve()
    if supplied.is_absolute() and (
        supplied_resolved == project_root or project_root in supplied_resolved.parents
    ):
        resolved = supplied_resolved
    else:
        resolved = (project_root / str(supplied).lstrip("/")).resolve()
    if resolved == project_root or project_root not in resolved.parents:
        raise ValueError(f"output_root must stay inside {project_root}: {value}")
    return resolved


def _new_run_id() -> str:
    return f"run_{uuid4().hex[:16]}"


def create_run_layout(
    project_root: Path,
    output_root: str = DEFAULT_OUTPUT_ROOT,
    run_id: str | None = None,
) -> RunLayout:
    """Create one collision-safe ``runs/<run_id>`` Agent sandbox."""
    explicit_run_id = run_id is not None
    if run_id is not None and re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id
    ) is None:
        raise ValueError("run_id may contain only letters, digits, '.', '_', and '-'")

    project_root = project_root.resolve()
    if not project_root.is_dir():
        raise ValueError(f"project_root must be an existing directory: {project_root}")
    output_path = _resolve_project_output_root(project_root, output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    while True:
        selected_run_id = run_id or _new_run_id()
        root = output_path / RUN_PREFIX / selected_run_id
        try:
            root.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            if explicit_run_id:
                raise

    for name in ("inputs", "outputs", "scratch"):
        (root / name).mkdir()
    return RunLayout(
        project_root=project_root,
        run_id=selected_run_id,
        root=root.resolve(),
    )


def resolve_input_paths(
    project_root: Path,
    input_paths: Sequence[Path],
) -> tuple[Path, ...]:
    """Validate explicit regular-file inputs without following project symlinks."""
    if isinstance(input_paths, (str, bytes, Path)):
        raise TypeError("input_paths must be a sequence of project files")
    if not input_paths:
        raise ValueError("input_paths must contain at least one project file")

    project_root = project_root.resolve()
    resolved_inputs: list[Path] = []
    seen: set[Path] = set()
    for supplied in input_paths:
        path = Path(supplied).expanduser()
        candidate = path if path.is_absolute() else project_root / path
        lexical = candidate.absolute()
        resolved = lexical.resolve()
        try:
            resolved.relative_to(project_root)
        except ValueError as exc:
            raise ValueError(
                f"input path must stay inside {project_root}: {supplied}"
            ) from exc

        current = lexical
        while True:
            if current.is_symlink():
                raise ValueError(f"input path may not contain symlinks: {supplied}")
            if current.resolve() == project_root:
                break
            parent = current.parent
            if parent == current:
                raise ValueError(f"input path must stay inside {project_root}: {supplied}")
            current = parent
        if not resolved.is_file():
            raise ValueError(f"input path must be a regular file: {supplied}")
        if resolved in seen:
            raise ValueError(f"input path was supplied more than once: {supplied}")
        seen.add(resolved)
        resolved_inputs.append(resolved)
    return tuple(resolved_inputs)


def validate_skill_source(skill_source: Path | None) -> Path | None:
    """Validate a Skill tree before copying it into a Run sandbox."""
    if skill_source is None:
        return None
    supplied = skill_source.expanduser()
    if supplied.is_symlink():
        raise ValueError(f"skill_source may not contain symlinks: {skill_source}")
    source = supplied.resolve()
    if not source.is_dir():
        raise ValueError(f"skill_source must be a directory: {skill_source}")
    for path in (source, *source.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"skill_source may not contain symlinks: {path}")
    return source


def snapshot_run_resources(
    layout: RunLayout,
    input_paths: Sequence[Path],
    skill_source: Path | None,
) -> list[dict[str, Any]]:
    """Copy all model-visible inputs and optional Skill into one Run sandbox."""
    records: list[dict[str, Any]] = []
    for index, source in enumerate(input_paths, start=1):
        input_id = f"input-{index:02d}"
        destination = layout.inputs / input_id / source.name
        destination.parent.mkdir()
        shutil.copy2(source, destination)
        size_bytes, digest = file_content_identity(destination)
        records.append(
            {
                "input_id": input_id,
                "source_file": source.relative_to(layout.project_root).as_posix(),
                "workspace_path": layout.agent_virtual_path(destination),
                "size_bytes": size_bytes,
                "blake2b": digest,
            }
        )
    if skill_source is not None:
        shutil.copytree(skill_source, layout.root / skill_source.name)
    return records


def invoke_agent_with_observer(
    agent: Any,
    prompt: str,
    recursion_limit: int,
    state_observer: StateObserver | None = None,
    callbacks: Sequence[Any] = (),
) -> Any:
    """Run an Agent and expose each state for failure-time audit capture."""
    payload = {"messages": [{"role": "user", "content": prompt}]}
    config: dict[str, Any] = {"recursion_limit": recursion_limit}
    if callbacks:
        config["callbacks"] = list(callbacks)
    stream = getattr(agent, "stream", None)
    if not callable(stream):
        result = agent.invoke(payload, config=config)
        if state_observer is not None:
            state_observer(result)
        return result

    latest_state: Any = {}
    for state in stream(payload, config=config, stream_mode="values"):
        latest_state = state
        if state_observer is not None:
            state_observer(state)
    return latest_state


def invoke_agent(
    agent: Any,
    prompt: str,
    recursion_limit: int,
    layout: RunLayout,
    state_observer: StateObserver | None = None,
    callbacks: Sequence[Any] = (),
) -> Any:
    """Run an Agent while guarding only its private Run sandbox."""
    with agent_workspace_guard(layout.agent_root, writable_roots=layout.writable_roots):
        return invoke_agent_with_observer(
            agent,
            prompt,
            recursion_limit,
            state_observer=state_observer,
            callbacks=callbacks,
        )

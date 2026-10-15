"""Experiment case discovery, immutable batch resources, and workspaces."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from afm_kite_core.artifacts.identity import file_content_identity
from afm_kite_core.identity import LEGACY_SKILL_DIR_NAME, SKILL_DIR_NAME

from .artifacts import RUN_OUTPUT_ROOT, RUN_SCRATCH_ROOT, write_json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA = PROJECT_ROOT / "data"
EXPERIMENT_OUTPUTS = PROJECT_ROOT / "experiments_outputs"
BATCH_SNAPSHOT_SCHEMA = "afm.experiment-batch-snapshot/v2"
BATCH_SNAPSHOT_DIR = "batch_snapshot"
PEROVSKITE_FILENAME_PATTERN = re.compile(
    r"^(?P<group>perovskite[1-9]\d*)-(?P<item>[1-9]\d*)$",
    flags=re.IGNORECASE,
)

DATA_DIRS = {
    "grating": DATA / "grating",
    "hs100_mg": DATA / "HS100-MG",
    "perovskite": DATA / "perovskite",
    "wafer": DATA / "wafer",
    "ag_tio2": DATA / "Ag-TiO2",
}


@dataclass(frozen=True)
class Case:
    case_id: str
    group: str
    inputs: tuple[str, ...]
    task: str
    required_metrics: tuple[str, ...]


def rel(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def natural_sort_key(value: str) -> tuple[tuple[int, Any], ...]:
    """Sort embedded numbers numerically, so item 10 follows item 9."""
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.lower())
        for part in re.split(r"(\d+)", value)
    )


def discover_files(folder: Path) -> list[Path]:
    files = sorted(folder.glob("*.txt"), key=lambda path: natural_sort_key(path.name))
    if not files:
        raise FileNotFoundError(f"No .txt files in {folder}")
    return files


def _case_prompt(group: str, prompts: Mapping[str, Any]) -> tuple[str, tuple[str, ...]]:
    prompt = prompts["groups"][group]
    return prompt["task"], tuple(prompt["required_metrics"])


def _discover_perovskite_cases(
    group: str,
    task: str,
    required_metrics: tuple[str, ...],
) -> list[Case]:
    grouped_paths: dict[str, dict[int, Path]] = {}
    for path in discover_files(DATA_DIRS[group]):
        match = PEROVSKITE_FILENAME_PATTERN.fullmatch(path.stem)
        if match is None:
            raise ValueError(
                "Perovskite filename must follow "
                f"perovskite<group>-<item>.txt: {path.name}"
            )
        group_name = match.group("group").lower()
        item_number = int(match.group("item"))
        members = grouped_paths.setdefault(group_name, {})
        if item_number in members:
            raise ValueError(
                f"Perovskite group {group_name} has duplicate item {item_number}"
            )
        members[item_number] = path

    cases = []
    for group_name in sorted(grouped_paths, key=natural_sort_key):
        members = grouped_paths[group_name]
        item_numbers = sorted(members)
        expected_items = list(range(1, len(item_numbers) + 1))
        if item_numbers != expected_items:
            raise ValueError(
                f"Perovskite group {group_name} items must be consecutively numbered "
                f"from 1; found {item_numbers}"
            )
        paths = [members[item_number] for item_number in item_numbers]
        if len(paths) < 3:
            raise ValueError(
                f"Perovskite group {group_name} needs at least three files for correlation"
            )
        group_label = group_name
        if group_name.lower().startswith("perovskite"):
            group_label = group_name[len("perovskite"):].lstrip("_-") or group_name
        cases.append(
            Case(
                case_id=f"perovskite-group-{group_label}",
                group=group,
                inputs=tuple(rel(path) for path in paths),
                task=task,
                required_metrics=required_metrics,
            )
        )
    return cases


def _discover_ag_tio2_cases(
    group: str,
    task: str,
    required_metrics: tuple[str, ...],
) -> list[Case]:
    grouped_paths: dict[str, dict[str, Path]] = {}
    pattern = re.compile(r"^Ag-TiO2-(.+)-(before|after)$", flags=re.IGNORECASE)
    for path in discover_files(DATA_DIRS[group]):
        match = pattern.fullmatch(path.stem)
        if match is None:
            raise ValueError(
                "Ag-TiO2 filename must follow Ag-TiO2-<case>-<before|after>.txt: "
                f"{rel(path)}"
            )
        pair_name, condition = match.group(1), match.group(2).lower()
        grouped_paths.setdefault(pair_name, {})[condition] = path

    cases = []
    for pair_name in sorted(grouped_paths, key=natural_sort_key):
        pair = grouped_paths[pair_name]
        missing = [condition for condition in ("before", "after") if condition not in pair]
        if missing:
            raise FileNotFoundError(
                f"Incomplete Ag-TiO2 pair {pair_name}: missing {', '.join(missing)}"
            )
        cases.append(
            Case(
                case_id=f"ag_tio2-pair-{pair_name}",
                group=group,
                inputs=(rel(pair["before"]), rel(pair["after"])),
                task=task,
                required_metrics=required_metrics,
            )
        )
    return cases


def discover_cases(group: str, prompts: Mapping[str, Any]) -> list[Case]:
    """Discover every case for one experiment without selecting a subset."""
    task, required_metrics = _case_prompt(group, prompts)
    if group == "perovskite":
        return _discover_perovskite_cases(group, task, required_metrics)
    if group == "ag_tio2":
        return _discover_ag_tio2_cases(group, task, required_metrics)
    return [
        Case(
            case_id=f"{group}-{path.stem.replace(' ', '_')}",
            group=group,
            inputs=(rel(path),),
            task=task,
            required_metrics=required_metrics,
        )
        for path in discover_files(DATA_DIRS[group])
    ]


def normalise_case_selection(value: Any) -> int | str:
    """Return a positive 1-based case number or the string ``all``."""
    if isinstance(value, str):
        value = value.strip()
        if value.lower() == "all":
            return "all"
        try:
            value = int(value)
        except ValueError as exc:
            raise ValueError("Case selection must be a positive integer or 'all'") from exc
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("Case selection must be a positive integer or 'all'")
    return value


def select_cases(cases: Sequence[Case], selection: Any) -> list[Case]:
    """Select all cases, a 1-based position, or one exact case identifier."""
    if isinstance(selection, str) and selection.strip().lower() != "all":
        case_id = selection.strip()
        matches = [case for case in cases if case.case_id == case_id]
        if matches:
            return matches
        if not case_id.isdigit():
            available = ", ".join(case.case_id for case in cases)
            raise ValueError(
                f"Case ID {case_id!r} is unavailable; available case IDs: {available}"
            )
    normalised = normalise_case_selection(selection)
    if normalised == "all":
        return list(cases)
    if normalised > len(cases):
        raise ValueError(f"Case {normalised} is unavailable; discovered {len(cases)} case(s)")
    return [cases[normalised - 1]]


def _input_content_digest(files: Sequence[Mapping[str, Any]]) -> str:
    """Return one digest over input records only.

    Skill resources are intentionally excluded: experiment snapshots preserve
    their file inventory and sizes without generating or recording Skill content
    hashes.
    """
    canonical = json.dumps(
        [dict(entry) for entry in files if entry.get("category") == "input"],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.blake2b(canonical, digest_size=32).hexdigest()


def _resolve_skill_root(resource_root: Path) -> Path:
    """Resolve the canonical Skill directory or a historical snapshot path."""
    for directory_name in (SKILL_DIR_NAME, LEGACY_SKILL_DIR_NAME):
        candidate = resource_root / directory_name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(resource_root / SKILL_DIR_NAME)


def create_batch_snapshot(
    cases: Sequence[Case],
    output_dir: Path,
    include_skill: bool,
) -> tuple[Path, dict[str, Any]]:
    """Freeze all selected inputs and the optional Skill once per experiment batch."""
    snapshot_root = output_dir / BATCH_SNAPSHOT_DIR
    snapshot_root.mkdir(parents=True, exist_ok=False)
    files: list[dict[str, Any]] = []

    def copy_input_file(source: Path, relative_path: Path) -> None:
        destination = snapshot_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        size_bytes, digest = file_content_identity(destination)
        files.append(
            {
                "path": relative_path.as_posix(),
                "category": "input",
                "size_bytes": size_bytes,
                "blake2b": digest,
            }
        )

    def copy_skill_file(source: Path, relative_path: Path) -> None:
        destination = snapshot_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        files.append(
            {
                "path": relative_path.as_posix(),
                "category": "skill",
                "size_bytes": destination.stat().st_size,
            }
        )

    input_paths = sorted(
        {relative_path for case in cases for relative_path in case.inputs},
        key=natural_sort_key,
    )
    for relative_path in input_paths:
        source = (PROJECT_ROOT / relative_path).resolve()
        if source != DATA and DATA not in source.parents:
            raise ValueError(f"Experiment input must stay inside {DATA}: {relative_path}")
        if not source.is_file():
            raise FileNotFoundError(source)
        copy_input_file(source, Path(relative_path))

    if include_skill:
        skill_root = _resolve_skill_root(PROJECT_ROOT)
        for source in sorted(skill_root.rglob("*")):
            if source.is_symlink():
                raise ValueError(f"Experiment Skill snapshot does not accept symlinks: {source}")
            if source.is_file():
                relative_path = Path(SKILL_DIR_NAME) / source.relative_to(skill_root)
                copy_skill_file(source, relative_path)

    files.sort(key=lambda entry: str(entry["path"]))
    record = {
        "schema": BATCH_SNAPSHOT_SCHEMA,
        "root": BATCH_SNAPSHOT_DIR,
        "manifest": f"{BATCH_SNAPSHOT_DIR}/manifest.json",
        "input_content_digest": _input_content_digest(files),
        "files": files,
    }
    write_json(snapshot_root / "manifest.json", record)
    return snapshot_root, record


def verify_batch_snapshot(snapshot_root: Path, record: Mapping[str, Any]) -> None:
    """Fail before a run when its frozen batch resources have changed."""
    snapshot_root = snapshot_root.resolve()
    expected_files = list(record.get("files", []))
    actual_paths = {
        path.relative_to(snapshot_root).as_posix()
        for path in snapshot_root.rglob("*")
        if path.is_file() and path != snapshot_root / "manifest.json"
    }
    declared_paths = {str(entry["path"]) for entry in expected_files}
    if actual_paths != declared_paths:
        raise RuntimeError("Experiment batch snapshot file inventory changed")

    verified_input_files = []
    for entry in expected_files:
        relative_path = str(entry["path"])
        path = (snapshot_root / relative_path).resolve()
        if snapshot_root not in path.parents:
            raise RuntimeError(f"Invalid batch snapshot path: {relative_path}")
        category = str(entry.get("category", ""))
        if category == "skill":
            if "blake2b" in entry:
                raise RuntimeError(
                    f"Experiment Skill snapshot must not record a hash: {relative_path}"
                )
            if path.stat().st_size != int(entry["size_bytes"]):
                raise RuntimeError(f"Experiment batch snapshot changed: {relative_path}")
            continue
        if category != "input":
            raise RuntimeError(
                f"Invalid batch snapshot category for {relative_path}: {category}"
            )
        size_bytes, digest = file_content_identity(path)
        if size_bytes != int(entry["size_bytes"]) or digest != str(entry["blake2b"]):
            raise RuntimeError(f"Experiment batch snapshot changed: {relative_path}")
        verified_input_files.append(dict(entry))
    if _input_content_digest(verified_input_files) != str(
        record.get("input_content_digest", "")
    ):
        raise RuntimeError("Experiment batch input digest changed")


def prepare_workspace(
    case: Case,
    workspace: Path,
    skill_bundle: bool,
    resource_root: Path = PROJECT_ROOT,
) -> list[dict[str, str]]:
    """Create an isolated run workspace and copy in only the required resources."""
    resource_root = resource_root.resolve()
    data_root = resource_root / "data"
    outputs_root = workspace / RUN_OUTPUT_ROOT.lstrip("/")
    (outputs_root / "by_input").mkdir(parents=True)
    (outputs_root / "cross_input").mkdir(parents=True)
    (workspace / RUN_SCRATCH_ROOT.lstrip("/")).mkdir(parents=True)
    inputs = []
    for index, relative_path in enumerate(case.inputs, start=1):
        source = (resource_root / relative_path).resolve()
        if source != data_root and data_root not in source.parents:
            raise ValueError(
                f"Experiment input must stay inside {data_root}: {relative_path}"
            )
        destination = workspace / "inputs" / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        input_id = f"input-{index:02d}"
        artifact_dir = f"{RUN_OUTPUT_ROOT}/by_input/{input_id}"
        (workspace / artifact_dir.lstrip("/")).mkdir(parents=True, exist_ok=True)
        inputs.append(
            {
                "input_id": input_id,
                "source_file": relative_path,
                "workspace_path": f"/{destination.relative_to(workspace).as_posix()}",
                "artifact_dir": artifact_dir,
            }
        )
    if skill_bundle:
        skill_root = _resolve_skill_root(resource_root)
        shutil.copytree(
            skill_root,
            workspace / SKILL_DIR_NAME,
        )
    return inputs


def next_output_dir(group: str) -> Path:
    """Return the next ``<group>-NNN`` directory under experiments_outputs."""
    prefix = f"{group}-"
    existing_numbers = []
    if EXPERIMENT_OUTPUTS.is_dir():
        for path in EXPERIMENT_OUTPUTS.iterdir():
            suffix = path.name.removeprefix(prefix)
            if path.is_dir() and suffix.isdigit():
                existing_numbers.append(int(suffix))
    return EXPERIMENT_OUTPUTS / f"{group}-{max(existing_numbers, default=0) + 1:03d}"

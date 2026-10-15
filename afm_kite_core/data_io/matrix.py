"""AFM matrix, metadata, and physical-coordinate loading helpers."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from .channels import parse_afm_channels
from .paths import resolve_readable_file


DEFAULT_PHYSICAL_UNIT = "μm"
DEFAULT_PIXEL_SIZE = 1.0
DEFAULTS_METADATA_KEY = "_afm_analysis_suite_defaults"


def parse_afm_txt(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Parse an AFM text export into its first matrix and metadata block."""

    txt_path = resolve_readable_file(path)
    first = parse_afm_channels(txt_path)[0]
    matrix = np.asarray(first["matrix"], dtype=float)
    metadata = first.get("metadata", {})
    if first.get("metadata_source") == "implicit":
        metadata = {}
    if matrix.ndim != 2 or matrix.size == 0:
        raise ValueError(f"Parsed matrix is empty or not 2D: {txt_path}")
    return matrix, metadata


def load_height_data(path: str | Path) -> np.ndarray:
    """Load a 2D AFM matrix from TXT/TSV-like text or NPY."""

    resolved = resolve_readable_file(path)
    matrix = np.load(resolved) if resolved.suffix.lower() == ".npy" else parse_afm_txt(resolved)[0]
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"Height data must be 2D, got {matrix.shape} from {resolved}")
    return matrix


def model_info(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return the model/mode metadata block if present."""

    info = metadata.get("ModelInfo") or metadata.get("ModeInfo") or {}
    return info if isinstance(info, dict) else {}


def height_unit(metadata: dict[str, Any]) -> str:
    """Infer the matrix value unit from AFM metadata."""

    show_mode = str(model_info(metadata).get("ImageShowMode") or "")
    match = re.search(r"\((.*?)\)", show_mode)
    unit = match.group(1).strip() if match else ""
    if unit:
        return unit

    candidate_blocks = (
        model_info(metadata),
        metadata,
        metadata.get("InitialPosition") or {},
        metadata.get("Resolution") or {},
    )
    for block in candidate_blocks:
        if not isinstance(block, dict):
            continue
        for key in ("ZUnit", "ValueUnit", "Unit"):
            recorded = str(block.get(key) or "").strip()
            if recorded:
                return recorded
    return DEFAULT_PHYSICAL_UNIT


def metadata_with_defaults(
    metadata: dict[str, Any],
    matrix_shape: tuple[int, int],
) -> dict[str, Any]:
    """Return metadata with explicit micrometre defaults for missing dimensions."""

    rows, cols = int(matrix_shape[0]), int(matrix_shape[1])
    result = copy.deepcopy(metadata) if metadata else {}
    resolution = dict(result.get("Resolution") or {})
    defaulted_fields: list[str] = []

    defaults = {
        "XResolution": cols,
        "YResolution": rows,
        "XSize": float(cols) * DEFAULT_PIXEL_SIZE,
        "YSize": float(rows) * DEFAULT_PIXEL_SIZE,
        "XUnit": DEFAULT_PHYSICAL_UNIT,
        "YUnit": DEFAULT_PHYSICAL_UNIT,
    }
    for key, fallback in defaults.items():
        if resolution.get(key) in {None, ""}:
            resolution[key] = fallback
            defaulted_fields.append(f"Resolution.{key}")
    result["Resolution"] = resolution

    if not model_info(result).get("ImageShowMode"):
        info = result.get("ModelInfo")
        if not isinstance(info, dict):
            info = {}
            result["ModelInfo"] = info
        info["ImageShowMode"] = f"Height({height_unit(result)})"
        defaulted_fields.append("ModelInfo.ImageShowMode")

    inherited_defaults = list(result.get(DEFAULTS_METADATA_KEY) or [])
    result[DEFAULTS_METADATA_KEY] = sorted(
        set(inherited_defaults + defaulted_fields)
    )
    return result


def metadata_summary(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return compact metadata fields used by analysis reports."""

    resolution = metadata.get("Resolution", {})
    info = model_info(metadata)
    return {
        "height_unit": height_unit(metadata),
        "x_resolution": resolution.get("XResolution"),
        "y_resolution": resolution.get("YResolution"),
        "x_size": resolution.get("XSize"),
        "y_size": resolution.get("YSize"),
        "x_unit": resolution.get("XUnit") or DEFAULT_PHYSICAL_UNIT,
        "y_unit": resolution.get("YUnit") or DEFAULT_PHYSICAL_UNIT,
        "scan_mode": info.get("ScanMode"),
        "image_show_mode": info.get("ImageShowMode"),
    }


def physical_area_context_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Compute pixel size and area context from AFM metadata when available."""

    resolution = metadata.get("Resolution", {})
    defaulted_fields = set(metadata.get(DEFAULTS_METADATA_KEY) or [])
    try:
        x_resolution = float(resolution["XResolution"])
        y_resolution = float(resolution["YResolution"])
        x_size = float(resolution["XSize"])
        y_size = float(resolution["YSize"])
    except (KeyError, TypeError, ValueError):
        return {
            "pixel_width": DEFAULT_PIXEL_SIZE,
            "pixel_height": DEFAULT_PIXEL_SIZE,
            "pixel_area": DEFAULT_PIXEL_SIZE**2,
            "x_unit": DEFAULT_PHYSICAL_UNIT,
            "y_unit": DEFAULT_PHYSICAL_UNIT,
            "height_unit": height_unit(metadata),
            "units_defaulted": True,
            "pixel_size_defaulted": True,
        }
    pixel_width = x_size / x_resolution
    pixel_height = y_size / y_resolution
    return {
        "pixel_width": pixel_width,
        "pixel_height": pixel_height,
        "pixel_area": pixel_width * pixel_height,
        "x_unit": resolution.get("XUnit") or DEFAULT_PHYSICAL_UNIT,
        "y_unit": resolution.get("YUnit") or DEFAULT_PHYSICAL_UNIT,
        "height_unit": height_unit(metadata),
        "units_defaulted": bool(defaulted_fields),
        "pixel_size_defaulted": any(
            field
            in {
                "Resolution.XResolution",
                "Resolution.YResolution",
                "Resolution.XSize",
                "Resolution.YSize",
            }
            for field in defaulted_fields
        ),
    }


def physical_area_context(metadata_path: str) -> dict[str, Any]:
    """Load physical pixel context from a metadata JSON path."""

    if not metadata_path:
        return physical_area_context_from_metadata({})
    try:
        path = resolve_readable_file(metadata_path)
    except FileNotFoundError:
        return physical_area_context_from_metadata({})
    metadata = json.loads(path.read_text(encoding="utf-8"))
    return physical_area_context_from_metadata(metadata)


def metadata_sidecar_path(matrix_path: str | Path) -> Path:
    """Return the conventional metadata sidecar path for a matrix artifact."""

    path = Path(matrix_path)
    return path.with_name(f"{path.stem}_metadata.json")


def resolve_matrix_metadata_path(
    input_path: str | Path,
    metadata_path: str = "",
) -> Path | None:
    """Resolve explicit metadata or auto-discover the matrix sidecar."""

    if metadata_path:
        return resolve_readable_file(metadata_path)
    candidate = metadata_sidecar_path(resolve_readable_file(input_path))
    return candidate.resolve() if candidate.is_file() else None


def load_height_data_and_metadata(
    input_path: str | Path,
    metadata_path: str = "",
) -> tuple[np.ndarray, dict[str, Any], Path]:
    """Load matrix plus metadata from a matrix path and optional metadata JSON."""

    resolved = resolve_readable_file(input_path)
    metadata: dict[str, Any] = {}
    if resolved.suffix.lower() == ".npy":
        matrix = np.load(resolved)
    else:
        matrix, metadata = parse_afm_txt(resolved)
    metadata_file = resolve_matrix_metadata_path(resolved, metadata_path)
    if metadata_file is not None and (metadata_path or not metadata):
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.size == 0:
        raise ValueError(f"Height data must be a non-empty 2D matrix: {resolved}")
    metadata = metadata_with_defaults(metadata, matrix.shape)
    return matrix, metadata, resolved


__all__ = [
    "DEFAULT_PHYSICAL_UNIT",
    "DEFAULT_PIXEL_SIZE",
    "height_unit",
    "load_height_data",
    "load_height_data_and_metadata",
    "metadata_sidecar_path",
    "metadata_summary",
    "metadata_with_defaults",
    "model_info",
    "parse_afm_txt",
    "physical_area_context",
    "physical_area_context_from_metadata",
    "resolve_matrix_metadata_path",
]

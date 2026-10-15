"""AFM channel ingestion and manifest assembly workflow."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from afm_kite_core.algorithms.statistics import finite_matrix_summary
from afm_kite_core.contracts import DATA_MANIFEST_SCHEMA
from afm_kite_core.data_io.channels import parse_afm_channels
from afm_kite_core.data_io.matrix import (
    DEFAULT_PHYSICAL_UNIT,
    metadata_with_defaults,
)
from afm_kite_core.artifacts.writers import (
    atomic_write_path,
    safe_stem,
    write_json_file,
    write_text_file,
)
from afm_kite_core.data_io.paths import ensure_output_dir, resolve_readable_file


def _safe_name(text: str) -> str:
    return safe_stem(
        text,
        fallback="channel",
        preserve_separators=False,
        normalize_micro=True,
    )


def _write_channel_metadata(
    out_dir: Path,
    channel_stem: str,
    metadata: dict[str, Any],
    matrix_shape: tuple[int, int],
) -> Path:
    metadata_path = out_dir / f"{channel_stem}_metadata.json"
    write_json_file(metadata_path, metadata_with_defaults(metadata, matrix_shape))
    return metadata_path


def _channel_mode(channel: dict[str, Any]) -> str:
    metadata = channel.get("metadata", {})
    mode_info = metadata.get("ModeInfo") or metadata.get("ModelInfo") or {}
    return str(mode_info.get("ImageShowMode", ""))


def _channel_modes(channels: list[dict[str, Any]]) -> list[str]:
    return [_channel_mode(channel) for channel in channels]


def _unit_from_mode(mode: str) -> str:
    match = re.search(r"\((.*?)\)", mode)
    return match.group(1) if match and match.group(1) else DEFAULT_PHYSICAL_UNIT


def _normalized_channel_name(mode: str, fallback: str = "channel") -> str:
    lower = mode.lower()
    if "heightsensor" in lower or "height sensor" in lower:
        return "height_sensor"
    if "height" in lower:
        return "height"
    if "current" in lower:
        return "current"
    if any(token in lower for token in ("dcbias", "dc bias", "dc_bias", "cpd")):
        return "dcbias"
    if any(token in lower for token in ("amplitude", "amp")):
        return "amplitude"
    return fallback


def _metadata_summary_for_manifest(metadata: dict[str, Any]) -> dict[str, Any]:
    mode_info = (
        metadata.get("ModeInfo") or metadata.get("ModelInfo") or {}
        if metadata
        else {}
    )
    resolution = metadata.get("Resolution", {}) if metadata else {}
    scan_time = metadata.get("ScanTime", {}) if metadata else {}
    mode = str(mode_info.get("ImageShowMode", ""))
    return {
        "scan_mode": mode_info.get("ScanMode"),
        "image_show_mode": mode,
        "direction": mode_info.get("Direction"),
        "lifted_state": mode_info.get("LiftedState"),
        "unit": _unit_from_mode(mode),
        "x_resolution": resolution.get("XResolution"),
        "y_resolution": resolution.get("YResolution"),
        "x_size": resolution.get("XSize"),
        "y_size": resolution.get("YSize"),
        "x_unit": resolution.get("XUnit") or DEFAULT_PHYSICAL_UNIT,
        "y_unit": resolution.get("YUnit") or DEFAULT_PHYSICAL_UNIT,
        "angle": resolution.get("Angle"),
        "scan_direction": resolution.get("ScanDirection"),
        "scan_line_type": resolution.get("ScanLineType"),
        "start_time": scan_time.get("StartTime"),
        "end_time": scan_time.get("EndTime"),
    }


def _manifest_channel_record(
    *,
    input_id: str,
    input_path: str,
    channel_id: str,
    index: int,
    raw_name: str,
    normalized_name: str,
    matrix_npy_path: Any,
    metadata_path: Any,
    metadata: dict[str, Any],
    metadata_source: str,
    summary: dict[str, Any],
) -> dict[str, Any]:
    metadata_summary = _metadata_summary_for_manifest(metadata)
    if not metadata_summary["image_show_mode"]:
        metadata_summary["image_show_mode"] = raw_name
    if not metadata_summary["unit"]:
        metadata_summary["unit"] = _unit_from_mode(raw_name)
    matrix_summary = dict(summary)
    shape = matrix_summary.get("shape") or []
    return {
        "channel_id": channel_id,
        "input_id": input_id,
        "input_path": input_path,
        "index": int(index),
        "raw_name": raw_name,
        "normalized_name": normalized_name,
        "unit": metadata_summary.get("unit", ""),
        "scan_mode": metadata_summary.get("scan_mode"),
        "direction": metadata_summary.get("direction"),
        "lifted_state": metadata_summary.get("lifted_state"),
        "shape": shape,
        "summary": matrix_summary,
        "metadata_source": metadata_source,
        "metadata_summary": metadata_summary,
        "matrix_npy_path": str(matrix_npy_path) if matrix_npy_path else "",
        "metadata_path": str(metadata_path) if metadata_path else "",
        "metadata": metadata,
    }


def _write_channels_tsv(path: Path, channels: list[dict[str, Any]]) -> None:
    columns = [
        "channel_id",
        "input_id",
        "index",
        "raw_name",
        "normalized_name",
        "unit",
        "scan_mode",
        "direction",
        "lifted_state",
        "shape",
        "finite_count",
        "value_min",
        "value_max",
        "value_median",
        "matrix_npy_path",
        "metadata_path",
    ]
    lines = ["\t".join(columns)]
    for channel in channels:
        summary = channel.get("summary", {}) or {}
        row = {
            "channel_id": channel.get("channel_id", ""),
            "input_id": channel.get("input_id", ""),
            "index": channel.get("index", ""),
            "raw_name": channel.get("raw_name", ""),
            "normalized_name": channel.get("normalized_name", ""),
            "unit": channel.get("unit", ""),
            "scan_mode": channel.get("scan_mode", ""),
            "direction": channel.get("direction", ""),
            "lifted_state": channel.get("lifted_state", ""),
            "shape": "x".join(str(item) for item in channel.get("shape", []) or []),
            "finite_count": summary.get("finite_count", ""),
            "value_min": summary.get("value_min", ""),
            "value_max": summary.get("value_max", ""),
            "value_median": summary.get("value_median", ""),
            "matrix_npy_path": channel.get("matrix_npy_path", ""),
            "metadata_path": channel.get("metadata_path", ""),
        }
        lines.append("\t".join(str(row[column]) for column in columns))
    write_text_file(path, "\n".join(lines) + "\n")


def _channel_records_from_parsed_channels(
    input_file: Path,
    channels: list[dict[str, Any]],
    out_dir: Path,
) -> list[dict[str, Any]]:
    source_stem = input_file.stem
    records: list[dict[str, Any]] = []
    for index, channel in enumerate(channels, start=1):
        matrix = np.asarray(channel["matrix"], dtype=float)
        metadata = metadata_with_defaults(
            channel.get("metadata", {}),
            matrix.shape,
        )
        raw_name = _channel_mode(channel) or f"channel_{index:02d}"
        channel_stem = f"{source_stem}_{index:02d}_{_safe_name(raw_name)}"
        npy_path = out_dir / f"{channel_stem}.npy"
        atomic_write_path(npy_path, lambda path: np.save(path, matrix))
        metadata_path = _write_channel_metadata(
            out_dir,
            channel_stem,
            metadata,
            matrix.shape,
        )
        records.append(
            {
                "index": index,
                "mode": raw_name,
                "normalized_name": _normalized_channel_name(raw_name, f"channel_{index:02d}"),
                "metadata_source": channel.get("metadata_source", ""),
                "shape": list(matrix.shape),
                "summary": finite_matrix_summary(matrix),
                "npy_path": npy_path,
                "metadata_path": metadata_path,
                "metadata": metadata,
                "source_path": str(input_file),
            }
        )
    return records


def _build_manifest_channels(
    payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    input_path = str(payload["input_path"])
    input_id = "input_001"
    raw_channels = list(payload["channels"])
    inputs = [
        {
            "input_id": input_id,
            "path": input_path,
            "channel_count": len(raw_channels),
        }
    ]
    channels = []
    for index, channel in enumerate(raw_channels, start=1):
        raw_name = str(channel.get("mode") or f"channel_{index:02d}")
        channels.append(
            _manifest_channel_record(
                input_id=input_id,
                input_path=input_path,
                channel_id=f"input_001_ch{index:03d}",
                index=index,
                raw_name=raw_name,
                normalized_name=str(
                    channel.get("normalized_name")
                    or _normalized_channel_name(raw_name, f"channel_{index:02d}")
                ),
                matrix_npy_path=channel["npy_path"],
                metadata_path=channel["metadata_path"],
                metadata=channel.get("metadata", {}),
                metadata_source=str(channel.get("metadata_source", "")),
                summary=channel["summary"],
            )
        )
    return inputs, channels


def _attach_read_manifest(payload: dict[str, Any], output_dir: str) -> dict[str, Any]:
    out_dir = ensure_output_dir(output_dir)
    source_stem = str(payload["source_stem"])
    manifest_stem = _safe_name(source_stem)
    manifest_json = out_dir / f"{manifest_stem}_data_manifest.json"
    channel_table = out_dir / f"{manifest_stem}_channels.tsv"
    inputs, channels = _build_manifest_channels(payload)

    shapes = [tuple(channel.get("shape", [])) for channel in channels]
    unique_shapes = sorted({shape for shape in shapes if shape})
    normalized_names = [str(channel.get("normalized_name", "")) for channel in channels]
    duplicate_names = sorted(
        {name for name in normalized_names if name and normalized_names.count(name) > 1}
    )
    warnings: list[dict[str, Any]] = []
    if len(unique_shapes) > 1:
        warnings.append(
            {
                "code": "channel_shape_mismatch",
                "message": "Detected channels do not all have the same matrix shape.",
                "shapes": [list(shape) for shape in unique_shapes],
            }
        )
    if duplicate_names:
        warnings.append(
            {
                "code": "duplicate_normalized_channel",
                "message": (
                    "Multiple channels share a normalized name; downstream "
                    "selection may require channel_id."
                ),
                "normalized_names": duplicate_names,
            }
        )

    source_path = str(payload.get("input_path", ""))
    if inputs:
        source_path = str(inputs[0].get("path", source_path))
    first_channel = channels[0] if channels else {}
    metadata_summary = first_channel.get("metadata_summary", {}) or {}
    x_resolution = metadata_summary.get("x_resolution")
    y_resolution = metadata_summary.get("y_resolution")
    x_size = metadata_summary.get("x_size")
    y_size = metadata_summary.get("y_size")

    def _pixel_size(size: Any, resolution: Any) -> float | None:
        try:
            size_value = float(size)
            resolution_value = float(resolution)
        except (TypeError, ValueError):
            return None
        if resolution_value <= 0:
            return None
        return size_value / resolution_value

    scan = {
        "shape": list(unique_shapes[0]) if len(unique_shapes) == 1 else None,
        "x_size": x_size,
        "y_size": y_size,
        "x_unit": metadata_summary.get("x_unit"),
        "y_unit": metadata_summary.get("y_unit"),
        "x_resolution": x_resolution,
        "y_resolution": y_resolution,
        "pixel_width": _pixel_size(x_size, x_resolution),
        "pixel_height": _pixel_size(y_size, y_resolution),
    }

    manifest = {
        "status": "ok",
        "step": "data_ingest",
        "schema": DATA_MANIFEST_SCHEMA,
        "data_id": manifest_stem,
        "source_stem": source_stem,
        "source": {
            "input_path": source_path,
            "source_stem": source_stem,
        },
        "detected_format": payload.get("detected_format", {}),
        "input_count": len(inputs),
        "channel_count": len(channels),
        "inputs": inputs,
        "scan": scan,
        "channels": channels,
        "summary": {
            "input_count": len(inputs),
            "channel_count": len(channels),
            "normalized_channels": normalized_names,
            "same_shape": len(unique_shapes) <= 1,
        },
        "warnings": warnings,
        "artifacts": {
            "data_manifest_path": manifest_json,
            "channel_table_path": channel_table,
        },
    }
    write_json_file(manifest_json, manifest)
    _write_channels_tsv(channel_table, channels)

    artifacts = dict(payload.get("artifacts", {}))
    artifacts["data_manifest_path"] = str(manifest_json)
    artifacts["channel_table_path"] = str(channel_table)
    payload["artifacts"] = artifacts
    payload["manifest"] = {
        "schema": manifest["schema"],
        "input_count": len(inputs),
        "channel_count": len(channels),
        "scan": scan,
        "warnings": warnings,
        "channels": [
            {
                "channel_id": channel["channel_id"],
                "input_id": channel["input_id"],
                "raw_name": channel["raw_name"],
                "normalized_name": channel["normalized_name"],
                "unit": channel["unit"],
                "shape": channel["shape"],
                "matrix_npy_path": channel["matrix_npy_path"],
                "metadata_path": channel["metadata_path"],
            }
            for channel in channels
        ],
    }
    return payload


def _write_generic_multichannel(
    input_path: str,
    channels: list[dict[str, Any]],
    output_dir: str,
) -> dict[str, Any]:
    input_file = resolve_readable_file(input_path)
    out_dir = ensure_output_dir(output_dir)
    source_stem = input_file.stem
    channel_records = _channel_records_from_parsed_channels(input_file, channels, out_dir)

    payload = {
        "status": "ok",
        "step": "read_afm_data",
        "detected_format": {
            "kind": "single_channel" if len(channels) == 1 else "channel_bundle",
            "input_count": 1,
            "channel_count": len(channels),
            "channel_modes": _channel_modes(channels),
            "metadata_sources": [channel.get("metadata_source", "") for channel in channels],
        },
        "source_stem": source_stem,
        "input_path": str(input_file),
        "channels": channel_records,
    }
    return _attach_read_manifest(payload, output_dir)


def read_afm_data(
    input_path: str,
    output_dir: str,
) -> dict[str, Any]:
    """Read one AFM file and persist every header-detected channel."""

    if not input_path:
        raise ValueError("input_path is required")

    channels = parse_afm_channels(input_path)
    return _write_generic_multichannel(input_path, channels, output_dir)

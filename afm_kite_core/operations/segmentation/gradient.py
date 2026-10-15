"""Gradient-based region segmentation workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from skimage.measure import label

from afm_kite_core.algorithms.segmentation import (
    describe_target_regions,
    segment_gradient_masks,
)
from afm_kite_core.contracts import DATA_MANIFEST_SCHEMA, GRADIENT_REGIONS_SCHEMA
from afm_kite_core.data_io.matrix import load_height_data
from afm_kite_core.artifacts.writers import (
    atomic_write_path,
    read_json_file,
    write_json_file,
)
from afm_kite_core.data_io.paths import ensure_output_dir, resolve_readable_file


def _channel_search_text(record: dict[str, Any]) -> str:
    metadata = record.get("metadata", {})
    mode_info = metadata.get("ModeInfo") or metadata.get("ModelInfo") or {}
    metadata_mode = mode_info.get("ImageShowMode", "")
    return " ".join(
        str(value)
        for value in (
            record.get("normalized_name", ""),
            record.get("mode", ""),
            record.get("raw_name", ""),
            metadata_mode,
        )
    ).lower()


def _height_channel_kind(record: dict[str, Any]) -> str:
    """Return the exact height-channel kind without relying on record order."""

    metadata = record.get("metadata", {})
    mode_info = metadata.get("ModeInfo") or metadata.get("ModelInfo") or {}
    metadata_summary = record.get("metadata_summary", {}) or {}
    names = (
        record.get("raw_name", ""),
        record.get("mode", ""),
        mode_info.get("ImageShowMode", ""),
        metadata_summary.get("image_show_mode", ""),
        record.get("normalized_name", ""),
    )
    for name in names:
        base_name = str(name).split("(", 1)[0]
        token = "".join(
            character for character in base_name.casefold() if character.isalnum()
        )
        if token == "height":
            return "height"
        if token == "heightsensor":
            return "height_sensor"
    return ""


def _find_height_channel_record(channels_payload: dict[str, Any]) -> dict[str, Any]:
    records = list(channels_payload.get("channels", []))
    for preferred_kind in ("height", "height_sensor"):
        for record in records:
            if _height_channel_kind(record) == preferred_kind:
                return record
    return _find_channel_record(
        channels_payload,
        ("height",),
        "Height or HeightSensor",
    )


def _find_channel_record(
    channels_payload: dict[str, Any],
    keywords: tuple[str, ...],
    role: str,
) -> dict[str, Any]:
    records = list(channels_payload.get("channels", []))
    for record in records:
        search_text = _channel_search_text(record)
        if any(keyword in search_text for keyword in keywords):
            return record
    names = [
        record.get("normalized_name") or record.get("mode") or record.get("raw_name")
        for record in records
    ]
    raise ValueError(f"Could not identify {role} channel from ingest output. Found: {names}")


def _load_manifest(manifest_path: str) -> dict[str, Any]:
    return read_json_file(
        manifest_path,
        expected_schema=DATA_MANIFEST_SCHEMA,
        required_keys=("channels",),
    )[1]


def segment_gradient_regions(
    matrix_data_path: str,
    output_dir: str,
    boundary_percentile: float,
    boundary_width_px: int,
    min_region_area_px: int,
    max_target_regions: int,
    source_manifest_path: str,
    segmentation_method: str,
    gradient_sigma: str | float,
    interior_buffer_px: int,
) -> dict[str, Any]:
    """Segment candidate interiors and boundary pixels from a matrix."""

    ingest_payload: dict[str, Any] | None = None
    height_record: dict[str, Any] | None = None
    if source_manifest_path:
        ingest_payload = _load_manifest(source_manifest_path)
        height_record = _find_height_channel_record(ingest_payload)

    height_source_path = resolve_readable_file(matrix_data_path)
    height = load_height_data(height_source_path)
    out_dir = ensure_output_dir(output_dir)
    source_stem = (
        ingest_payload.get("source_stem") if ingest_payload else ""
    ) or Path(matrix_data_path).stem

    segmentation = segment_gradient_masks(
        height,
        boundary_percentile,
        boundary_width_px,
        min_region_area_px,
        max_target_regions,
        segmentation_method,
        gradient_sigma,
        interior_buffer_px,
    )
    gradient = segmentation["gradient"]
    boundary = segmentation["boundary"]
    boundary_buffer = segmentation["boundary_buffer"]
    interior_mask = segmentation["clean_interior"]
    threshold = segmentation["threshold"]
    finite = np.isfinite(height)
    target_labels = segmentation["target_labels"]
    target_regions = describe_target_regions(
        target_labels,
        height,
    )
    interior_components = label(interior_mask)

    gradient_npy = out_dir / f"{source_stem}_height_gradient.npy"
    boundary_npy = out_dir / f"{source_stem}_boundary_mask.npy"
    boundary_buffer_npy = out_dir / f"{source_stem}_boundary_buffer_mask.npy"
    interior_npy = out_dir / f"{source_stem}_interior_mask.npy"
    target_labels_npy = out_dir / f"{source_stem}_target_region_labels.npy"
    segmentation_json = out_dir / f"{source_stem}_gradient_regions.json"

    atomic_write_path(gradient_npy, lambda path: np.save(path, gradient))
    atomic_write_path(
        boundary_npy,
        lambda path: np.save(path, boundary.astype(bool)),
    )
    atomic_write_path(
        boundary_buffer_npy,
        lambda path: np.save(path, boundary_buffer.astype(bool)),
    )
    atomic_write_path(
        interior_npy,
        lambda path: np.save(path, interior_mask.astype(bool)),
    )
    atomic_write_path(
        target_labels_npy,
        lambda path: np.save(path, target_labels.astype(np.int32)),
    )

    payload = {
        "status": "ok",
        "step": "segment_gradient_regions",
        "schema": GRADIENT_REGIONS_SCHEMA,
        "basis": "gradient",
        "feature_type": "gradient_boundary_and_interiors",
        "matrix_data_path": str(height_source_path),
        "source_manifest_path": (
            str(resolve_readable_file(source_manifest_path))
            if source_manifest_path
            else ""
        ),
        "source_stem": source_stem,
        "method": (
            "provided matrix + multi-candidate gradient boundary mask + "
            "clean-interior connected-component targets"
        ),
        "leveling": {
            "performed": False,
            "note": (
                "Run matrix_level first and pass the detrended matrix as "
                "matrix_data_path when leveled input is required."
            ),
        },
        "height_channel": {
            "channel_id": height_record.get("channel_id") if height_record else "",
            "index": height_record.get("index") if height_record else None,
            "raw_name": height_record.get("raw_name") if height_record else "",
            "normalized_name": height_record.get("normalized_name") if height_record else "",
            "mode": height_record.get("mode") if height_record else "",
            "matrix_path": str(height_source_path),
        },
        "boundary_percentile": float(boundary_percentile),
        "boundary_width_px": int(boundary_width_px),
        "segmentation_method": str(segmentation_method),
        "selected_segmentation_method": segmentation["selected_method"],
        "selected_segmentation_score": segmentation["selected_score"],
        "gradient_sigma": str(gradient_sigma),
        "gradient_sigmas_used": segmentation["gradient_sigmas"],
        "interior_buffer_px": int(interior_buffer_px),
        "boundary_gradient_threshold": threshold,
        "candidate_diagnostics": segmentation["candidate_diagnostics"],
        "boundary_pixel_fraction": float(np.mean(boundary[finite])),
        "boundary_buffer_pixel_fraction": float(np.mean(boundary_buffer[finite])),
        "interior_pixel_fraction": float(np.mean(interior_mask[finite])),
        "interior_component_count": int(np.max(interior_components)),
        "feature_count": len(target_regions),
        "target_region_label_values": [
            int(item["label_value"]) for item in target_regions
        ],
        "features": target_regions,
        "min_region_area_px": int(min_region_area_px),
        "max_target_regions": int(max_target_regions),
        "artifacts": {
            "matrix_data_path": height_source_path,
            "height_gradient_npy_path": gradient_npy,
            "boundary_mask_npy_path": boundary_npy,
            "boundary_buffer_mask_npy_path": boundary_buffer_npy,
            "interior_mask_npy_path": interior_npy,
            "feature_labels_npy_path": target_labels_npy,
            "segmentation_json_path": segmentation_json,
        },
    }
    write_json_file(segmentation_json, payload)
    return payload

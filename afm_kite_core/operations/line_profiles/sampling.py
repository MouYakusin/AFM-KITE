"""Line-profile sampling stage."""

from __future__ import annotations

from typing import Any

import numpy as np

from afm_kite_core.algorithms.line_profiles import (
    physical_distance_along_line,
    sample_matrix_band_along_coordinates,
)
from afm_kite_core.algorithms.statistics import finite_matrix_summary
from afm_kite_core.contracts import FEATURE_LINE_SCHEMA, LINE_PROFILE_SCHEMA
from afm_kite_core.data_io.matrix import (
    height_unit,
    load_height_data_and_metadata,
    physical_area_context_from_metadata,
    resolve_matrix_metadata_path,
)
from afm_kite_core.artifacts.writers import (
    atomic_write_path,
    read_json_file,
    write_json_file,
)
from afm_kite_core.data_io.paths import ensure_output_dir

from .contracts import line_id, safe_profile_name


def aggregate_band(
    band_values: np.ndarray,
    aggregation: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate finite band samples and retain per-position diagnostics."""

    if aggregation not in {"median", "mean"}:
        raise ValueError("aggregation must be median or mean")
    values = np.asarray(band_values, dtype=float)
    if values.ndim != 2:
        raise ValueError("band_values must be two-dimensional")
    finite_count = np.sum(np.isfinite(values), axis=0).astype(int)
    aggregated = np.full(values.shape[1], np.nan, dtype=float)
    band_std = np.full(values.shape[1], np.nan, dtype=float)
    usable = finite_count > 0
    if np.any(usable):
        usable_values = values[:, usable]
        if aggregation == "median":
            aggregated[usable] = np.nanmedian(usable_values, axis=0)
        else:
            aggregated[usable] = np.nanmean(usable_values, axis=0)
        band_std[usable] = np.nanstd(usable_values, axis=0)
    return aggregated, finite_count, band_std


def sample_line_profile(
    line_contract_json_path: str,
    matrix_data_path: str,
    output_dir: str,
    profile_label: str,
    metadata_path: str,
    interpolation: str,
    line_width_px: int,
    aggregation: str,
) -> dict[str, Any]:
    """Sample one matrix along a reusable center line or perpendicular band."""

    line_path, line_payload = read_json_file(
        line_contract_json_path,
        expected_schema=FEATURE_LINE_SCHEMA,
        required_keys=("line_id", "coordinates", "windows"),
    )
    matrix, metadata, matrix_path = load_height_data_and_metadata(
        matrix_data_path,
        metadata_path,
    )
    expected_shape = tuple(int(item) for item in line_payload.get("shape", []))
    if expected_shape and tuple(matrix.shape) != expected_shape:
        raise ValueError(
            f"Matrix shape {matrix.shape} does not match line shape {expected_shape}"
        )

    coordinates = line_payload["coordinates"]
    x_px = np.asarray(coordinates["x_px"], dtype=float)
    y_px = np.asarray(coordinates["y_px"], dtype=float)
    if x_px.size < 3:
        raise ValueError("Line contract contains too few coordinates")
    band_values, _band_x_px, _band_y_px, offsets_px = (
        sample_matrix_band_along_coordinates(
            matrix,
            x_px,
            y_px,
            line_width_px=line_width_px,
            interpolation=interpolation,
        )
    )
    values, band_finite_count, band_std = aggregate_band(
        band_values,
        aggregation,
    )

    if int(line_width_px) == 1 and not np.allclose(
        offsets_px,
        np.asarray([0.0]),
    ):
        raise ValueError("Internal width-one band offset is not centered")

    physical_context = (
        physical_area_context_from_metadata(metadata) if metadata else {}
    )
    pixel_width = float(physical_context.get("pixel_width", 1.0))
    pixel_height = float(physical_context.get("pixel_height", 1.0))
    distance_unit = str(physical_context.get("x_unit") or "μm")
    distance = physical_distance_along_line(
        x_px,
        y_px,
        pixel_width,
        pixel_height,
    )
    value_unit = height_unit(metadata) if metadata else "μm"
    label = safe_profile_name(profile_label or matrix_path.stem)
    source_stem = safe_profile_name(
        f"{line_payload.get('source_stem', line_path.stem)}_{label}"
    )
    out_dir = ensure_output_dir(output_dir)
    profile_tsv = out_dir / f"{source_stem}_line_profile.tsv"
    profile_npy = out_dir / f"{source_stem}_line_profile.npy"
    profile_band_npy = out_dir / f"{source_stem}_line_profile_band.npy"
    profile_json = out_dir / f"{source_stem}_line_profile.json"
    segments = np.asarray(
        coordinates.get("segment", [""] * x_px.size),
        dtype=object,
    )
    rows = np.column_stack(
        [np.arange(x_px.size), x_px, y_px, distance, segments, values]
    )
    atomic_write_path(
        profile_tsv,
        lambda path: np.savetxt(
            path,
            rows,
            delimiter="\t",
            fmt="%s",
            header=(
                "sample_index\tx_px\ty_px\tdistance_{}\tsegment\t{}"
            ).format(safe_profile_name(distance_unit), label),
            comments="",
        ),
    )
    atomic_write_path(profile_npy, lambda path: np.save(path, values))
    atomic_write_path(profile_band_npy, lambda path: np.save(path, band_values))

    payload: dict[str, Any] = {
        "status": "ok",
        "step": "line_profile_sample",
        "schema": LINE_PROFILE_SCHEMA,
        "line_id": line_id(line_payload),
        "line_contract_json_path": line_path,
        "matrix_data_path": matrix_path,
        "metadata_path": str(
            resolve_matrix_metadata_path(matrix_path, metadata_path) or ""
        ),
        "profile_label": label,
        "source_stem": source_stem,
        "shape": list(matrix.shape),
        "sample_count": int(values.size),
        "interpolation": interpolation,
        "sampling": {
            "mode": "center_line" if int(line_width_px) == 1 else "band",
            "line_width_px": int(line_width_px),
            "aggregation": aggregation,
            "interpolation": interpolation,
            "offsets_px": offsets_px,
            "out_of_bounds_policy": "omit",
            "finite_count_per_position": band_finite_count,
            "fully_missing_position_count": int(
                np.sum(band_finite_count == 0)
            ),
            "standard_deviation_per_position": band_std,
        },
        "value_unit": value_unit,
        "physical_context": physical_context,
        "line": line_payload.get("line", {}),
        "windows": line_payload["windows"],
        "coordinates": {
            "x_px": x_px,
            "y_px": y_px,
            "distance": distance,
            "distance_unit": distance_unit,
            "segment": segments.tolist(),
        },
        "values": values,
        "summary": finite_matrix_summary(values.reshape(1, -1)),
        "artifacts": {
            "line_contract_json_path": line_path,
            "matrix_data_path": matrix_path,
            "metadata_path": str(
                resolve_matrix_metadata_path(matrix_path, metadata_path) or ""
            ),
            "line_profile_tsv": profile_tsv,
            "line_profile_npy": profile_npy,
            "line_profile_band_npy": profile_band_npy,
            "line_profile_json": profile_json,
        },
    }
    write_json_file(profile_json, payload)
    return payload


__all__ = ["aggregate_band", "sample_line_profile"]

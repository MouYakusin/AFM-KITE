"""Median-profile extraction stage implementation."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage as ndi

from afm_kite_core.algorithms.leveling import level_matrix
from afm_kite_core.algorithms.profiles import median_profile, projected_median_profile
from afm_kite_core.algorithms.statistics import surface_stats
from afm_kite_core.data_io.matrix import (
    height_unit,
    load_height_data_and_metadata,
    physical_area_context_from_metadata,
)
from afm_kite_core.artifacts.writers import atomic_write_path
from afm_kite_core.data_io.paths import ensure_output_dir


def extract_median_profile(
    height_data_path: str,
    output_dir: str,
    profile_axis: str,
    leveling: str,
    smooth_sigma_px: float,
    metadata_path: str,
    profile_angle_degrees: float | None,
    projection_bin_size_px: float,
) -> dict[str, Any]:
    """Extract and save a reusable axis-aligned or projected median profile."""

    if profile_axis not in {"x", "y", "angle"}:
        raise ValueError("profile_axis must be x, y, or angle")
    if profile_axis == "angle" and profile_angle_degrees is None:
        raise ValueError("--profile-angle-degrees is required when --profile-axis angle")
    if leveling not in {"", "none"}:
        raise ValueError(
            "profile extraction does not perform leveling; run level_height_data first "
            "and pass the leveled matrix"
        )

    matrix, metadata, resolved_input = load_height_data_and_metadata(
        height_data_path,
        metadata_path,
    )
    physical_context = physical_area_context_from_metadata(metadata) if metadata else {}
    leveled, leveling_diagnostics = level_matrix(matrix, "none", physical_context)
    projection: dict[str, object] = {}
    if profile_axis == "angle":
        coordinates, profile, projection = projected_median_profile(
            leveled,
            float(profile_angle_degrees),
            projection_bin_size_px,
        )
        axis_label = f"angle_{float(profile_angle_degrees):.3f}".replace(
            "-", "minus_").replace(".", "p")
    else:
        profile = median_profile(leveled, profile_axis)
        coordinates = np.arange(profile.size, dtype=float)
        axis_label = profile_axis
    if float(smooth_sigma_px) > 0:
        smoothed = ndi.gaussian_filter1d(profile, sigma=float(smooth_sigma_px))
    else:
        smoothed = profile.copy()

    out_dir = ensure_output_dir(output_dir)
    profile_tsv = out_dir / f"{resolved_input.stem}_{axis_label}_median_profile.tsv"
    profile_npy = out_dir / f"{resolved_input.stem}_{axis_label}_median_profile.npy"
    table = np.c_[coordinates, profile, smoothed]
    atomic_write_path(
        profile_tsv,
        lambda path: np.savetxt(
            path,
            table,
            delimiter="\t",
            fmt="%.12g",
            header="coordinate_px\tprofile_height\tsmoothed_profile_height",
            comments="",
        ),
    )
    atomic_write_path(profile_npy, lambda path: np.save(path, smoothed))

    payload = {
        "status": "ok",
        "step": "extract_median_profile",
        "input_path": str(resolved_input),
        "profile_axis": profile_axis,
        "profile_angle_degrees": (
            float(profile_angle_degrees)
            if profile_angle_degrees is not None
            else None
        ),
        "projection_bin_size_px": float(projection_bin_size_px),
        "projection": projection,
        "leveling": leveling_diagnostics,
        "physical_context": physical_context,
        "smooth_sigma_px": float(smooth_sigma_px),
        "height_unit": height_unit(metadata) if metadata else "μm",
        "profile_length": int(profile.size),
        "profile_stats": surface_stats(smoothed),
        "profile_tsv_path": profile_tsv,
        "profile_npy_path": profile_npy,
    }
    return payload


__all__ = ["extract_median_profile"]

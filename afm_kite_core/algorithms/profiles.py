"""One-dimensional profile algorithms operating on in-memory matrices."""

from __future__ import annotations

from typing import Any

import numpy as np


def median_profile(matrix: np.ndarray, profile_axis: str) -> np.ndarray:
    """Extract an X or Y median profile from a 2D matrix."""

    matrix = np.asarray(matrix, dtype=float)
    if profile_axis == "x":
        return np.nanmedian(matrix, axis=0)
    if profile_axis == "y":
        return np.nanmedian(matrix, axis=1)
    raise ValueError("profile_axis must be x or y")


def projected_median_profile(
    matrix: np.ndarray,
    angle_degrees: float,
    bin_size_px: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Extract a median profile along an arbitrary projection direction.

    The angle specifies the profile-variation direction in image coordinates.
    Zero degrees matches the X-axis profile.
    Ninety degrees matches the Y-axis profile.
    This direction is normal to grating stripes.
    """

    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D")
    bin_size = float(bin_size_px)
    if not np.isfinite(bin_size) or bin_size <= 0:
        raise ValueError("bin_size_px must be a positive finite value")

    theta = np.deg2rad(float(angle_degrees))
    unit_x = float(np.cos(theta))
    unit_y = float(np.sin(theta))
    yy, xx = np.indices(matrix.shape, dtype=float)
    projection = (xx * unit_x) + (yy * unit_y)
    finite = np.isfinite(matrix)
    if not np.any(finite):
        raise ValueError("matrix contains no finite values")

    projection_values = projection[finite]
    height_values = matrix[finite]
    projection_origin = float(np.nanmin(projection_values))
    bin_indices = np.floor((projection_values - projection_origin) / bin_size).astype(np.int64)
    order = np.argsort(bin_indices, kind="stable")
    sorted_bins = bin_indices[order]
    sorted_values = height_values[order]
    unique_bins, starts, counts = np.unique(
        sorted_bins,
        return_index=True,
        return_counts=True,
    )

    profile = np.empty(unique_bins.size, dtype=float)
    for idx, (start, count) in enumerate(zip(starts, counts)):
        profile[idx] = float(np.nanmedian(sorted_values[start: start + count]))

    coordinates = (unique_bins.astype(float) + 0.5) * bin_size
    diagnostics = {
        "mode": "projected",
        "angle_degrees": float(angle_degrees),
        "bin_size_px": bin_size,
        "projection_origin_px": projection_origin,
        "unit_vector_xy": [unit_x, unit_y],
        "profile_length": int(profile.size),
        "sample_count": int(height_values.size),
        "min_bin_count": int(np.min(counts)),
        "median_bin_count": float(np.median(counts)),
        "max_bin_count": int(np.max(counts)),
    }
    return coordinates, profile, diagnostics


__all__ = ["median_profile", "projected_median_profile"]

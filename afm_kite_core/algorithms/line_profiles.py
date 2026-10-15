"""In-memory sampling and measurement primitives for explicit line profiles."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage as ndi


def sample_matrix_along_coordinates(
    matrix: np.ndarray,
    x_px: np.ndarray,
    y_px: np.ndarray,
    interpolation: str = "linear",
) -> np.ndarray:
    """Sample one matrix at explicit pixel coordinates."""

    matrix = np.asarray(matrix, dtype=float)
    x_px = np.asarray(x_px, dtype=float)
    y_px = np.asarray(y_px, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if x_px.ndim != 1 or y_px.ndim != 1 or x_px.size != y_px.size:
        raise ValueError("x_px and y_px must be same-length one-dimensional arrays")
    if interpolation not in {"linear", "nearest"}:
        raise ValueError("interpolation must be linear or nearest")
    order = 1 if interpolation == "linear" else 0
    return ndi.map_coordinates(matrix, [y_px, x_px], order=order, mode="nearest")


def sample_matrix_band_along_coordinates(
    matrix: np.ndarray,
    x_px: np.ndarray,
    y_px: np.ndarray,
    line_width_px: int = 1,
    interpolation: str = "linear",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample a perpendicular pixel band around explicit center-line coordinates.

    Returns band values, X coordinates, Y coordinates, and symmetric perpendicular offsets.
    Out-of-bounds band samples are NaN.
    Downstream aggregation can omit them instead of duplicating the nearest image edge.
    """

    matrix = np.asarray(matrix, dtype=float)
    x_px = np.asarray(x_px, dtype=float)
    y_px = np.asarray(y_px, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if x_px.ndim != 1 or y_px.ndim != 1 or x_px.size != y_px.size:
        raise ValueError("x_px and y_px must be same-length one-dimensional arrays")
    if x_px.size < 2:
        raise ValueError("at least two line coordinates are required")
    if isinstance(line_width_px, bool) or int(line_width_px) != line_width_px:
        raise ValueError("line_width_px must be a positive integer")
    width = int(line_width_px)
    if width < 1:
        raise ValueError("line_width_px must be a positive integer")
    if interpolation not in {"linear", "nearest"}:
        raise ValueError("interpolation must be linear or nearest")

    tangent_x = np.gradient(x_px)
    tangent_y = np.gradient(y_px)
    tangent_norm = np.hypot(tangent_x, tangent_y)
    if np.any(tangent_norm <= 1e-12):
        raise ValueError("line coordinates contain a zero-length local direction")
    normal_x = -tangent_y / tangent_norm
    normal_y = tangent_x / tangent_norm
    offsets_px = np.arange(width, dtype=float) - ((width - 1.0) / 2.0)
    band_x_px = x_px[np.newaxis, :] + offsets_px[:, np.newaxis] * normal_x
    band_y_px = y_px[np.newaxis, :] + offsets_px[:, np.newaxis] * normal_y

    order = 1 if interpolation == "linear" else 0
    band_values = ndi.map_coordinates(
        matrix,
        [band_y_px.ravel(), band_x_px.ravel()],
        order=order,
        mode="nearest",
    ).reshape(band_x_px.shape)
    rows, cols = matrix.shape
    in_bounds = (
        (band_x_px >= 0.0)
        & (band_x_px <= float(cols - 1))
        & (band_y_px >= 0.0)
        & (band_y_px <= float(rows - 1))
    )
    band_values = np.where(in_bounds, band_values, np.nan)
    return band_values, band_x_px, band_y_px, offsets_px


def physical_distance_along_line(
    x_px: np.ndarray,
    y_px: np.ndarray,
    pixel_width: float = 1.0,
    pixel_height: float = 1.0,
) -> np.ndarray:
    """Return cumulative physical distance along explicit coordinates."""

    x_px = np.asarray(x_px, dtype=float)
    y_px = np.asarray(y_px, dtype=float)
    if x_px.size == 0:
        return np.asarray([], dtype=float)
    dx = np.diff(x_px) * float(pixel_width)
    dy = np.diff(y_px) * float(pixel_height)
    steps = np.hypot(dx, dy)
    return np.r_[0.0, np.cumsum(steps)]


def window_stats(values: np.ndarray) -> dict[str, float | int | None]:
    """Return robust descriptive statistics for one profile window."""

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "std": None,
            "p05": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    return {
        "n": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(np.std(finite)),
        "p05": float(np.percentile(finite, 5.0)),
        "p95": float(np.percentile(finite, 95.0)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def measure_profile_windows(
    profile: np.ndarray,
    left_indices: np.ndarray,
    feature_indices: np.ndarray,
    right_indices: np.ndarray,
    distance: np.ndarray | None = None,
) -> dict[str, Any]:
    """Measure one feature window relative to its two substrate windows."""

    profile = np.asarray(profile, dtype=float)
    left = np.asarray(left_indices, dtype=int)
    feature = np.asarray(feature_indices, dtype=int)
    right = np.asarray(right_indices, dtype=int)
    if not left.size or not feature.size or not right.size:
        raise ValueError("left substrate, feature, and right substrate windows are required")
    for name, indices in (("left", left), ("feature", feature), ("right", right)):
        if np.min(indices) < 0 or np.max(indices) >= profile.size:
            raise ValueError(f"{name} window indices exceed profile length")

    substrate_values = np.r_[profile[left], profile[right]]
    feature_values = profile[feature]
    substrate = window_stats(substrate_values)
    feature_stats = window_stats(feature_values)
    if substrate["median"] is None or feature_stats["median"] is None:
        raise ValueError("Cannot measure feature/substrate contrast from empty windows")

    signed_contrast = float(feature_stats["median"] - substrate["median"])
    if signed_contrast >= 0:
        robust_extreme = float(feature_stats["p95"] - substrate["median"])
        direction = "higher"
    else:
        robust_extreme = float(feature_stats["p05"] - substrate["median"])
        direction = "lower"

    feature_width_samples = int(feature.size)
    feature_width = None
    if distance is not None:
        distance_values = np.asarray(distance, dtype=float)
        if distance_values.size == profile.size:
            if feature.size > 1:
                step = float(np.median(np.diff(distance_values)))
                feature_width = float(
                    distance_values[feature[-1]] - distance_values[feature[0]] + step
                )
            else:
                feature_width = 0.0

    left_stats = window_stats(profile[left])
    right_stats = window_stats(profile[right])
    substrate_asymmetry = None
    if left_stats["median"] is not None and right_stats["median"] is not None:
        substrate_asymmetry = float(right_stats["median"] - left_stats["median"])

    return {
        "substrate": substrate,
        "feature": feature_stats,
        "left_substrate": left_stats,
        "right_substrate": right_stats,
        "feature_minus_substrate_median": signed_contrast,
        "absolute_feature_substrate_contrast": abs(signed_contrast),
        "feature_robust_extreme_minus_substrate": robust_extreme,
        "direction": direction,
        "feature_width_samples": feature_width_samples,
        "feature_width": feature_width,
        "right_minus_left_substrate_median": substrate_asymmetry,
    }


__all__ = [
    "measure_profile_windows",
    "physical_distance_along_line",
    "sample_matrix_band_along_coordinates",
    "sample_matrix_along_coordinates",
    "window_stats",
]

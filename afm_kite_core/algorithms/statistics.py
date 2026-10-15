"""Reusable measurement primitives for AFM matrices."""

from __future__ import annotations

from typing import Any

import numpy as np


def surface_stats(values: np.ndarray) -> dict[str, float | int | None]:
    """Calculate standard surface roughness statistics for finite values."""

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "n": 0,
            "mean_height": None,
            "median_height": None,
            "min_height": None,
            "max_height": None,
            "Sa": None,
            "Sq": None,
            "Sz": None,
            "Ssk": None,
            "Sku": None,
        }

    mean_height = float(np.mean(values))
    centered = values - mean_height
    sq = float(np.sqrt(np.mean(centered**2)))
    if sq > 0:
        ssk = float(np.mean(centered**3) / (sq**3))
        sku = float(np.mean(centered**4) / (sq**4))
    else:
        ssk = 0.0
        sku = 0.0

    return {
        "n": int(values.size),
        "mean_height": mean_height,
        "median_height": float(np.median(values)),
        "min_height": float(np.min(values)),
        "max_height": float(np.max(values)),
        "Sa": float(np.mean(np.abs(centered))),
        "Sq": sq,
        "Sz": float(np.max(values) - np.min(values)),
        "Ssk": ssk,
        "Sku": sku,
    }


def robust_location_scale(values: np.ndarray) -> tuple[float, float]:
    """Return median and MAD-derived robust sigma for finite values."""

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Cannot estimate robust scale from empty values")

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    sigma = float(1.4826 * mad)
    if sigma <= 1e-12:
        sigma = float(np.std(values))
    if sigma <= 1e-12:
        sigma = max(float(np.ptp(values)) / 6.0, 1e-12)
    return median, sigma


def image_clip_limits(matrix: np.ndarray, percentile_clip: float = 1.0) -> tuple[float, float]:
    """Calculate robust display limits for a matrix image."""

    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        raise ValueError("Cannot determine image limits from all-NaN values")

    clip = max(0.0, min(float(percentile_clip), 20.0))
    vmin = float(np.percentile(finite, clip))
    vmax = float(np.percentile(finite, 100.0 - clip))
    if vmin == vmax:
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
    return vmin, vmax


def finite_matrix_summary(matrix: np.ndarray) -> dict[str, Any]:
    """Return shape and finite-value diagnostics without writing artifacts."""

    matrix = np.asarray(matrix, dtype=float)
    finite = matrix[np.isfinite(matrix)]
    return {
        "shape": list(matrix.shape),
        "finite_count": int(finite.size),
        "nan_count": int(matrix.size - finite.size),
        "value_min": float(np.min(finite)) if finite.size else None,
        "value_max": float(np.max(finite)) if finite.size else None,
        "value_median": float(np.median(finite)) if finite.size else None,
    }

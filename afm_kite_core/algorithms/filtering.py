"""Image-filtering algorithms operating on in-memory matrices."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage as ndi


def _normalized_filter(
    matrix: np.ndarray,
    finite: np.ndarray,
    operation: Any,
) -> np.ndarray:
    """Filter finite values without allowing missing pixels to contaminate neighbors."""

    values = operation(np.where(finite, matrix, 0.0))
    weights = operation(finite.astype(float))
    filtered = np.full(matrix.shape, np.nan, dtype=float)
    np.divide(values, weights, out=filtered, where=weights > np.finfo(float).eps)
    return filtered


def _nearest_finite_fill(matrix: np.ndarray, finite: np.ndarray) -> np.ndarray:
    """Fill invalid pixels from their nearest finite neighbor for derivative filters."""

    if np.all(finite):
        return matrix
    indices = ndi.distance_transform_edt(
        ~finite,
        return_distances=False,
        return_indices=True,
    )
    return matrix[tuple(indices)]


def _finite_median(values: np.ndarray) -> float:
    finite_values = values[np.isfinite(values)]
    return float(np.median(finite_values)) if finite_values.size else float("nan")


def filter_matrix(
    matrix: np.ndarray,
    filter_name: str = "gaussian",
    sigma: float = 1.0,
    size: int = 3,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply a named image filter and return the matrix plus diagnostics."""

    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.size == 0:
        raise ValueError("matrix must be a non-empty 2D array")
    finite = np.isfinite(matrix)
    if not np.any(finite):
        raise ValueError("Cannot filter a matrix with no finite values")

    filter_key = filter_name.lower().strip()
    size_value = max(int(size), 1)
    sigma_value = max(float(sigma), 0.0)

    if filter_key == "gaussian":
        filtered = _normalized_filter(
            matrix,
            finite,
            lambda values: ndi.gaussian_filter(values, sigma=sigma_value),
        )
    elif filter_key == "median":
        filtered = ndi.generic_filter(matrix, _finite_median, size=size_value)
    elif filter_key == "mean":
        filtered = _normalized_filter(
            matrix,
            finite,
            lambda values: ndi.uniform_filter(values, size=size_value),
        )
    elif filter_key == "sobel":
        filled = _nearest_finite_fill(matrix, finite)
        sx = ndi.sobel(filled, axis=1)
        sy = ndi.sobel(filled, axis=0)
        filtered = np.hypot(sx, sy)
    elif filter_key == "laplace":
        filtered = ndi.laplace(_nearest_finite_fill(matrix, finite))
    else:
        raise ValueError("filter_name must be gaussian, median, mean, sobel, or laplace")

    filtered[~finite] = np.nan
    return filtered, {
        "filter": filter_key,
        "sigma": sigma_value,
        "size": size_value,
        "nan_policy": "preserve_invalid_mask",
        "finite_input_count": int(np.count_nonzero(finite)),
        "invalid_input_count": int(matrix.size - np.count_nonzero(finite)),
    }


__all__ = ["filter_matrix"]

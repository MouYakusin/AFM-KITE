"""Reusable row, mean, background-mask, and candidate leveling operations."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage as ndi

from afm_kite_core.algorithms.statistics import robust_location_scale

from .surface_fit import mask_values, subtract_surface


def row_align(
    matrix: np.ndarray,
    mask: np.ndarray | None = None,
    quantile: float = 0.5,
) -> tuple[np.ndarray, dict[str, Any]]:
    matrix = np.asarray(matrix, dtype=float)
    finite = np.isfinite(matrix)
    usable = finite if mask is None else finite & np.asarray(mask, dtype=bool)
    row_refs = np.zeros(matrix.shape[0], dtype=float)
    fallback_rows = 0
    for row_index in range(matrix.shape[0]):
        row_values = matrix[row_index, usable[row_index]]
        if row_values.size < 3:
            row_values = matrix[row_index, finite[row_index]]
            fallback_rows += 1
        if row_values.size:
            row_refs[row_index] = float(np.quantile(row_values, quantile))
        else:
            row_refs[row_index] = 0.0
            fallback_rows += 1
    return matrix - row_refs[:, None], {
        "operation": "row_align",
        "quantile": float(quantile),
        "mask_used": mask is not None,
        "fallback_rows": int(fallback_rows),
        "row_reference_std": float(np.std(row_refs)) if row_refs.size else 0.0,
        "row_reference_min": float(np.min(row_refs)) if row_refs.size else 0.0,
        "row_reference_max": float(np.max(row_refs)) if row_refs.size else 0.0,
    }


def zero_to_mask_mean(
    matrix: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    values = mask_values(matrix, mask)
    basis = "mask"
    if values.size == 0:
        values = mask_values(matrix)
        basis = "finite"
    offset = float(np.mean(values)) if values.size else 0.0
    return matrix - offset, {
        "operation": "zero_mean",
        "basis": basis,
        "offset": offset,
        "points": int(values.size),
    }


def background_aware_pass(
    matrix: np.ndarray,
    mask: np.ndarray | None = None,
    include_poly2: bool = False,
    physical_context: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    leveled = np.asarray(matrix, dtype=float).copy()
    operations: list[dict[str, Any]] = []
    leveled, row_diag = row_align(leveled, mask=mask)
    operations.append(row_diag)
    leveled, plane_diag = subtract_surface(
        leveled,
        mask=mask,
        model="plane",
        physical_context=physical_context,
    )
    operations.append(plane_diag)
    if include_poly2:
        leveled, poly_diag = subtract_surface(
            leveled,
            mask=mask,
            model="poly2",
            physical_context=physical_context,
        )
        operations.append(poly_diag)
    leveled, zero_diag = zero_to_mask_mean(leveled, mask=mask)
    operations.append(zero_diag)
    return leveled, {
        "include_poly2": bool(include_poly2),
        "mask_used": mask is not None,
        "operations": operations,
    }


def _remove_small_components(mask: np.ndarray, min_size: int) -> np.ndarray:
    labeled, count = ndi.label(mask)
    if count == 0:
        return mask.astype(bool, copy=True)
    sizes = np.bincount(labeled.ravel())
    keep = sizes >= int(min_size)
    keep[0] = False
    return keep[labeled]


def foreground_background_masks(
    coarse_leveled: np.ndarray,
    sigma_threshold: float = 2.5,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    finite = np.isfinite(coarse_leveled)
    if not np.any(finite):
        raise ValueError("Cannot build mask from an all-NaN height matrix")
    median, sigma = robust_location_scale(coarse_leveled[finite])
    high = median + float(sigma_threshold) * sigma
    low = median - float(sigma_threshold) * sigma
    raw_foreground = finite & ((coarse_leveled > high) | (coarse_leveled < low))
    min_size = max(3, int(round(coarse_leveled.size * 0.00005)))
    foreground = _remove_small_components(raw_foreground, min_size=min_size)
    if int(foreground.sum()) == 0 and int(raw_foreground.sum()) > 0:
        foreground = raw_foreground
    dilation_iterations = max(1, min(coarse_leveled.shape) // 128)
    foreground = ndi.binary_dilation(foreground, iterations=int(dilation_iterations))
    foreground &= finite
    background = finite & ~foreground
    finite_count = int(finite.sum())
    return foreground, background, {
        "method": "robust_mad_threshold_on_first_pass",
        "sigma_threshold": float(sigma_threshold),
        "median": float(median),
        "robust_sigma": float(sigma),
        "low_threshold": float(low),
        "high_threshold": float(high),
        "min_component_size_px": int(min_size),
        "dilation_iterations": int(dilation_iterations),
        "finite_count": finite_count,
        "foreground_count": int(foreground.sum()),
        "background_count": int(background.sum()),
        "foreground_fraction": float(foreground.sum() / finite_count),
        "background_fraction": float(background.sum() / finite_count),
    }


def simple_candidate(
    matrix: np.ndarray,
    mode: str,
    physical_context: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    leveled = np.asarray(matrix, dtype=float).copy()
    operations: list[dict[str, Any]] = []
    finite = np.isfinite(leveled)
    if mode == "none":
        leveled, diag = zero_to_mask_mean(leveled, mask=finite)
        operations.append(diag)
    elif mode == "plane":
        leveled, diag = subtract_surface(
            leveled, finite, "plane", physical_context
        )
        operations.append(diag)
        leveled, diag = zero_to_mask_mean(leveled, mask=finite)
        operations.append(diag)
    elif mode == "line":
        leveled, diag = row_align(leveled)
        operations.append(diag)
        leveled, diag = zero_to_mask_mean(leveled, mask=finite)
        operations.append(diag)
    elif mode == "line_then_plane":
        leveled, diag = row_align(leveled)
        operations.append(diag)
        leveled, diag = subtract_surface(
            leveled, finite, "plane", physical_context
        )
        operations.append(diag)
        leveled, diag = zero_to_mask_mean(leveled, mask=finite)
        operations.append(diag)
    elif mode == "plane_then_line":
        leveled, diag = subtract_surface(
            leveled, finite, "plane", physical_context
        )
        operations.append(diag)
        leveled, diag = row_align(leveled)
        operations.append(diag)
        leveled, diag = zero_to_mask_mean(leveled, mask=finite)
        operations.append(diag)
    else:
        raise ValueError(f"Unknown simple candidate mode: {mode}")
    return leveled, {"mode": mode, "operations": operations}


__all__ = [
    "background_aware_pass",
    "foreground_background_masks",
    "row_align",
    "simple_candidate",
    "zero_to_mask_mean",
]

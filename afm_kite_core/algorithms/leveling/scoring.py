"""Background-flatness and foreground-preservation candidate scoring."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage as ndi

from .surface_fit import fit_surface, mask_values


def _median_profile_std(matrix: np.ndarray, mask: np.ndarray, axis: int) -> float:
    matrix = np.asarray(matrix, dtype=float)
    mask = np.asarray(mask, dtype=bool) & np.isfinite(matrix)
    medians: list[float] = []
    ranges = (
        (matrix[row, mask[row]] for row in range(matrix.shape[0]))
        if axis == 1
        else (matrix[mask[:, col], col] for col in range(matrix.shape[1]))
    )
    if axis not in {0, 1}:
        raise ValueError("axis must be 0 or 1")
    for values in ranges:
        if values.size >= 3:
            medians.append(float(np.median(values)))
    return float(np.std(np.asarray(medians))) if len(medians) >= 2 else 0.0


def _low_frequency_background_std(matrix: np.ndarray, mask: np.ndarray) -> float:
    values = mask_values(matrix, mask)
    if values.size == 0:
        return 0.0
    filled = np.where(np.isfinite(matrix) & mask, matrix, float(np.median(values)))
    sigma = max(2.0, min(matrix.shape) / 16.0)
    low_frequency = ndi.gaussian_filter(filled, sigma=float(sigma))
    return float(np.std(low_frequency[np.isfinite(matrix) & mask]))


def _surface_fit_range(
    matrix: np.ndarray,
    mask: np.ndarray,
    physical_context: dict[str, Any] | None = None,
) -> float:
    surface, diagnostics = fit_surface(matrix, mask, "plane", physical_context)
    if diagnostics.get("fit_status") != "ok":
        return 0.0
    valid = np.isfinite(matrix) & mask
    return (
        float(np.max(surface[valid]) - np.min(surface[valid]))
        if np.any(valid)
        else 0.0
    )


def feature_amplitude(
    matrix: np.ndarray,
    foreground: np.ndarray,
    background: np.ndarray,
) -> float:
    foreground_values = mask_values(matrix, foreground)
    background_values = mask_values(matrix, background)
    if foreground_values.size == 0 or background_values.size == 0:
        return 0.0
    background_median = float(np.median(background_values))
    return float(np.median(np.abs(foreground_values - background_median)))


def score_candidate(
    leveled: np.ndarray,
    mode: str,
    background: np.ndarray,
    foreground: np.ndarray,
    scale: float,
    reference_feature_amplitude: float,
    complexity_rank: int,
    physical_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    background_values = mask_values(leveled, background)
    if background_values.size == 0:
        background = np.isfinite(leveled)
        background_values = mask_values(leveled, background)
    background_mean = (
        float(np.mean(background_values)) if background_values.size else 0.0
    )
    background_sq = (
        float(np.sqrt(np.mean((background_values - background_mean) ** 2)))
        if background_values.size
        else 0.0
    )
    metrics = {
        "background_plane_range": _surface_fit_range(
            leveled, background, physical_context
        ),
        "row_median_std": _median_profile_std(leveled, background, axis=1),
        "col_median_std": _median_profile_std(leveled, background, axis=0),
        "low_frequency_background_std": _low_frequency_background_std(
            leveled, background
        ),
        "background_Sq": background_sq,
    }
    current_feature_amplitude = feature_amplitude(leveled, foreground, background)
    feature_loss_fraction = (
        max(
            0.0,
            (0.75 * reference_feature_amplitude - current_feature_amplitude)
            / reference_feature_amplitude,
        )
        if reference_feature_amplitude > 1e-12
        else 0.0
    )
    complexity_penalty = float(complexity_rank) * 0.01 * float(scale)
    overfit_penalty = float(feature_loss_fraction) * float(scale)
    weighted_score = (
        0.30 * metrics["background_plane_range"]
        + 0.25 * metrics["row_median_std"]
        + 0.15 * metrics["col_median_std"]
        + 0.15 * metrics["low_frequency_background_std"]
        + 0.10 * metrics["background_Sq"]
        + 0.05 * overfit_penalty
        + complexity_penalty
    )
    severe_overfit = bool(feature_loss_fraction > 0.60 and "poly2" in mode)
    return {
        "mode": mode,
        "score": float(weighted_score),
        "accepted": not severe_overfit,
        "rejection_reason": "foreground_feature_loss" if severe_overfit else "",
        "metrics": metrics,
        "foreground_feature_amplitude": float(current_feature_amplitude),
        "reference_feature_amplitude": float(reference_feature_amplitude),
        "feature_loss_fraction": float(feature_loss_fraction),
        "complexity_penalty": complexity_penalty,
        "overfit_penalty": overfit_penalty,
    }


__all__ = ["feature_amplitude", "score_candidate"]

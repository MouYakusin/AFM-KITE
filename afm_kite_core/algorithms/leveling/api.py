"""Public in-memory leveling API and automatic candidate orchestration."""

from __future__ import annotations

from typing import Any

import numpy as np

from afm_kite_core.algorithms.statistics import robust_location_scale

from .operations import (
    background_aware_pass,
    foreground_background_masks,
    simple_candidate,
)
from .scoring import feature_amplitude, score_candidate
from .surface_fit import subtract_surface


def auto_level_matrix(
    matrix: np.ndarray,
    physical_context: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray, np.ndarray]:
    """Select an AFM flattening candidate by background-aware scoring."""

    matrix = np.asarray(matrix, dtype=float)
    finite = np.isfinite(matrix)
    if not np.any(finite):
        raise ValueError("Cannot level an all-NaN height matrix")

    _, scale = robust_location_scale(matrix[finite])
    first_pass, first_pass_diag = background_aware_pass(
        matrix,
        include_poly2=True,
        physical_context=physical_context,
    )
    foreground, background, mask_diag = foreground_background_masks(first_pass)
    background_fraction = float(mask_diag["background_fraction"])

    candidate_order = [
        "none",
        "plane",
        "line",
        "line_then_plane",
        "plane_then_line",
        "background_aware_basic",
        "background_aware_poly2",
    ]
    complexity_rank = {mode: rank for rank, mode in enumerate(candidate_order)}
    candidate_arrays: dict[str, np.ndarray] = {}
    candidate_diagnostics: dict[str, dict[str, Any]] = {}
    skipped: list[dict[str, Any]] = []

    for mode in ["none", "plane", "line", "line_then_plane", "plane_then_line"]:
        candidate_arrays[mode], candidate_diagnostics[mode] = simple_candidate(
            matrix,
            mode,
            physical_context,
        )

    if background_fraction >= 0.15:
        (
            candidate_arrays["background_aware_basic"],
            candidate_diagnostics["background_aware_basic"],
        ) = background_aware_pass(
            matrix,
            mask=background,
            include_poly2=False,
            physical_context=physical_context,
        )
    else:
        skipped.append(
            {
                "mode": "background_aware_basic",
                "accepted": False,
                "rejection_reason": "background_fraction_below_0.15",
            }
        )

    if background_fraction >= 0.25:
        (
            candidate_arrays["background_aware_poly2"],
            candidate_diagnostics["background_aware_poly2"],
        ) = background_aware_pass(
            matrix,
            mask=background,
            include_poly2=True,
            physical_context=physical_context,
        )
    else:
        skipped.append(
            {
                "mode": "background_aware_poly2",
                "accepted": False,
                "rejection_reason": "background_fraction_below_0.25",
            }
        )

    reference_mode = (
        "line_then_plane" if "line_then_plane" in candidate_arrays else "plane"
    )
    reference_feature_amplitude = feature_amplitude(
        candidate_arrays[reference_mode],
        foreground,
        background,
    )
    candidate_scores: list[dict[str, Any]] = []
    for mode in candidate_order:
        if mode not in candidate_arrays:
            continue
        score = score_candidate(
            candidate_arrays[mode],
            mode,
            background,
            foreground,
            scale,
            reference_feature_amplitude,
            complexity_rank[mode],
            physical_context,
        )
        score["diagnostics"] = candidate_diagnostics[mode]
        candidate_scores.append(score)
    candidate_scores.extend(skipped)

    accepted_scores = [
        score
        for score in candidate_scores
        if score.get("accepted") and "score" in score
    ]
    if not accepted_scores:
        accepted_scores = [score for score in candidate_scores if "score" in score]
    best_score = min(float(score["score"]) for score in accepted_scores)
    tolerance = max(0.01 * float(scale), 0.03 * abs(best_score))
    near_best = [
        score
        for score in accepted_scores
        if float(score["score"]) <= best_score + tolerance
    ]
    selected = min(
        near_best,
        key=lambda score: (
            complexity_rank.get(str(score["mode"]), 999),
            float(score["score"]),
        ),
    )
    selected_mode = str(selected["mode"])
    for score in candidate_scores:
        score["selected"] = score.get("mode") == selected_mode

    diagnostics = {
        "mode": "auto",
        "description": (
            "Compared simple leveling and two-pass background-aware flattening "
            "candidates, then selected the simplest near-best accepted score."
        ),
        "selected_mode": selected_mode,
        "selection_reason": (
            "lowest accepted score within tolerance, with lower-complexity "
            "candidates preferred among near ties"
        ),
        "score_tolerance": float(tolerance),
        "robust_height_scale": float(scale),
        "first_pass": first_pass_diag,
        "masking": mask_diag,
        "candidate_scores": candidate_scores,
    }
    return candidate_arrays[selected_mode], diagnostics, foreground, background


def level_matrix(
    matrix: np.ndarray,
    mode: str,
    physical_context: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply ``none``, ``line``, ``plane``, or ``auto`` leveling."""

    matrix = np.asarray(matrix, dtype=float)
    finite = np.isfinite(matrix)
    if not np.any(finite):
        raise ValueError("Cannot level an all-NaN height matrix")
    if mode == "none":
        return matrix.copy(), {"mode": "none"}
    if mode == "line":
        row_medians = np.nanmedian(np.where(finite, matrix, np.nan), axis=1)
        return matrix - row_medians[:, None], {
            "mode": "line",
            "description": "Subtracted each row median from the height map.",
        }
    if mode == "plane":
        leveled, fit_diagnostics = subtract_surface(
            matrix,
            mask=finite,
            model="plane",
            physical_context=physical_context,
        )
        return leveled, {
            "mode": "plane",
            "plane_coefficients": fit_diagnostics.get("coefficients", {}),
            "coefficient_units": fit_diagnostics.get("coefficient_units", {}),
            "coordinate_context": fit_diagnostics.get("coordinate_context", {}),
            "surface_fit": fit_diagnostics,
        }
    if mode == "auto":
        leveled, diagnostics, _, _ = auto_level_matrix(matrix, physical_context)
        return leveled, diagnostics
    raise ValueError("leveling must be one of: plane, line, none, auto")


__all__ = ["auto_level_matrix", "level_matrix"]

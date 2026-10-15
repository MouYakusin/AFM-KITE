"""Pure array algorithms for value- and gradient-based segmentation."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import closing, disk, opening

from .morphology import remove_objects_smaller_than


def regions_from_label_image(label_image: np.ndarray, matrix: np.ndarray) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    for index, prop in enumerate(regionprops(label_image, intensity_image=matrix), start=1):
        min_row, min_col, max_row, max_col = prop.bbox
        values = matrix[label_image == prop.label]
        regions.append(
            {
                "region_id": f"region_{index:03d}",
                "label_value": int(prop.label),
                "bbox_xyxy": [int(min_col), int(min_row), int(max_col), int(max_row)],
                "centroid_yx": [float(prop.centroid[0]), float(prop.centroid[1])],
                "area_px": int(prop.area),
                "mean_height": float(np.mean(values)),
                "min_height": float(np.min(values)),
                "max_height": float(np.max(values)),
            }
        )
    return regions


def build_value_candidate_labels(
    matrix: np.ndarray,
    polarity: str,
    threshold: float,
    min_area_px: int,
) -> np.ndarray:
    if polarity == "bright":
        binary = matrix > threshold
    elif polarity == "dark":
        binary = matrix < threshold
    else:
        raise ValueError("polarity must be bright or dark")

    binary = opening(binary, disk(1))
    binary = closing(binary, disk(2))
    binary = ndi.binary_fill_holes(binary)
    binary = remove_objects_smaller_than(binary, min_area_px)
    return label(binary)


def _evaluate_value_labels(
    label_image: np.ndarray,
    expected_count: int,
) -> tuple[float, dict[str, Any]]:
    props = regionprops(label_image)
    count = len(props)
    coverage = float(np.mean(label_image > 0))
    score = 0.0
    diagnostics: dict[str, Any] = {
        "region_count": count,
        "coverage": coverage,
        "max_area_to_median_area": None,
        "largest_span_fraction": None,
    }

    if 1 <= count <= 80:
        score += 2.0
    else:
        score -= 4.0

    if 0.005 <= coverage <= 0.60:
        score += 2.0
    else:
        score -= 3.0

    if count > 0:
        areas = np.asarray([prop.area for prop in props], dtype=float)
        median_area = max(float(np.median(areas)), 1.0)
        max_to_median_area = float(np.max(areas)) / median_area
        if float(np.median(areas)) > 64:
            score += 1.0
        if max_to_median_area > 4.0:
            score -= min(4.0, (max_to_median_area - 4.0) * 0.75)
        if max_to_median_area > 25.0:
            score -= 1.0

        rows, cols = label_image.shape
        largest = props[int(np.argmax(areas))]
        min_row, min_col, max_row, max_col = largest.bbox
        largest_width_fraction = (max_col - min_col) / max(float(cols), 1.0)
        largest_height_fraction = (max_row - min_row) / max(float(rows), 1.0)
        largest_span_fraction = max(
            largest_width_fraction,
            largest_height_fraction,
        )
        diagnostics.update(
            {
                "median_area_px": float(np.median(areas)),
                "max_area_px": float(np.max(areas)),
                "max_area_to_median_area": max_to_median_area,
                "largest_bbox_yxyx": [
                    int(min_row),
                    int(min_col),
                    int(max_row),
                    int(max_col),
                ],
                "largest_span_fraction": float(largest_span_fraction),
            }
        )
        if (
            largest_span_fraction >= 0.80
            and max_to_median_area > 3.0
        ):
            score -= 2.0
        if largest_span_fraction >= 0.80 and coverage > 0.50:
            score -= 2.0

    if expected_count > 0:
        score -= abs(count - expected_count) * 0.35

    return score, diagnostics


def evaluate_value_segmentation_quality(
    score: float,
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate whether one value-region candidate is safe for measurement."""

    flags: list[str] = []
    region_count = int(diagnostics.get("region_count") or 0)
    coverage = float(diagnostics.get("coverage") or 0.0)
    largest_span_value = diagnostics.get("largest_span_fraction")
    largest_span = (
        float(largest_span_value)
        if largest_span_value is not None
        else None
    )

    if region_count <= 0:
        flags.append("no_regions")
    if float(score) < 0.0:
        flags.append("negative_candidate_score")
    if largest_span is not None and largest_span >= 0.80 and coverage > 0.50:
        flags.append("dominant_full_field_component")

    return {
        "passed": not flags,
        "flags": flags,
    }


def build_value_candidate_record(
    label_image: np.ndarray,
    *,
    method: str,
    polarity: str,
    threshold: float,
    expected_count: int,
    detrend_sigma_px: float | None = None,
) -> dict[str, Any]:
    score, diagnostics = _evaluate_value_labels(label_image, expected_count)
    return {
        "label_image": label_image,
        "method": method,
        "polarity": polarity,
        "threshold": threshold,
        "detrend_sigma_px": detrend_sigma_px,
        "score": score,
        "diagnostics": diagnostics,
        "quality": evaluate_value_segmentation_quality(score, diagnostics),
    }


def build_detrended_value_candidate(
    matrix: np.ndarray,
    polarity: str,
    min_area_px: int,
    sigma_px: float,
) -> tuple[np.ndarray, float]:
    finite = np.isfinite(matrix)
    if not np.any(finite):
        raise ValueError("Cannot segment an all-NaN height matrix")

    filled = np.where(finite, matrix, float(np.nanmedian(matrix[finite])))
    background = ndi.gaussian_filter(filled, sigma=float(sigma_px))
    residual = np.where(finite, matrix - background, np.nan)
    threshold = float(threshold_otsu(residual[finite]))
    return build_value_candidate_labels(residual, polarity, threshold, min_area_px), threshold


def _parse_gradient_sigmas(gradient_sigma: str | float | int,
                           shape: tuple[int, int]) -> list[float]:
    if isinstance(gradient_sigma, (int, float)):
        return [max(float(gradient_sigma), 0.0)]
    text = str(gradient_sigma or "auto").strip().lower()
    if text == "auto":
        short_side = min(int(shape[0]), int(shape[1]))
        if short_side < 96:
            return [1.0, 1.6]
        if short_side < 256:
            return [1.0, 1.8, 2.8]
        return [1.0, 2.0, 3.5]
    sigmas: list[float] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value < 0:
            raise ValueError("gradient sigma values must be non-negative")
        sigmas.append(value)
    if not sigmas:
        raise ValueError("gradient_sigma must be 'auto' or one or more numeric sigma values")
    return sigmas


def _combined_gradient(
    height: np.ndarray,
    finite: np.ndarray,
    gradient_sigmas: list[float],
) -> np.ndarray:
    fill_value = float(np.nanmedian(height[finite]))
    filled = np.where(finite, height, fill_value)
    normalized_gradients: list[np.ndarray] = []
    for sigma in gradient_sigmas:
        smoothed = ndi.gaussian_filter(filled, sigma=float(sigma)) if sigma > 0 else filled
        gy, gx = np.gradient(smoothed)
        gradient = np.hypot(gx, gy)
        values = gradient[finite & np.isfinite(gradient)]
        if values.size == 0:
            continue
        scale = float(np.percentile(values, 95.0))
        if not np.isfinite(scale) or scale <= 0:
            scale = float(np.max(values)) if values.size else 1.0
        normalized_gradients.append(gradient / max(scale, np.finfo(float).eps))
    if not normalized_gradients:
        raise ValueError("Gradient matrix contains no finite values")
    return np.max(np.stack(normalized_gradients, axis=0), axis=0)


def _prepare_boundary_candidate(
    raw_boundary: np.ndarray,
    finite: np.ndarray,
    boundary_width_px: int,
    min_region_area_px: int,
    interior_buffer_px: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boundary = np.asarray(raw_boundary, dtype=bool) & finite
    boundary = remove_objects_smaller_than(
        boundary,
        max(3, int(min_region_area_px) // 16),
    )
    if int(boundary_width_px) > 0:
        boundary = ndi.binary_dilation(boundary, iterations=max(int(boundary_width_px), 1)) & finite
    buffer_mask = np.zeros(boundary.shape, dtype=bool)
    if int(interior_buffer_px) > 0 and np.any(boundary):
        buffer_mask = (
            ndi.binary_dilation(boundary, iterations=max(int(interior_buffer_px), 1))
            & finite
            & ~boundary
        )
    clean_interior = finite & ~boundary & ~buffer_mask
    clean_interior = remove_objects_smaller_than(
        clean_interior,
        max(int(min_region_area_px), 1),
    )
    return boundary, buffer_mask, clean_interior


def _score_boundary_candidate(
    method: str,
    threshold: float | list[float],
    boundary: np.ndarray,
    buffer_mask: np.ndarray,
    clean_interior: np.ndarray,
    finite: np.ndarray,
    min_region_area_px: int,
    max_target_regions: int,
) -> dict[str, Any]:
    finite_count = max(int(np.sum(finite)), 1)
    boundary_fraction = float(np.sum(boundary & finite) / finite_count)
    buffer_fraction = float(np.sum(buffer_mask & finite) / finite_count)
    interior_fraction = float(np.sum(clean_interior & finite) / finite_count)
    interior_components = label(clean_interior)
    boundary_components = label(boundary)
    target_labels = _target_region_labels(
        clean_interior,
        min_region_area_px,
        max_target_regions,
    )
    target_count = int(np.max(target_labels))
    boundary_component_count = int(np.max(boundary_components))
    interior_component_count = int(np.max(interior_components))

    coverage_score = max(0.0, 1.0 - abs(boundary_fraction - 0.12) / 0.18)
    interior_score = min(interior_fraction / 0.55, 1.0)
    target_score = min(
        target_count / max(min(int(max_target_regions), 6), 1),
        1.0,
    )
    component_score = min(math.log1p(max(interior_component_count, 0)) / math.log1p(12.0), 1.0)
    fragmentation_penalty = min(
        boundary_component_count / max(float(np.sum(boundary)) / 18.0, 1.0),
        4.0,
    ) * 0.04
    score = (
        0.35 * coverage_score
        + 0.25 * interior_score
        + 0.25 * target_score
        + 0.15 * component_score
        - fragmentation_penalty
    )
    if boundary_fraction < 0.005 or boundary_fraction > 0.45:
        score -= 1.0
    if interior_fraction < 0.08:
        score -= 0.7
    if target_count == 0:
        score -= 0.5
    return {
        "method": method,
        "threshold": threshold,
        "score": float(score),
        "boundary_pixel_fraction": boundary_fraction,
        "boundary_buffer_pixel_fraction": buffer_fraction,
        "clean_interior_pixel_fraction": interior_fraction,
        "boundary_component_count": boundary_component_count,
        "interior_component_count": interior_component_count,
        "target_region_count": target_count,
        "target_labels": target_labels,
    }


def _candidate_boundary_masks(
    gradient: np.ndarray,
    finite: np.ndarray,
    boundary_percentile: float,
    segmentation_method: str,
) -> list[tuple[str, float | list[float], np.ndarray]]:
    values = gradient[finite & np.isfinite(gradient)]
    if values.size == 0:
        raise ValueError("Gradient matrix contains no finite values")
    requested = str(segmentation_method or "auto").lower()
    if requested not in {"auto", "percentile", "otsu", "hysteresis"}:
        raise ValueError("segmentation_method must be auto, percentile, otsu, or hysteresis")

    candidates: list[tuple[str, float | list[float], np.ndarray]] = []
    pct_threshold = float(np.percentile(values, float(boundary_percentile)))
    if requested in {"auto", "percentile"}:
        candidates.append(("percentile", pct_threshold, (gradient >= pct_threshold) & finite))

    if requested in {"auto", "otsu"} and threshold_otsu is not None:
        otsu_threshold = float(threshold_otsu(values))
        candidates.append(("otsu", otsu_threshold, (gradient >= otsu_threshold) & finite))

    if requested in {"auto", "hysteresis"}:
        high = pct_threshold
        low_percentile = max(50.0, float(boundary_percentile) - 12.0)
        low = float(np.percentile(values, low_percentile))
        high_mask = (gradient >= high) & finite
        low_mask = (gradient >= low) & finite
        hysteresis = ndi.binary_propagation(high_mask, mask=low_mask) & finite
        candidates.append(("hysteresis", [low, high], hysteresis))

    if not candidates:
        candidates.append(("percentile", pct_threshold, (gradient >= pct_threshold) & finite))
    return candidates


def segment_gradient_masks(
    height: np.ndarray,
    boundary_percentile: float,
    boundary_width_px: int,
    min_region_area_px: int,
    max_target_regions: int,
    segmentation_method: str,
    gradient_sigma: str | float | int,
    interior_buffer_px: int,
) -> dict[str, Any]:
    finite = np.isfinite(height)
    if not np.any(finite):
        raise ValueError("Height/reference matrix contains no finite values")
    gradient_sigmas = _parse_gradient_sigmas(gradient_sigma, height.shape)
    gradient = _combined_gradient(height, finite, gradient_sigmas)

    evaluated: list[dict[str, Any]] = []
    for method, threshold, raw_boundary in _candidate_boundary_masks(
        gradient,
        finite,
        boundary_percentile,
        segmentation_method,
    ):
        boundary, buffer_mask, clean_interior = _prepare_boundary_candidate(
            raw_boundary,
            finite,
            boundary_width_px,
            min_region_area_px,
            interior_buffer_px,
        )
        diagnostics = _score_boundary_candidate(
            method,
            threshold,
            boundary,
            buffer_mask,
            clean_interior,
            finite,
            min_region_area_px,
            max_target_regions,
        )
        diagnostics.update(
            {
                "boundary": boundary,
                "boundary_buffer": buffer_mask,
                "clean_interior": clean_interior,
            }
        )
        evaluated.append(diagnostics)

    selected = max(evaluated, key=lambda item: float(item["score"]))
    candidate_diagnostics = [
        {
            key: value
            for key, value in item.items()
            if key not in {"boundary", "boundary_buffer", "clean_interior", "target_labels"}
        }
        for item in evaluated
    ]
    return {
        "gradient": gradient,
        "boundary": selected["boundary"],
        "boundary_buffer": selected["boundary_buffer"],
        "clean_interior": selected["clean_interior"],
        "target_labels": selected["target_labels"],
        "threshold": selected["threshold"],
        "selected_method": selected["method"],
        "selected_score": float(selected["score"]),
        "candidate_diagnostics": candidate_diagnostics,
        "gradient_sigmas": gradient_sigmas,
    }


def _target_region_labels(
    interior: np.ndarray,
    min_region_area_px: int,
    max_target_regions: int,
) -> np.ndarray:
    distance = ndi.distance_transform_edt(interior)
    if not np.any(distance > 0):
        return np.zeros(interior.shape, dtype=np.int32)
    components = label(interior)
    rows, cols = interior.shape
    candidates: list[tuple[float, int]] = []
    for prop in regionprops(components, intensity_image=distance):
        if int(prop.area) < max(8, int(min_region_area_px) // 2):
            continue
        min_row, min_col, max_row, max_col = prop.bbox
        edge_touched = min_row == 0 or min_col == 0 or max_row == rows or max_col == cols
        distance_values = distance[components == prop.label]
        max_distance = float(np.max(distance_values)) if distance_values.size else 0.0
        mean_distance = float(np.mean(distance_values)) if distance_values.size else 0.0
        edge_penalty = 0.55 if edge_touched else 1.0
        score = edge_penalty * math.sqrt(float(prop.area)) * \
            (0.7 * max_distance + 0.3 * mean_distance)
        candidates.append((score, int(prop.label)))

    if not candidates:
        return np.zeros(interior.shape, dtype=np.int32)
    candidates.sort(reverse=True)
    labels = np.zeros(interior.shape, dtype=np.int32)
    for index, (_, component_label) in enumerate(
            candidates[: max(int(max_target_regions), 1)], start=1):
        mask = components == component_label
        labels[mask] = index
    return labels


def describe_target_regions(
    target_labels: np.ndarray,
    height: np.ndarray,
) -> list[dict[str, Any]]:
    rows, cols = target_labels.shape
    records: list[dict[str, Any]] = []
    for index, prop in enumerate(regionprops(target_labels), start=1):
        mask = target_labels == prop.label
        min_row, min_col, max_row, max_col = prop.bbox
        edge_touched = min_row == 0 or min_col == 0 or max_row == rows or max_col == cols
        height_values = height[mask]
        records.append(
            {
                "feature_id": f"feature_{index:03d}",
                "label_value": int(prop.label),
                "bbox_xyxy": [int(min_col), int(min_row), int(max_col), int(max_row)],
                "centroid_yx": [float(prop.centroid[0]), float(prop.centroid[1])],
                "area_px": int(prop.area),
                "edge_touched": bool(edge_touched),
                "height_median": float(np.nanmedian(height_values)),
                "is_target": True,
            }
        )
    return records


__all__ = [
    "build_detrended_value_candidate",
    "build_value_candidate_labels",
    "build_value_candidate_record",
    "describe_target_regions",
    "evaluate_value_segmentation_quality",
    "regions_from_label_image",
    "segment_gradient_masks",
]

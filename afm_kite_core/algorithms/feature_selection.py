"""Deterministic ranking helpers for labeled feature candidates."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import ndimage as ndi

SELECTION_METRICS = frozenset(
    {
        "area_px",
        "mean_value",
        "median_value",
        "median_prominence",
        "absolute_prominence",
    }
)
SORT_ORDERS = frozenset({"ascending", "descending"})
POLARITIES = frozenset({"any", "raised", "depressed"})


def rank_feature_candidates(
    candidates: list[dict[str, Any]],
    *,
    selection_metric: str,
    sort_order: str,
    rank: int,
    polarity: str,
    min_area_px: int = 0,
    max_area_px: int = 0,
    min_edge_margin_px: int = 0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Filter and deterministically rank generic feature candidate records."""

    if selection_metric not in SELECTION_METRICS:
        raise ValueError(
            "selection_metric must be one of " + ", ".join(sorted(SELECTION_METRICS))
        )
    if sort_order not in SORT_ORDERS:
        raise ValueError("sort_order must be ascending or descending")
    if polarity not in POLARITIES:
        raise ValueError("polarity must be any, raised, or depressed")
    if int(rank) < 1:
        raise ValueError("rank must be a one-based positive integer")
    if int(min_area_px) < 0 or int(max_area_px) < 0:
        raise ValueError("area limits must be non-negative")
    if int(max_area_px) and int(max_area_px) < int(min_area_px):
        raise ValueError("max_area_px must be zero or at least min_area_px")
    if int(min_edge_margin_px) < 0:
        raise ValueError("min_edge_margin_px must be non-negative")

    prepared: list[dict[str, Any]] = []
    for source in candidates:
        candidate = dict(source)
        prominence = float(candidate.get("median_prominence", 0.0))
        candidate["absolute_prominence"] = abs(prominence)
        reasons: list[str] = []
        area_px = int(candidate.get("area_px", 0))
        edge_margin_px = int(candidate.get("edge_margin_px", 0))
        if area_px < int(min_area_px):
            reasons.append("area_below_minimum")
        if int(max_area_px) and area_px > int(max_area_px):
            reasons.append("area_above_maximum")
        if edge_margin_px < int(min_edge_margin_px):
            reasons.append("edge_margin_below_minimum")
        if polarity == "raised" and prominence <= 0:
            reasons.append("not_raised")
        if polarity == "depressed" and prominence >= 0:
            reasons.append("not_depressed")
        metric_value = candidate.get(selection_metric)
        if metric_value is None:
            reasons.append("selection_metric_unavailable")
        candidate["selection_metric"] = selection_metric
        candidate["selection_metric_value"] = (
            float(metric_value) if metric_value is not None else None
        )
        candidate["eligible"] = not reasons
        candidate["rejection_reasons"] = reasons
        candidate["rank"] = None
        prepared.append(candidate)

    eligible = [candidate for candidate in prepared if candidate["eligible"]]
    if sort_order == "descending":
        eligible.sort(
            key=lambda item: (
                -float(item["selection_metric_value"]),
                -int(item["area_px"]),
                int(item["label_value"]),
            )
        )
    else:
        eligible.sort(
            key=lambda item: (
                float(item["selection_metric_value"]),
                -int(item["area_px"]),
                int(item["label_value"]),
            )
        )

    for index, candidate in enumerate(eligible, start=1):
        candidate["rank"] = index

    if int(rank) > len(eligible):
        raise ValueError(
            f"Requested rank {rank}, but only {len(eligible)} eligible features remain"
        )
    selected = eligible[int(rank) - 1]
    rejected = sorted(
        (candidate for candidate in prepared if not candidate["eligible"]),
        key=lambda item: int(item["label_value"]),
    )
    return selected, [*eligible, *rejected]


def feature_candidate_record(
    labels: np.ndarray,
    matrix: np.ndarray,
    label_value: int,
    *,
    global_background_median: float,
    global_background_sample_count: int,
    prominence_reference: str,
    local_ring_inner_px: int,
    local_ring_outer_px: int,
    min_local_background_samples: int,
) -> dict[str, Any]:
    """Measure one labeled feature against local or global background samples."""

    mask = labels == label_value
    values = matrix[mask & np.isfinite(matrix)]
    if values.size == 0:
        median_value = None
        mean_value = None
        prominence = 0.0
    else:
        median_value = float(np.median(values))
        mean_value = float(np.mean(values))

    coords = np.argwhere(mask)
    rows, cols = labels.shape
    min_row, min_col = np.min(coords, axis=0)
    max_row, max_col = np.max(coords, axis=0) + 1
    padding = int(local_ring_outer_px) + 1
    crop_min_row = max(0, int(min_row) - padding)
    crop_min_col = max(0, int(min_col) - padding)
    crop_max_row = min(rows, int(max_row) + padding)
    crop_max_col = min(cols, int(max_col) + padding)
    crop_labels = labels[crop_min_row:crop_max_row, crop_min_col:crop_max_col]
    crop_matrix = matrix[crop_min_row:crop_max_row, crop_min_col:crop_max_col]
    crop_mask = crop_labels == int(label_value)
    distance_from_feature = ndi.distance_transform_edt(~crop_mask)
    local_ring = (
        (distance_from_feature > float(local_ring_inner_px))
        & (distance_from_feature <= float(local_ring_outer_px))
        & (crop_labels == 0)
        & np.isfinite(crop_matrix)
    )
    local_background_values = crop_matrix[local_ring]
    local_background_sample_count = int(local_background_values.size)
    local_background_median = (
        float(np.median(local_background_values))
        if local_background_sample_count
        else None
    )
    local_background_mad = None
    if local_background_median is not None:
        local_background_mad = float(
            np.median(np.abs(local_background_values - local_background_median))
        )

    if (
        prominence_reference == "local_ring"
        and local_background_sample_count >= int(min_local_background_samples)
        and local_background_median is not None
    ):
        background_median = local_background_median
        background_reference_used = "local_ring"
        background_sample_count = local_background_sample_count
        background_fallback_reason = ""
    else:
        background_median = float(global_background_median)
        background_reference_used = (
            "global_background"
            if prominence_reference == "global_background"
            else "global_background_fallback"
        )
        background_sample_count = int(global_background_sample_count)
        background_fallback_reason = (
            ""
            if prominence_reference == "global_background"
            else (
                "local_background_sample_count_below_minimum: "
                f"{local_background_sample_count} < {int(min_local_background_samples)}"
            )
        )

    if median_value is not None:
        prominence = float(median_value - background_median)
    edge_margin = int(min(min_row, min_col, rows - max_row, cols - max_col))
    area_px = int(mask.sum())
    return {
        "label_value": int(label_value),
        "area_px": area_px,
        "median_value": median_value,
        "mean_value": mean_value,
        "background_median": background_median,
        "background_reference_requested": prominence_reference,
        "background_reference_used": background_reference_used,
        "background_sample_count": background_sample_count,
        "background_fallback_reason": background_fallback_reason,
        "global_background_median": float(global_background_median),
        "global_background_sample_count": int(global_background_sample_count),
        "local_background_median": local_background_median,
        "local_background_sample_count": local_background_sample_count,
        "local_background_mad": local_background_mad,
        "local_ring_inner_px": int(local_ring_inner_px),
        "local_ring_outer_px": int(local_ring_outer_px),
        "median_prominence": float(prominence),
        "edge_margin_px": edge_margin,
    }


def region_axis(mask: np.ndarray) -> tuple[float, float]:
    """Return the normalized major-axis direction of a binary region."""

    coords_yx = np.argwhere(mask)
    if coords_yx.shape[0] < 3:
        return 1.0, 0.0
    coords_xy = coords_yx[:, ::-1].astype(float)
    coords_xy -= np.mean(coords_xy, axis=0)
    cov = np.cov(coords_xy, rowvar=False)
    values, vectors = np.linalg.eigh(cov)
    vector = vectors[:, int(np.argmax(values))]
    dx, dy = float(vector[0]), float(vector[1])
    length = math.hypot(dx, dy)
    if length == 0:
        return 1.0, 0.0
    if dx < 0:
        dx, dy = -dx, -dy
    return dx / length, dy / length


def line_direction(
    selected_mask: np.ndarray,
    direction_mode: str,
    angle_degrees: float | None,
) -> tuple[float, float]:
    """Resolve a requested feature-crossing direction from a selected mask."""

    major_dx, major_dy = region_axis(selected_mask)
    if direction_mode == "major_axis":
        return major_dx, major_dy
    if direction_mode == "minor_axis":
        return -major_dy, major_dx
    if direction_mode == "x":
        return 1.0, 0.0
    if direction_mode == "y":
        return 0.0, 1.0
    if direction_mode == "angle":
        if angle_degrees is None:
            raise ValueError("angle_degrees is required when direction_mode=angle")
        angle = math.radians(float(angle_degrees))
        return math.cos(angle), math.sin(angle)
    raise ValueError(
        "direction_mode must be minor_axis, major_axis, x, y, or angle"
    )


def rotate_direction(dx: float, dy: float, offset_degrees: float) -> tuple[float, float]:
    """Rotate and normalize a two-dimensional direction vector."""

    angle = math.radians(float(offset_degrees))
    rotated_dx = (dx * math.cos(angle)) - (dy * math.sin(angle))
    rotated_dy = (dx * math.sin(angle)) + (dy * math.cos(angle))
    length = math.hypot(rotated_dx, rotated_dy)
    return rotated_dx / length, rotated_dy / length


def line_boundary_t(
    center_x: float,
    center_y: float,
    dx: float,
    dy: float,
    shape: tuple[int, int],
) -> tuple[float, float]:
    """Return parametric intersections of a line with an image boundary."""

    rows, cols = shape
    candidates: list[float] = []
    if abs(dx) > 1e-9:
        for x_edge in (0.0, float(cols - 1)):
            t = (x_edge - center_x) / dx
            y = center_y + t * dy
            if 0.0 <= y <= rows - 1:
                candidates.append(t)
    if abs(dy) > 1e-9:
        for y_edge in (0.0, float(rows - 1)):
            t = (y_edge - center_y) / dy
            x = center_x + t * dx
            if 0.0 <= x <= cols - 1:
                candidates.append(t)
    if len(candidates) < 2:
        raise ValueError("Could not extend selected line to image boundaries")
    return min(candidates), max(candidates)


def sample_nearest(mask: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Sample a mask or label image at nearest integer coordinates."""

    rows, cols = mask.shape
    xi = np.clip(np.rint(xs).astype(int), 0, cols - 1)
    yi = np.clip(np.rint(ys).astype(int), 0, rows - 1)
    return mask[yi, xi]


def line_windows(
    selected_mask: np.ndarray,
    region_labels: np.ndarray,
    selected_label: int,
    center_x: float,
    center_y: float,
    dx: float,
    dy: float,
    background_margin_px: int,
) -> dict[str, Any]:
    """Build and validate substrate-feature-substrate index windows."""

    t_min, t_max = line_boundary_t(center_x, center_y, dx, dy, selected_mask.shape)
    t_values = np.arange(math.ceil(t_min), math.floor(t_max) + 1, dtype=float)
    if t_values.size < 8:
        raise ValueError("Selected line is too short")
    xs = center_x + t_values * dx
    ys = center_y + t_values * dy
    on_feature = sample_nearest(selected_mask, xs, ys)
    true_indices = np.flatnonzero(on_feature)
    if true_indices.size == 0:
        raise ValueError("Selected line does not cross the selected feature mask")

    center_index = int(np.argmin(np.abs(t_values)))
    feature_seed = int(true_indices[np.argmin(np.abs(true_indices - center_index))])
    start = feature_seed
    while start > 0 and on_feature[start - 1]:
        start -= 1
    end = feature_seed
    while end + 1 < on_feature.size and on_feature[end + 1]:
        end += 1

    margin = max(int(background_margin_px), 6)
    line_start = max(0, start - margin)
    line_end = min(on_feature.size - 1, end + margin)
    gap = 2
    left = np.arange(line_start, max(line_start, start - gap), dtype=int)
    right = np.arange(min(line_end + 1, end + gap + 1), line_end + 1, dtype=int)
    feature = np.arange(start, end + 1, dtype=int)

    if left.size < 3 or right.size < 3 or feature.size < 3:
        raise ValueError("Line does not contain enough substrate and feature samples")

    used = np.arange(line_start, line_end + 1, dtype=int)
    sampled_region_labels = sample_nearest(region_labels, xs, ys).astype(int)
    used_region_labels = sampled_region_labels[used]
    left_local = left - line_start
    right_local = right - line_start
    feature_local = feature - line_start
    if int(used_region_labels[0]) != 0 or int(used_region_labels[-1]) != 0:
        raise ValueError("Line start and end must both lie on label-0 substrate")
    if np.any(used_region_labels[left_local] != 0) or np.any(
        used_region_labels[right_local] != 0
    ):
        raise ValueError("Both substrate windows must contain only label-0 samples")
    allowed = (used_region_labels == 0) | (used_region_labels == int(selected_label))
    if not np.all(allowed):
        raise ValueError("Line crosses another segmented feature")
    if not np.all(used_region_labels[feature_local] == int(selected_label)):
        raise ValueError("Feature window does not stay inside the selected label")
    return {
        "t_values": t_values[used],
        "x_px": xs[used],
        "y_px": ys[used],
        "indices": used,
        "left_indices": left_local,
        "right_indices": right_local,
        "feature_indices": feature_local,
        "line_start_index": int(line_start),
        "line_end_index": int(line_end),
        "feature_start_index": int(start - line_start),
        "feature_end_index": int(end - line_start),
        "validation": {
            "start_on_substrate": True,
            "end_on_substrate": True,
            "substrate_windows_label_zero": True,
            "crosses_selected_feature": True,
            "crosses_other_features": False,
        },
    }


def segment_labels(
    sample_count: int,
    left: np.ndarray,
    right: np.ndarray,
    feature: np.ndarray,
) -> np.ndarray:
    """Label profile samples by substrate, transition, and feature windows."""

    labels = np.full(sample_count, "transition", dtype=object)
    labels[left] = "substrate_left"
    labels[right] = "substrate_right"
    labels[feature] = "feature"
    return labels


__all__ = [
    "POLARITIES",
    "SELECTION_METRICS",
    "SORT_ORDERS",
    "feature_candidate_record",
    "line_direction",
    "line_windows",
    "rank_feature_candidates",
    "rotate_direction",
    "segment_labels",
]

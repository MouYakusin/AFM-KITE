"""Pure local-outlier component detection for AFM matrices."""

from __future__ import annotations

from typing import Any

import numpy as np
from skimage.measure import label, regionprops

from .morphology import remove_objects_smaller_than


def detect_local_outlier_components(
    matrix: np.ndarray,
    *,
    scope: str,
    regions: list[dict[str, Any]],
    label_image: np.ndarray | None,
    polarity: str,
    robust_sigma_threshold: float,
    min_area_px: int,
    min_height_contrast: float,
    min_global_range_fraction: float,
    pixel_area: float | None,
    pixel_width: float | None,
    pixel_height: float | None,
    height_unit: str,
    area_unit: str,
) -> dict[str, Any]:
    """Detect robust local outlier components without reading or writing files."""

    finite = matrix[np.isfinite(matrix)]
    global_range = float(np.max(finite) - np.min(finite)) if finite.size else 0.0
    dynamic_min_contrast = max(
        float(min_height_contrast),
        global_range * max(float(min_global_range_fraction), 0.0),
    )

    contaminant_label = np.zeros(matrix.shape, dtype=np.int32)
    contaminants: list[dict[str, Any]] = []
    contaminant_index = 1
    analysis_regions: list[dict[str, Any]] = []

    if scope == "full":
        analysis_regions.append(
            {
                "region_id": "full_image",
                "label_value": None,
                "analysis_scope": "full",
                "mask": np.isfinite(matrix),
            }
        )
    else:
        assert label_image is not None
        if scope in {"regions", "all"}:
            for region in regions:
                label_value = int(region["label_value"])
                analysis_regions.append(
                    {
                        "region_id": region["region_id"],
                        "label_value": label_value,
                        "analysis_scope": "regions",
                        "mask": label_image == label_value,
                    }
                )
        if scope in {"background", "all"}:
            analysis_regions.append(
                {
                    "region_id": "background",
                    "label_value": 0,
                    "analysis_scope": "background",
                    "mask": label_image == 0,
                }
            )

    analysis_region_summaries: list[dict[str, Any]] = []
    for analysis_region in analysis_regions:
        region_id = analysis_region["region_id"]
        label_value = analysis_region["label_value"]
        analysis_scope = analysis_region["analysis_scope"]
        region_mask = np.asarray(analysis_region["mask"], dtype=bool)
        values = matrix[region_mask]
        values = values[np.isfinite(values)]
        analysis_region_summaries.append(
            {
                "region_id": region_id,
                "region_label_value": label_value,
                "analysis_scope": analysis_scope,
                "area_px": int(np.sum(region_mask)),
                "finite_px": int(values.size),
            }
        )
        if values.size < max(int(min_area_px), 1):
            continue

        median_height = float(np.median(values))
        mad = float(np.median(np.abs(values - median_height)))
        robust_sigma = float(1.4826 * mad)
        if robust_sigma <= 1e-12:
            robust_sigma = float(np.std(values))
        if robust_sigma <= 1e-12:
            robust_sigma = max(float(np.ptp(values)) / 6.0, 1e-12)

        sigma_contrast = float(robust_sigma_threshold) * robust_sigma
        height_contrast_threshold = max(sigma_contrast, dynamic_min_contrast)
        candidate_specs: list[tuple[str, np.ndarray]] = []
        if polarity in {"bright", "both"}:
            candidate_specs.append(
                (
                    "bright",
                    region_mask & (matrix >= median_height + height_contrast_threshold),
                )
            )
        if polarity in {"dark", "both"}:
            candidate_specs.append(
                (
                    "dark",
                    region_mask & (matrix <= median_height - height_contrast_threshold),
                )
            )

        for contaminant_type, candidate_mask in candidate_specs:
            candidate_mask = remove_objects_smaller_than(
                candidate_mask.astype(bool),
                min_area_px,
            )
            component_labels = label(candidate_mask)
            for prop in regionprops(component_labels, intensity_image=matrix):
                component_mask = component_labels == prop.label
                component_values = matrix[component_mask]
                if component_values.size == 0:
                    continue

                min_row, min_col, max_row, max_col = prop.bbox
                centroid_y, centroid_x = prop.centroid_weighted
                if contaminant_type == "bright":
                    extremum_height = float(np.max(component_values))
                    height_contrast = extremum_height - median_height
                else:
                    extremum_height = float(np.min(component_values))
                    height_contrast = median_height - extremum_height
                max_robust_z = float(height_contrast / robust_sigma) if robust_sigma else None
                area_px = int(prop.area)

                record: dict[str, Any] = {
                    "contaminant_id": f"contaminant_{contaminant_index:03d}",
                    "region_id": region_id,
                    "region_label_value": label_value,
                    "analysis_scope": analysis_scope,
                    "type": contaminant_type,
                    "bbox_xyxy": [int(min_col), int(min_row), int(max_col), int(max_row)],
                    "centroid_yx": [float(centroid_y), float(centroid_x)],
                    "area_px": area_px,
                    "area_physical": float(area_px * pixel_area) if pixel_area else None,
                    "area_unit": area_unit,
                    "region_median_height": median_height,
                    "height_extremum": extremum_height,
                    "height_contrast": float(height_contrast),
                    "height_unit": height_unit,
                    "robust_sigma": robust_sigma,
                    "height_contrast_threshold": height_contrast_threshold,
                    "max_robust_z": max_robust_z,
                }
                if pixel_width and pixel_height:
                    record["centroid_xy_physical"] = [
                        float(centroid_x * pixel_width),
                        float(centroid_y * pixel_height),
                    ]

                contaminants.append(record)
                contaminant_label[component_mask] = contaminant_index
                contaminant_index += 1

    return {
        "global_range": global_range,
        "dynamic_min_contrast": dynamic_min_contrast,
        "label_image": contaminant_label,
        "contaminants": contaminants,
        "analysis_region_count": len(analysis_regions),
        "analysis_regions": analysis_region_summaries,
    }


__all__ = ["detect_local_outlier_components"]

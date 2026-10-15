"""Public region-segmentation and local-outlier tools."""

from __future__ import annotations

from typing import Optional

from afm_kite_core.contracts import (
    GradientSegmentationMethod,
    OutlierPolarity,
    OutlierScope,
    RegionPolarity,
)
from afm_kite_core.tools.boundary import JsonDict, invoke_operation


def value_regions_segment(
    matrix_data_path: str,
    output_dir: str = "outputs/data/value_regions",
    polarity: RegionPolarity = "auto",
    min_area_fraction: float = 0.002,
    min_area_px: int = 0,
    threshold_percentile: Optional[float] = None,
    expected_count: int = 0,
    metadata_path: str = "",
    write_plots: bool = False,
) -> JsonDict:
    """Segment connected regions and optionally render qualitative review plots."""
    from afm_kite_core.operations.segmentation.value_and_outliers import (
        segment_value_regions,
    )

    return invoke_operation(
        "value-regions-segment",
        segment_value_regions,
        matrix_data_path=matrix_data_path,
        output_dir=output_dir,
        polarity=polarity,
        min_area_fraction=min_area_fraction,
        expected_count=expected_count,
        write_plots=write_plots,
        threshold_percentile=threshold_percentile,
        min_area_px=min_area_px,
        metadata_path=metadata_path,
    )


def gradient_regions_segment(
    matrix_data_path: str,
    output_dir: str = "outputs/data/gradient_regions",
    boundary_percentile: float = 88.0,
    boundary_width_px: int = 1,
    min_region_area_px: int = 64,
    max_target_regions: int = 20,
    source_manifest_path: str = "",
    segmentation_method: GradientSegmentationMethod = "auto",
    gradient_sigma: str = "auto",
    interior_buffer_px: int = 2,
) -> JsonDict:
    """Segment edge-like and interior regions from matrix gradients."""
    from afm_kite_core.operations.segmentation.gradient import segment_gradient_regions

    return invoke_operation(
        "gradient-regions-segment",
        segment_gradient_regions,
        matrix_data_path=matrix_data_path,
        output_dir=output_dir,
        boundary_percentile=boundary_percentile,
        boundary_width_px=boundary_width_px,
        min_region_area_px=min_region_area_px,
        max_target_regions=max_target_regions,
        source_manifest_path=source_manifest_path,
        segmentation_method=segmentation_method,
        gradient_sigma=gradient_sigma,
        interior_buffer_px=interior_buffer_px,
    )


def local_outliers_detect(
    matrix_data_path: str,
    regions_json_path: str = "",
    output_dir: str = "outputs/data/local_outliers",
    metadata_path: str = "",
    polarity: OutlierPolarity = "bright",
    scope: OutlierScope = "regions",
    robust_sigma_threshold: float = 6.0,
    min_area_px: int = 3,
    min_value_contrast: float = 0.0,
    min_global_range_fraction: float = 0.05,
) -> JsonDict:
    """Detect local outlier components in a matrix or supplied regions."""
    from afm_kite_core.operations.segmentation.value_and_outliers import (
        detect_local_outliers,
    )

    return invoke_operation(
        "local-outliers-detect",
        detect_local_outliers,
        height_data_path=matrix_data_path,
        regions_json_path=regions_json_path,
        output_dir=output_dir,
        metadata_path=metadata_path,
        polarity=polarity,
        robust_sigma_threshold=robust_sigma_threshold,
        min_area_px=min_area_px,
        min_height_contrast=min_value_contrast,
        min_global_range_fraction=min_global_range_fraction,
        scope=scope,
    )


__all__ = [
    "gradient_regions_segment",
    "local_outliers_detect",
    "value_regions_segment",
]

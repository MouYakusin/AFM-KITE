"""Public transform-aware paired-field and paired-region tools."""

from __future__ import annotations

from typing import Optional

from afm_kite_core.contracts import (
    InterpolationMode,
    PairedRegionMaskMode,
    PairedRegionSelectionMetric,
    SortOrder,
)
from afm_kite_core.tools.boundary import JsonDict, invoke_operation


def paired_region_catalog(
    data_group_json_path: str,
    regions_json_path: str,
    target_channel: str = "DCbias",
    mask_mode: PairedRegionMaskMode = "eroded",
    erosion_px: int = 2,
    interpolation: InterpolationMode = "linear",
    min_coverage: float = 0.95,
    min_region_pixels: int = 20,
    max_shift_fraction: float = 0.1,
    render_change_atlas: bool = False,
    output_dir: str = "outputs/data/paired_region_catalog",
) -> JsonDict:
    """Build a full-field difference and per-region catalog for an ordered pair."""
    from afm_kite_core.operations.paired_regions import build_paired_region_catalog

    return invoke_operation(
        "paired-region-catalog",
        build_paired_region_catalog,
        data_group_json_path=data_group_json_path,
        regions_json_path=regions_json_path,
        output_dir=output_dir,
        target_channel=target_channel,
        mask_mode=mask_mode,
        erosion_px=erosion_px,
        interpolation=interpolation,
        min_coverage=min_coverage,
        min_region_pixels=min_region_pixels,
        max_shift_fraction=max_shift_fraction,
        render_change_atlas=render_change_atlas,
    )


def paired_region_select(
    paired_region_catalog_json_path: str,
    feature_id: str = "",
    x_px: Optional[float] = None,
    y_px: Optional[float] = None,
    selection_metric: PairedRegionSelectionMetric = "absolute_change",
    sort_order: SortOrder = "descending",
    rank: int = 1,
    min_area_px: int = 0,
    min_coverage: float = 0.95,
    render_selected_overlay: bool = False,
    output_dir: str = "outputs/data/paired_region_selection",
) -> JsonDict:
    """Select one cataloged region by ID, coordinate, or ranked change metric."""
    from afm_kite_core.operations.paired_regions import select_paired_region

    return invoke_operation(
        "paired-region-select",
        select_paired_region,
        paired_region_catalog_json_path=paired_region_catalog_json_path,
        output_dir=output_dir,
        feature_id=feature_id,
        x_px=x_px,
        y_px=y_px,
        selection_metric=selection_metric,
        sort_order=sort_order,
        rank=rank,
        min_area_px=min_area_px,
        min_coverage=min_coverage,
        render_selected_overlay=render_selected_overlay,
    )


def paired_region_measure(
    data_group_json_path: str,
    selected_feature_json_path: str,
    target_channel: str = "DCbias",
    mask_mode: PairedRegionMaskMode = "eroded",
    erosion_px: int = 2,
    interpolation: InterpolationMode = "linear",
    min_coverage: float = 0.95,
    min_region_pixels: int = 20,
    max_shift_fraction: float = 0.1,
    calculate_background_normalized: bool = False,
    background_ring_inner_px: int = 3,
    background_ring_outer_px: int = 15,
    write_difference_matrix: bool = False,
    output_dir: str = "outputs/data/paired_region",
) -> JsonDict:
    """Measure one shared region across an ordered registered condition pair."""
    from afm_kite_core.operations.paired_regions import measure_paired_region

    return invoke_operation(
        "paired-region-measure",
        measure_paired_region,
        data_group_json_path=data_group_json_path,
        selected_feature_json_path=selected_feature_json_path,
        output_dir=output_dir,
        target_channel=target_channel,
        mask_mode=mask_mode,
        erosion_px=erosion_px,
        interpolation=interpolation,
        min_coverage=min_coverage,
        min_region_pixels=min_region_pixels,
        max_shift_fraction=max_shift_fraction,
        calculate_background_normalized=calculate_background_normalized,
        background_ring_inner_px=background_ring_inner_px,
        background_ring_outer_px=background_ring_outer_px,
        write_difference_matrix=write_difference_matrix,
    )


__all__ = [
    "paired_region_catalog",
    "paired_region_measure",
    "paired_region_select",
]

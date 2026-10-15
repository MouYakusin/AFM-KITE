"""Public generic feature-selection and feature-line construction tools."""

from __future__ import annotations

from typing import Optional

from afm_kite_core.contracts import (
    FeaturePolarity,
    FeatureSelectionMetric,
    LineDirectionMode,
    ProminenceReference,
    SortOrder,
)
from afm_kite_core.tools.boundary import JsonDict, invoke_operation


def feature_select(
    regions_json_path: str,
    matrix_data_path: str,
    output_dir: str = "outputs/data/feature_select",
    selection_metric: FeatureSelectionMetric = "median_prominence",
    sort_order: SortOrder = "descending",
    rank: int = 1,
    polarity: FeaturePolarity = "any",
    min_area_px: int = 0,
    max_area_px: int = 0,
    min_edge_margin_px: int = 0,
    prominence_reference: ProminenceReference = "local_ring",
    local_ring_inner_px: int = 3,
    local_ring_outer_px: int = 12,
    min_local_background_samples: int = 20,
    render_candidates_overlay: bool = False,
    render_selected_overlay: bool = False,
    colormap: str = "afm_gold_sampled",
) -> JsonDict:
    """Select the requested ranked feature from existing labeled regions."""
    from afm_kite_core.operations.features.selection import feature_select as _feature_select

    return invoke_operation(
        "feature-select",
        _feature_select,
        regions_json_path=regions_json_path,
        matrix_data_path=matrix_data_path,
        output_dir=output_dir,
        selection_metric=selection_metric,
        sort_order=sort_order,
        rank=rank,
        polarity=polarity,
        min_area_px=min_area_px,
        max_area_px=max_area_px,
        min_edge_margin_px=min_edge_margin_px,
        prominence_reference=prominence_reference,
        local_ring_inner_px=local_ring_inner_px,
        local_ring_outer_px=local_ring_outer_px,
        min_local_background_samples=min_local_background_samples,
        render_candidates_overlay=render_candidates_overlay,
        render_selected_overlay=render_selected_overlay,
        colormap=colormap,
    )


def feature_line_build(
    feature_json_path: str,
    output_dir: str = "outputs/data/feature_line",
    direction_mode: LineDirectionMode = "minor_axis",
    angle_degrees: Optional[float] = None,
    substrate_margin_px: int = 28,
    min_substrate_samples: int = 6,
) -> JsonDict:
    """Build one validated substrate-feature-substrate line contract."""
    from afm_kite_core.operations.features.line_contract import (
        feature_line_build as _feature_line_build,
    )

    return invoke_operation(
        "feature-line-build",
        _feature_line_build,
        feature_json_path=feature_json_path,
        output_dir=output_dir,
        background_margin_px=substrate_margin_px,
        direction_mode=direction_mode,
        angle_degrees=angle_degrees,
        min_substrate_samples=min_substrate_samples,
    )


__all__ = ["feature_line_build", "feature_select"]

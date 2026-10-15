"""Public residual-feature preparation, segmentation, and measurement tools."""

from __future__ import annotations

from typing import Literal, Union

from afm_kite_core.contracts import (
    LevelMode,
    ResidualPolarity,
    ResidualSelectionMode,
)
from afm_kite_core.tools.boundary import JsonDict, invoke_operation


def residual_features_prepare(
    source_path: str,
    leveled_data_path: str,
    residual_data_path: str,
    output_dir: str = "outputs/data/residual_features",
    metadata_path: str = "",
    leveling: Union[LevelMode, Literal[""]] = "",
    background_sigma_px: float = 12.0,
    leveled_table_path: str = "",
    residual_table_path: str = "",
) -> JsonDict:
    """Write a contract linking original, leveled, and residual artifacts."""
    from afm_kite_core.operations.residual.preparation import write_defect_prepared_json

    return invoke_operation(
        "residual-features-prepare",
        write_defect_prepared_json,
        input_path=source_path,
        leveled_data_path=leveled_data_path,
        residual_data_path=residual_data_path,
        output_dir=output_dir,
        metadata_path=metadata_path,
        leveling=leveling,
        background_sigma_px=background_sigma_px,
        leveled_tsv_path=leveled_table_path,
        residual_tsv_path=residual_table_path,
    )


def residual_features_segment(
    prepared_json_path: str = "",
    residual_data_path: str = "",
    leveled_data_path: str = "",
    output_dir: str = "outputs/data/residual_features",
    polarity: ResidualPolarity = "dark",
    robust_sigma_threshold: float = 5.0,
    min_area_px: int = 3,
    min_contrast: float = 0.0,
    min_global_range_fraction: float = 0.03,
) -> JsonDict:
    """Segment candidate components in an existing residual matrix."""
    from afm_kite_core.operations.residual.segmentation import segment_defect_candidates

    return invoke_operation(
        "residual-features-segment",
        segment_defect_candidates,
        prepared_json_path=prepared_json_path,
        residual_data_path=residual_data_path,
        leveled_data_path=leveled_data_path,
        output_dir=output_dir,
        polarity=polarity,
        robust_sigma_threshold=robust_sigma_threshold,
        min_area_px=min_area_px,
        min_depth=min_contrast,
        min_global_range_fraction=min_global_range_fraction,
    )


def residual_features_measure(
    prepared_json_path: str,
    candidates_json_path: str,
    output_dir: str = "outputs/data/residual_features",
    reference_radius_px: int = 8,
    min_contrast: float = 0.0,
    colormap: str = "afm_gold_sampled",
    selection_mode: ResidualSelectionMode = "all",
    render_overlay: bool = False,
    render_candidates_overlay: bool = False,
    render_selected_overlay: bool = False,
) -> JsonDict:
    """Measure residual features and optionally render two review overlays."""
    from afm_kite_core.operations.residual.measurement import measure_residual_features

    return invoke_operation(
        "residual-features-measure",
        measure_residual_features,
        prepared_json_path=prepared_json_path,
        candidates_json_path=candidates_json_path,
        output_dir=output_dir,
        annulus_radius_px=reference_radius_px,
        min_depth=min_contrast,
        colormap=colormap,
        selection_mode=selection_mode,
        render_overlay=render_overlay,
        render_candidates_overlay=render_candidates_overlay,
        render_selected_overlay=render_selected_overlay,
    )


__all__ = [
    "residual_features_measure",
    "residual_features_prepare",
    "residual_features_segment",
]

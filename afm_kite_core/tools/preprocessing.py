"""Public matrix-preprocessing and rendering tools."""

from __future__ import annotations

from typing import Optional

from afm_kite_core.contracts import (
    FilterName,
    LevelMode,
    NanPolicy,
    PseudocolorScaleMode,
)
from afm_kite_core.tools.boundary import JsonDict, invoke_operation


def matrix_level(
    matrix_data_path: str,
    trend_mode: LevelMode,
    output_dir: str = "outputs/data/level",
    metadata_path: str = "",
) -> JsonDict:
    """Remove the selected low-order trend from a numeric matrix."""
    from afm_kite_core.operations.preprocessing import level_height_data

    return invoke_operation(
        "matrix-level",
        level_height_data,
        provenance_arguments={
            "matrix_data_path": matrix_data_path,
            "trend_mode": trend_mode,
            "output_dir": output_dir,
            "metadata_path": metadata_path,
        },
        height_data_path=matrix_data_path,
        output_dir=output_dir,
        mode=trend_mode,
        metadata_path=metadata_path,
    )


def matrix_filter(
    matrix_data_path: str,
    output_dir: str = "outputs/data/filter",
    filter_name: FilterName = "gaussian",
    sigma: float = 1.0,
    size: int = 3,
    metadata_path: str = "",
) -> JsonDict:
    """Apply a generic numeric image filter and write filtered matrix artifacts."""
    from afm_kite_core.operations.preprocessing import apply_matrix_filter

    return invoke_operation(
        "matrix-filter",
        apply_matrix_filter,
        height_data_path=matrix_data_path,
        output_dir=output_dir,
        filter_name=filter_name,
        sigma=sigma,
        size=size,
        metadata_path=metadata_path,
    )


def pseudocolor_render(
    matrix_data_path: str,
    output_dir: str = "outputs/data/render",
    colormap: str = "afm_gold_sampled",
    percentile_clip: float = 1.0,
    value_label: str = "value",
    metadata_path: str = "",
    plot_title: str = "",
    color_scale_mode: PseudocolorScaleMode = "auto",
    color_min: Optional[float] = None,
    color_max: Optional[float] = None,
) -> JsonDict:
    """Render a matrix with a colorbar and scale bar.

    Height maps always use the canonical bold ``Height`` title. ``plot_title``
    remains available for non-height matrices such as residuals.
    """
    from afm_kite_core.operations.preprocessing import render_pseudocolor

    return invoke_operation(
        "pseudocolor-render",
        render_pseudocolor,
        matrix_data_path=matrix_data_path,
        output_dir=output_dir,
        colormap=colormap,
        percentile_clip=percentile_clip,
        value_label=value_label,
        metadata_path=metadata_path,
        plot_title=plot_title,
        color_scale_mode=color_scale_mode,
        color_min=color_min,
        color_max=color_max,
    )


def matrices_subtract(
    minuend_data_path: str,
    subtrahend_data_path: str,
    output_dir: str = "outputs/data/matrix_ops",
    output_stem: str = "",
    nan_policy: NanPolicy = "propagate",
    metadata_path: str = "",
) -> JsonDict:
    """Subtract one same-shaped matrix from another and write residual artifacts."""
    from afm_kite_core.operations.preprocessing import subtract_matrices

    return invoke_operation(
        "matrices-subtract",
        subtract_matrices,
        minuend_data_path=minuend_data_path,
        subtrahend_data_path=subtrahend_data_path,
        output_dir=output_dir,
        output_stem=output_stem,
        nan_policy=nan_policy,
        metadata_path=metadata_path,
    )


__all__ = [
    "matrix_filter",
    "matrix_level",
    "matrices_subtract",
    "pseudocolor_render",
]

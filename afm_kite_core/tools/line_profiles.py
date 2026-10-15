"""Public explicit-line sampling, measurement, and rendering tools."""

from __future__ import annotations

from typing import Optional

from afm_kite_core.contracts import InterpolationMode, PlotKind, ProfileAggregation
from afm_kite_core.tools.boundary import JsonDict, invoke_operation


def line_profile_sample(
    line_contract_json_path: str,
    matrix_data_path: str,
    output_dir: str = "outputs/data/line_profile",
    profile_label: str = "",
    metadata_path: str = "",
    interpolation: InterpolationMode = "linear",
    line_width_px: int = 1,
    aggregation: ProfileAggregation = "median",
) -> JsonDict:
    """Sample one matrix along a center line or perpendicular pixel band."""
    from afm_kite_core.operations.line_profiles.sampling import sample_line_profile

    return invoke_operation(
        "line-profile-sample",
        sample_line_profile,
        line_contract_json_path=line_contract_json_path,
        matrix_data_path=matrix_data_path,
        output_dir=output_dir,
        profile_label=profile_label,
        metadata_path=metadata_path,
        interpolation=interpolation,
        line_width_px=line_width_px,
        aggregation=aggregation,
    )


def line_profile_measure(
    line_profile_json_paths: list[str],
    output_dir: str = "outputs/data/line_measurement",
) -> JsonDict:
    """Measure compatible feature-crossing profiles and compare an ordered pair."""
    from afm_kite_core.operations.line_profiles.measurement import measure_line_profiles

    return invoke_operation(
        "line-profile-measure",
        measure_line_profiles,
        line_profile_json_paths=line_profile_json_paths,
        output_dir=output_dir,
    )


def line_profile_render(
    line_contract_json_path: str,
    line_profile_json_paths: list[str],
    output_dir: str = "outputs/data/line_render",
    measurement_json_path: str = "",
    overlay_matrix_path: str = "",
    plot_kinds: Optional[list[PlotKind]] = None,
    colormap: str = "afm_gold_sampled",
    overlay_metadata_path: str = "",
    overlay_value_label: str = "Height",
) -> JsonDict:
    """Render optional profile and line-position plots without measuring values."""
    from afm_kite_core.operations.line_profiles.rendering import render_line_profiles

    return invoke_operation(
        "line-profile-render",
        render_line_profiles,
        line_contract_json_path=line_contract_json_path,
        line_profile_json_paths=line_profile_json_paths,
        output_dir=output_dir,
        measurement_json_path=measurement_json_path,
        overlay_matrix_path=overlay_matrix_path,
        plot_kinds=plot_kinds,
        colormap=colormap,
        overlay_metadata_path=overlay_metadata_path,
        overlay_value_label=overlay_value_label,
    )


__all__ = ["line_profile_measure", "line_profile_render", "line_profile_sample"]

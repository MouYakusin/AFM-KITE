"""Public matrix-roughness, directional-series, and frequency tools."""

from __future__ import annotations

from typing import Optional

from afm_kite_core.contracts import MatrixStatisticsScope, ProfileAxis
from afm_kite_core.tools.boundary import JsonDict, invoke_operation


def matrix_roughness_statistics(
    matrix_data_path: str,
    statistics_scopes: list[MatrixStatisticsScope],
    regions_data_path: str = "",
    output_dir: str = "outputs/data/roughness",
    value_unit: str = "",
    metadata_path: str = "",
    outliers_data_path: str = "",
) -> JsonDict:
    """Calculate deviation and distribution statistics for selected matrix scopes."""
    from afm_kite_core.operations.roughness import calculate_roughness

    scope_mapping = {
        "matrix": "surface",
        "regions": "regions",
        "region_union": "region_union",
        "background": "background",
    }
    return invoke_operation(
        "matrix-roughness-statistics",
        calculate_roughness,
        provenance_arguments={
            "matrix_data_path": matrix_data_path,
            "statistics_scopes": statistics_scopes,
            "regions_data_path": regions_data_path,
            "output_dir": output_dir,
            "value_unit": value_unit,
            "metadata_path": metadata_path,
            "outliers_data_path": outliers_data_path,
        },
        height_data_path=matrix_data_path,
        regions_json_path=regions_data_path,
        output_dir=output_dir,
        unit=value_unit,
        metadata_path=metadata_path,
        contaminants_json_path=outliers_data_path,
        scopes=[scope_mapping[scope] for scope in statistics_scopes],
    )


def frequency_direction_candidates(
    matrix_data_path: str,
    minimum_period_samples: float,
    maximum_period_samples: float,
    output_dir: str = "outputs/data/periodic_orientation",
    metadata_path: str = "",
) -> JsonDict:
    """Return direction candidates from a 2D frequency representation."""
    from afm_kite_core.operations.periodic.orientation import analyze_periodic_orientation

    return invoke_operation(
        "frequency-direction-candidates",
        analyze_periodic_orientation,
        provenance_arguments={
            "matrix_data_path": matrix_data_path,
            "minimum_period_samples": minimum_period_samples,
            "maximum_period_samples": maximum_period_samples,
            "output_dir": output_dir,
            "metadata_path": metadata_path,
        },
        height_data_path=matrix_data_path,
        output_dir=output_dir,
        min_period_px=minimum_period_samples,
        max_period_px=maximum_period_samples,
        metadata_path=metadata_path,
    )


def directional_series_extract(
    matrix_data_path: str,
    direction_mode: ProfileAxis,
    direction_degrees: Optional[float],
    bin_size_samples: float,
    smoothing_sigma_samples: float,
    output_dir: str = "outputs/data/profiles",
    metadata_path: str = "",
) -> JsonDict:
    """Reduce a matrix to a one-dimensional series along a supplied direction."""
    from afm_kite_core.operations.profiles import extract_median_profile

    return invoke_operation(
        "directional-series-extract",
        extract_median_profile,
        provenance_arguments={
            "matrix_data_path": matrix_data_path,
            "direction_mode": direction_mode,
            "direction_degrees": direction_degrees,
            "bin_size_samples": bin_size_samples,
            "smoothing_sigma_samples": smoothing_sigma_samples,
            "output_dir": output_dir,
            "metadata_path": metadata_path,
        },
        height_data_path=matrix_data_path,
        output_dir=output_dir,
        profile_axis=direction_mode,
        leveling="none",
        smooth_sigma_px=smoothing_sigma_samples,
        metadata_path=metadata_path,
        profile_angle_degrees=direction_degrees,
        projection_bin_size_px=bin_size_samples,
    )


def periodic_series_statistics(
    series_data_path: str,
    upper_percentile: float,
    lower_percentile: float,
    minimum_period_samples: float,
    maximum_period_samples: float,
    sample_spacing: float,
    coordinate_unit: str,
    value_unit: str,
    output_dir: str = "outputs/data/periodic_profile",
    coordinate_column: str = "",
    value_column: str = "",
) -> JsonDict:
    """Calculate periodic, segment, and level statistics for a numeric series."""
    from afm_kite_core.operations.periodic.measurement import measure_periodic_profile

    return invoke_operation(
        "periodic-series-statistics",
        measure_periodic_profile,
        provenance_arguments={
            "series_data_path": series_data_path,
            "upper_percentile": upper_percentile,
            "lower_percentile": lower_percentile,
            "minimum_period_samples": minimum_period_samples,
            "maximum_period_samples": maximum_period_samples,
            "sample_spacing": sample_spacing,
            "coordinate_unit": coordinate_unit,
            "value_unit": value_unit,
            "output_dir": output_dir,
            "coordinate_column": coordinate_column,
            "value_column": value_column,
        },
        profile_data_path=series_data_path,
        output_dir=output_dir,
        high_percentile=upper_percentile,
        low_percentile=lower_percentile,
        min_period_px=minimum_period_samples,
        max_period_px=maximum_period_samples,
        pixel_size=sample_spacing,
        axis_unit=coordinate_unit,
        height_unit=value_unit,
        coordinate_column=coordinate_column,
        height_column=value_column,
    )


__all__ = [
    "directional_series_extract",
    "frequency_direction_candidates",
    "matrix_roughness_statistics",
    "periodic_series_statistics",
]

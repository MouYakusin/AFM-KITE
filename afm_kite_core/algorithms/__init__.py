"""Reusable in-memory numerical algorithms for AFM analysis.

Public tools use these modules for NumPy/SciPy calculations without file I/O.
Artifact generation and tool registration remain outside these modules.
"""

from .feature_selection import rank_feature_candidates
from .filtering import filter_matrix
from .leveling import auto_level_matrix, level_matrix
from .line_profiles import (
    measure_profile_windows,
    physical_distance_along_line,
    sample_matrix_along_coordinates,
    sample_matrix_band_along_coordinates,
)
from .morphology import remove_objects_smaller_than
from .outliers import detect_local_outlier_components
from .periodic import (
    dominant_period,
    profile_extrema,
    profile_segments,
    refine_periodic_angle_by_projection,
    wrap_angle_0_180,
)
from .profiles import median_profile, projected_median_profile
from .roughness import masked_roughness_stats, normalise_roughness_scopes
from .statistics import (
    finite_matrix_summary,
    image_clip_limits,
    robust_location_scale,
    surface_stats,
)

__all__ = [
    "auto_level_matrix",
    "dominant_period",
    "detect_local_outlier_components",
    "filter_matrix",
    "finite_matrix_summary",
    "image_clip_limits",
    "level_matrix",
    "measure_profile_windows",
    "masked_roughness_stats",
    "median_profile",
    "normalise_roughness_scopes",
    "projected_median_profile",
    "physical_distance_along_line",
    "profile_extrema",
    "profile_segments",
    "rank_feature_candidates",
    "remove_objects_smaller_than",
    "refine_periodic_angle_by_projection",
    "robust_location_scale",
    "sample_matrix_band_along_coordinates",
    "sample_matrix_along_coordinates",
    "surface_stats",
    "wrap_angle_0_180",
]

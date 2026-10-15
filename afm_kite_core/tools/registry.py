"""Canonical public AFM tool catalog.

This is the only maintained list of model-exposed AFM analysis tools.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from afm_kite_core.tools.boundary import JsonDict

from .data_group import data_group_build
from .features import feature_line_build, feature_select
from .ingest import data_ingest, matrix_inspect
from .line_profiles import line_profile_measure, line_profile_render, line_profile_sample
from .paired_regions import (
    paired_region_catalog,
    paired_region_measure,
    paired_region_select,
)
from .periodic import (
    directional_series_extract,
    frequency_direction_candidates,
    matrix_roughness_statistics,
    periodic_series_statistics,
)
from .preprocessing import matrix_filter, matrix_level, matrices_subtract, pseudocolor_render
from .regions import gradient_regions_segment, local_outliers_detect, value_regions_segment
from .residual import (
    residual_features_measure,
    residual_features_prepare,
    residual_features_segment,
)

TOOL_FUNCTIONS: Sequence[Callable[..., JsonDict]] = (
    data_ingest,
    data_group_build,
    matrix_inspect,
    matrix_level,
    matrix_filter,
    pseudocolor_render,
    matrices_subtract,
    value_regions_segment,
    gradient_regions_segment,
    local_outliers_detect,
    residual_features_prepare,
    residual_features_segment,
    residual_features_measure,
    matrix_roughness_statistics,
    frequency_direction_candidates,
    directional_series_extract,
    periodic_series_statistics,
    feature_select,
    paired_region_catalog,
    paired_region_select,
    paired_region_measure,
    feature_line_build,
    line_profile_sample,
    line_profile_measure,
    line_profile_render,
)


def get_tools() -> list[Any]:
    """Build the canonical public functions as LangChain structured tools."""
    try:
        from langchain_core.tools import StructuredTool
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("langchain-core is required to build Deep Agents tools") from exc
    return [StructuredTool.from_function(function) for function in TOOL_FUNCTIONS]


__all__ = ["TOOL_FUNCTIONS", "get_tools"]

"""Shared types and lightweight artifact-contract declarations."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import Any, Literal, get_args

DATA_MANIFEST_SCHEMA = "afm.data-manifest/v1"
DATA_GROUP_SCHEMA = "afm.data-group/v1"
VALUE_REGIONS_SCHEMA = "afm.value-regions/v1"
GRADIENT_REGIONS_SCHEMA = "afm.gradient-regions/v1"
LOCAL_OUTLIERS_SCHEMA = "afm.local-outliers/v1"
SELECTED_FEATURE_SCHEMA = "afm.selected-feature/v1"
FEATURE_LINE_SCHEMA = "afm.feature-line/v1"
LINE_PROFILE_SCHEMA = "afm.line-profile/v1"
LINE_PROFILE_MEASUREMENT_SCHEMA = "afm.line-profile-measurement/v1"
PAIRED_REGION_MEASUREMENT_SCHEMA = "afm.paired-region-measurement/v1"
PAIRED_REGION_CATALOG_SCHEMA = "afm.paired-region-catalog/v1"
RESIDUAL_PREPARATION_SCHEMA = "afm.residual-preparation/v1"
RESIDUAL_CANDIDATES_SCHEMA = "afm.residual-candidates/v1"
RESIDUAL_MEASUREMENT_SCHEMA = "afm.residual-measurement/v1"


class ArtifactContractError(ValueError):
    """An artifact is missing a required schema or contract field."""


def coordinate_line_id(
    shape: Sequence[Any],
    x_coordinates: Sequence[Any],
    y_coordinates: Sequence[Any],
) -> str:
    """Build a readable line identifier from its shape and endpoints."""

    x_values = list(x_coordinates)
    y_values = list(y_coordinates)
    shape_text = "x".join(str(int(value)) for value in shape) or "unknown"
    if not x_values or not y_values:
        return f"line-{shape_text}-empty"
    return "line-{}-{}-{:.6g}-{:.6g}-{:.6g}-{:.6g}".format(
        shape_text,
        min(len(x_values), len(y_values)),
        float(x_values[0]),
        float(y_values[0]),
        float(x_values[-1]),
        float(y_values[-1]),
    )


def _nested_value(payload: Mapping[str, Any], dotted_key: str) -> Any:
    value: Any = payload
    for part in dotted_key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ArtifactContractError(f"Artifact is missing required field: {dotted_key}")
        value = value[part]
    return value


def validate_artifact_contract(
    payload: Mapping[str, Any],
    *,
    expected_schema: str | Collection[str] | None = None,
    required_keys: Sequence[str] = (),
) -> None:
    """Validate schema identity and required dotted fields for one JSON artifact."""

    if expected_schema is not None:
        allowed = (
            {expected_schema}
            if isinstance(expected_schema, str)
            else {str(item) for item in expected_schema}
        )
        actual = str(payload.get("schema") or "")
        if actual not in allowed:
            expected = ", ".join(sorted(allowed))
            raise ArtifactContractError(
                f"Unsupported artifact schema {actual!r}; expected one of: {expected}"
            )
    for key in required_keys:
        value = _nested_value(payload, key)
        if value in (None, ""):
            raise ArtifactContractError(f"Artifact field must not be empty: {key}")


DataRelationship = Literal[
    "collection",
    "ordered_conditions",
    "replicates",
    "time_series",
]
RelationshipSource = Literal[
    "user_explicit",
    "workflow_required",
    "agent_default",
]
RegistrationMode = Literal["none", "translation"]
LevelMode = Literal["plane", "line", "none", "auto"]
FilterName = Literal["gaussian", "median", "mean", "sobel", "laplace"]
NanPolicy = Literal["propagate", "left"]
PseudocolorScaleMode = Literal["auto", "symmetric_zero"]
ProfileAxis = Literal["x", "y", "angle"]
MatrixStatisticsScope = Literal[
    "matrix",
    "regions",
    "region_union",
    "background",
]
InterpolationMode = Literal["linear", "nearest"]
PairedRegionMaskMode = Literal["full", "eroded"]
PairedRegionSelectionMetric = Literal[
    "area_px",
    "condition_a_mean",
    "condition_b_mean",
    "change_mean",
    "absolute_change",
    "paired_finite_coverage",
]
ProfileAggregation = Literal["median", "mean"]
PlotKind = Literal["profile", "overlay"]
FeatureSelectionMetric = Literal[
    "area_px",
    "mean_value",
    "median_value",
    "median_prominence",
    "absolute_prominence",
]
SortOrder = Literal["ascending", "descending"]
FeaturePolarity = Literal["any", "raised", "depressed"]
ProminenceReference = Literal["local_ring", "global_background"]
LineDirectionMode = Literal["minor_axis", "major_axis", "x", "y", "angle"]
RegionPolarity = Literal["auto", "bright", "dark"]
GradientSegmentationMethod = Literal["auto", "percentile", "otsu", "hysteresis"]
OutlierPolarity = Literal["bright", "dark", "both"]
OutlierScope = Literal["regions", "full", "background", "all"]
ResidualPolarity = Literal["dark", "bright", "both"]
ResidualSelectionMode = Literal["all", "largest_area"]
RoughnessScope = Literal["surface", "regions", "background"]


DATA_RELATIONSHIPS = frozenset(get_args(DataRelationship))
RELATIONSHIP_SOURCES = frozenset(get_args(RelationshipSource))


__all__ = [
    "ArtifactContractError",
    "coordinate_line_id",
    "DATA_RELATIONSHIPS",
    "DATA_GROUP_SCHEMA",
    "DATA_MANIFEST_SCHEMA",
    "DataRelationship",
    "FeaturePolarity",
    "FEATURE_LINE_SCHEMA",
    "FeatureSelectionMetric",
    "FilterName",
    "GradientSegmentationMethod",
    "GRADIENT_REGIONS_SCHEMA",
    "InterpolationMode",
    "LevelMode",
    "LineDirectionMode",
    "LINE_PROFILE_MEASUREMENT_SCHEMA",
    "LINE_PROFILE_SCHEMA",
    "LOCAL_OUTLIERS_SCHEMA",
    "MatrixStatisticsScope",
    "NanPolicy",
    "OutlierPolarity",
    "OutlierScope",
    "PAIRED_REGION_MEASUREMENT_SCHEMA",
    "PAIRED_REGION_CATALOG_SCHEMA",
    "PairedRegionMaskMode",
    "PairedRegionSelectionMetric",
    "PlotKind",
    "ProfileAggregation",
    "ProfileAxis",
    "ProminenceReference",
    "PseudocolorScaleMode",
    "RELATIONSHIP_SOURCES",
    "RegistrationMode",
    "RegionPolarity",
    "RelationshipSource",
    "ResidualPolarity",
    "ResidualSelectionMode",
    "RESIDUAL_CANDIDATES_SCHEMA",
    "RESIDUAL_MEASUREMENT_SCHEMA",
    "RESIDUAL_PREPARATION_SCHEMA",
    "RoughnessScope",
    "SortOrder",
    "SELECTED_FEATURE_SCHEMA",
    "VALUE_REGIONS_SCHEMA",
    "validate_artifact_contract",
]

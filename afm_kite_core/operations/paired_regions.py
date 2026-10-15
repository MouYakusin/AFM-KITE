"""Transform-aware measurement of one selected region across an ordered pair."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from afm_kite_core.algorithms.statistics import image_clip_limits
from afm_kite_core.artifacts.plotting import (
    PLOT_DPI,
    PSEUDOCOLOR_FIGURE_SIZE,
    canonical_plot_title,
    decorate_height_map,
    plt,
    render_height_image,
    save_figure,
    set_plot_title,
)
from afm_kite_core.artifacts.writers import (
    atomic_write_path,
    read_json_file,
    safe_stem,
    write_json_file,
    write_text_file,
)
from afm_kite_core.contracts import (
    DATA_GROUP_SCHEMA,
    PAIRED_REGION_CATALOG_SCHEMA,
    PAIRED_REGION_MEASUREMENT_SCHEMA,
    SELECTED_FEATURE_SCHEMA,
    VALUE_REGIONS_SCHEMA,
)
from afm_kite_core.data_io.matrix import (
    load_height_data,
    load_height_data_and_metadata,
    metadata_sidecar_path,
    physical_area_context_from_metadata,
)
from afm_kite_core.data_io.paths import ensure_output_dir, resolve_readable_file


def _normal_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def _normal_unit(value: Any) -> str:
    return str(value or "").strip().casefold().replace("μ", "u").replace("µ", "u")


def _resolve_channel_key(
    channel_index: dict[str, Any],
    requested_channel: str,
) -> str:
    if requested_channel in channel_index:
        return requested_channel
    requested = _normal_name(requested_channel)
    candidates = [
        key
        for key in channel_index
        if _normal_name(key.split("#", 1)[0]) == requested
    ]
    if not candidates:
        raise ValueError(f"Target channel {requested_channel!r} is not present")
    if len(candidates) > 1:
        raise ValueError(
            f"Target channel {requested_channel!r} is ambiguous; choose one of "
            f"{candidates}"
        )
    return candidates[0]


def _distribution_statistics(values: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        raise ValueError("Cannot calculate statistics from an empty paired sample")
    return {
        "n": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "standard_deviation": float(np.std(finite)),
        "q05": float(np.percentile(finite, 5.0)),
        "q95": float(np.percentile(finite, 95.0)),
        "minimum": float(np.min(finite)),
        "maximum": float(np.max(finite)),
    }


def _sample_pair(
    reference_matrix: np.ndarray,
    moving_matrix: np.ndarray,
    candidate_mask: np.ndarray,
    shift_yx_px: tuple[float, float],
    interpolation: str,
) -> dict[str, Any]:
    y_px, x_px = np.nonzero(candidate_mask)
    candidate_count = int(y_px.size)
    if not candidate_count:
        return {
            "reference_values": np.asarray([], dtype=float),
            "moving_values": np.asarray([], dtype=float),
            "differences": np.asarray([], dtype=float),
            "valid_mask": np.zeros(candidate_mask.shape, dtype=bool),
            "candidate_count": 0,
            "in_bounds_count": 0,
            "paired_count": 0,
            "in_bounds_coverage": 0.0,
            "paired_coverage": 0.0,
        }

    shift_y, shift_x = shift_yx_px
    moving_y = y_px.astype(float) - float(shift_y)
    moving_x = x_px.astype(float) - float(shift_x)
    rows, cols = moving_matrix.shape
    in_bounds = (
        (moving_y >= 0.0)
        & (moving_y <= float(rows - 1))
        & (moving_x >= 0.0)
        & (moving_x <= float(cols - 1))
    )
    order = 1 if interpolation == "linear" else 0
    sampled_moving = ndi.map_coordinates(
        moving_matrix,
        [moving_y, moving_x],
        order=order,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    )
    sampled_reference = reference_matrix[y_px, x_px]
    paired = in_bounds & np.isfinite(sampled_reference) & np.isfinite(sampled_moving)

    valid_mask = np.zeros(candidate_mask.shape, dtype=bool)
    valid_mask[y_px[paired], x_px[paired]] = True
    reference_values = sampled_reference[paired]
    moving_values = sampled_moving[paired]
    differences = moving_values - reference_values
    in_bounds_count = int(np.sum(in_bounds))
    paired_count = int(np.sum(paired))
    return {
        "reference_values": reference_values,
        "moving_values": moving_values,
        "differences": differences,
        "valid_mask": valid_mask,
        "candidate_count": candidate_count,
        "in_bounds_count": in_bounds_count,
        "paired_count": paired_count,
        "in_bounds_coverage": float(in_bounds_count / candidate_count),
        "paired_coverage": float(paired_count / candidate_count),
    }


def _sampling_diagnostics(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_pixel_count": int(sample["candidate_count"]),
        "in_bounds_pixel_count": int(sample["in_bounds_count"]),
        "paired_finite_pixel_count": int(sample["paired_count"]),
        "in_bounds_coverage": float(sample["in_bounds_coverage"]),
        "paired_finite_coverage": float(sample["paired_coverage"]),
    }


def _paired_measurement(
    sample: dict[str, Any],
    *,
    first_member: dict[str, Any],
    second_member: dict[str, Any],
    value_unit: str,
) -> dict[str, Any]:
    first_statistics = _distribution_statistics(sample["reference_values"])
    second_statistics = _distribution_statistics(sample["moving_values"])
    difference_statistics = _distribution_statistics(sample["differences"])
    return {
        "condition_a": {
            "member_id": first_member["member_id"],
            "label": first_member["label"],
            "statistics": first_statistics,
        },
        "condition_b": {
            "member_id": second_member["member_id"],
            "label": second_member["label"],
            "statistics": second_statistics,
        },
        "paired_change_b_minus_a": {
            "operation": "condition_b_minus_condition_a",
            "value_unit": value_unit,
            "statistics": difference_statistics,
            "mean_identity_residual": float(
                difference_statistics["mean"]
                - (second_statistics["mean"] - first_statistics["mean"])
            ),
        },
        "value_unit": value_unit,
    }


def _region_mask(
    selected_mask: np.ndarray,
    *,
    mask_mode: str,
    erosion_px: int,
    min_region_pixels: int,
) -> tuple[np.ndarray, str, list[dict[str, Any]]]:
    warnings: list[dict[str, Any]] = []
    if mask_mode == "full":
        return selected_mask.copy(), "full", warnings

    eroded = (
        ndi.binary_erosion(selected_mask, iterations=int(erosion_px))
        if int(erosion_px) > 0
        else selected_mask.copy()
    )
    if int(np.sum(eroded)) >= int(min_region_pixels):
        return eroded.astype(bool), "eroded", warnings

    warnings.append(
        {
            "code": "eroded_mask_too_small",
            "message": (
                "The eroded selected-region mask contained fewer than the requested "
                "minimum pixels; the full selected mask was used instead."
            ),
            "eroded_pixel_count": int(np.sum(eroded)),
            "minimum_region_pixels": int(min_region_pixels),
        }
    )
    return selected_mask.copy(), "full_fallback", warnings


def _background_ring(
    selected_mask: np.ndarray,
    feature_labels: np.ndarray,
    *,
    inner_px: int,
    outer_px: int,
) -> np.ndarray:
    inner = (
        ndi.binary_dilation(selected_mask, iterations=int(inner_px))
        if int(inner_px) > 0
        else selected_mask.copy()
    )
    outer = ndi.binary_dilation(selected_mask, iterations=int(outer_px))
    return (outer & ~inner & (feature_labels == 0)).astype(bool)


def _load_catalog_pair_context(
    data_group_json_path: str,
    target_channel: str,
) -> dict[str, Any]:
    """Load and validate the ordered pair needed by full-field cataloging."""

    group_path, group = read_json_file(
        data_group_json_path,
        expected_schema=DATA_GROUP_SCHEMA,
        required_keys=(
            "status",
            "relationship.value",
            "members",
            "original_channel_index",
            "registration",
        ),
    )
    if group.get("status") != "ok":
        raise ValueError(f"Data group is not successful: {group_path}")
    relationship = group.get("relationship", {}) or {}
    if relationship.get("value") != "ordered_conditions" or not relationship.get(
        "ordered"
    ):
        raise ValueError(
            "paired_region_catalog requires an ordered_conditions data group"
        )
    members = list(group.get("members") or [])
    if len(members) != 2:
        raise ValueError("paired_region_catalog requires exactly two group members")
    if int(group.get("reference_data_index", -1)) != 0:
        raise ValueError(
            "paired_region_catalog requires condition A (member index 0) "
            "as the registration reference"
        )

    first_member, second_member = members
    first_id = str(first_member["member_id"])
    second_id = str(second_member["member_id"])
    if str(group.get("reference_member_id") or "") != first_id:
        raise ValueError("Data-group reference member does not match condition A")

    original_channel_index = group.get("original_channel_index") or {}
    channel_key = _resolve_channel_key(original_channel_index, target_channel)
    target_paths = original_channel_index[channel_key]
    if first_id not in target_paths or second_id not in target_paths:
        raise ValueError(
            f"Target channel {channel_key!r} is not present in both group members"
        )
    compatibility = (
        (group.get("compatibility") or {}).get("by_channel", {}).get(channel_key, {})
    )
    if compatibility.get("same_shape") is False:
        raise ValueError("Paired target-channel matrices must have the same shape")
    if compatibility.get("unit_compatible") is False:
        raise ValueError("Paired target-channel units are incompatible")
    if compatibility.get("same_scan_size") is False:
        raise ValueError("Paired target-channel scan sizes are incompatible")
    if compatibility.get("same_pixel_size") is False:
        raise ValueError("Paired target-channel pixel sizes are incompatible")

    records_by_member = (group.get("channel_records") or {}).get(channel_key, {})
    first_record = records_by_member.get(first_id, {}) or {}
    second_record = records_by_member.get(second_id, {}) or {}
    record_units = [
        str(record.get("unit") or "")
        for record in (first_record, second_record)
        if str(record.get("unit") or "").strip()
    ]
    if len({_normal_unit(unit) for unit in record_units}) > 1:
        raise ValueError("Paired target-channel units are incompatible")
    value_unit = record_units[0] if record_units else ""

    first_matrix, first_metadata, first_matrix_path = load_height_data_and_metadata(
        target_paths[first_id],
        str(first_record.get("metadata_path") or ""),
    )
    second_matrix_path = resolve_readable_file(target_paths[second_id])
    second_matrix = load_height_data(second_matrix_path)
    if first_matrix.shape != second_matrix.shape:
        raise ValueError(
            "Paired target-channel matrix shapes do not match: "
            f"{first_matrix.shape} and {second_matrix.shape}"
        )

    registration = group.get("registration") or {}
    requested_registration = str(registration.get("requested") or "none")
    applied_registration = str(registration.get("applied") or "none")
    if requested_registration == "translation":
        if applied_registration != "phase_cross_correlation_translation":
            raise ValueError("Requested translation registration was not applied")
        if str(registration.get("reference_member_id") or "") != first_id:
            raise ValueError("Translation registration is not referenced to condition A")
        transform = (registration.get("transforms") or {}).get(second_id)
        if not isinstance(transform, dict):
            raise ValueError("Condition-B translation transform is missing")
        shift_values = list(transform.get("shift_yx_px") or [])
        if len(shift_values) != 2:
            raise ValueError("Condition-B translation must contain shift_yx_px")
        shift_yx_px = (float(shift_values[0]), float(shift_values[1]))
        registration_error = transform.get("registration_error")
        phase_difference = transform.get("phase_difference")
    elif requested_registration == "none" and applied_registration == "none":
        shift_yx_px = (0.0, 0.0)
        registration_error = None
        phase_difference = None
    else:
        raise ValueError("Unsupported or inconsistent data-group registration contract")
    if not all(math.isfinite(value) for value in shift_yx_px):
        raise ValueError("Condition-B translation contains a non-finite shift")

    return {
        "group_path": group_path,
        "group": group,
        "first_member": first_member,
        "second_member": second_member,
        "first_id": first_id,
        "second_id": second_id,
        "channel_key": channel_key,
        "value_unit": value_unit,
        "first_matrix": first_matrix,
        "second_matrix": second_matrix,
        "first_metadata": first_metadata,
        "first_matrix_path": first_matrix_path,
        "second_matrix_path": second_matrix_path,
        "registration": registration,
        "requested_registration": requested_registration,
        "applied_registration": applied_registration,
        "shift_yx_px": shift_yx_px,
        "registration_error": registration_error,
        "phase_difference": phase_difference,
    }


def _register_moving_field(
    moving_matrix: np.ndarray,
    shift_yx_px: tuple[float, float],
    interpolation: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a moving matrix over the complete reference coordinate frame."""

    rows, cols = moving_matrix.shape
    y_px, x_px = np.mgrid[:rows, :cols]
    moving_y = y_px.astype(float) - float(shift_yx_px[0])
    moving_x = x_px.astype(float) - float(shift_yx_px[1])
    in_bounds = (
        (moving_y >= 0.0)
        & (moving_y <= float(rows - 1))
        & (moving_x >= 0.0)
        & (moving_x <= float(cols - 1))
    )
    sampled = ndi.map_coordinates(
        moving_matrix,
        [moving_y, moving_x],
        order=1 if interpolation == "linear" else 0,
        mode="constant",
        cval=np.nan,
        prefilter=False,
    )
    sampled[~in_bounds] = np.nan
    return sampled, in_bounds


def _field_region_sample(
    reference_matrix: np.ndarray,
    registered_moving_matrix: np.ndarray,
    in_bounds_mask: np.ndarray,
    candidate_mask: np.ndarray,
) -> dict[str, Any]:
    """Build the paired sample contract for one mask from registered fields."""

    candidate = np.asarray(candidate_mask, dtype=bool)
    candidate_count = int(np.sum(candidate))
    in_bounds = candidate & in_bounds_mask
    valid = (
        in_bounds
        & np.isfinite(reference_matrix)
        & np.isfinite(registered_moving_matrix)
    )
    reference_values = reference_matrix[valid]
    moving_values = registered_moving_matrix[valid]
    return {
        "reference_values": reference_values,
        "moving_values": moving_values,
        "differences": moving_values - reference_values,
        "valid_mask": valid,
        "candidate_count": candidate_count,
        "in_bounds_count": int(np.sum(in_bounds)),
        "paired_count": int(np.sum(valid)),
        "in_bounds_coverage": (
            float(np.sum(in_bounds) / candidate_count) if candidate_count else 0.0
        ),
        "paired_coverage": (
            float(np.sum(valid) / candidate_count) if candidate_count else 0.0
        ),
    }


def _region_geometry(mask: np.ndarray) -> dict[str, Any]:
    y_px, x_px = np.nonzero(mask)
    if not y_px.size:
        raise ValueError("Cannot describe an empty feature region")
    return {
        "area_px": int(y_px.size),
        "centroid_yx": [float(np.mean(y_px)), float(np.mean(x_px))],
        "bbox_xyxy": [
            int(np.min(x_px)),
            int(np.min(y_px)),
            int(np.max(x_px) + 1),
            int(np.max(y_px) + 1),
        ],
    }


def _render_paired_change_overlay(
    *,
    output_path: Path,
    difference_matrix: np.ndarray,
    labels: np.ndarray,
    region_records: list[dict[str, Any]],
    physical_context: dict[str, Any],
    value_unit: str,
    target_channel: str,
    selected_label_value: int | None,
) -> dict[str, Any]:
    """Render an existing full-field difference with catalog feature overlays."""

    display_channel = (
        "DCbias" if _normal_name(target_channel) == "dcbias" else target_channel
    )
    vmin, vmax = image_clip_limits(difference_matrix)
    bound = max(abs(float(vmin)), abs(float(vmax)))
    if not bound > 0:
        bound = 1.0
    fig, ax = plt.subplots(figsize=PSEUDOCOLOR_FIGURE_SIZE, dpi=PLOT_DPI)
    image = render_height_image(
        ax,
        difference_matrix,
        cmap="coolwarm",
        vmin=-bound,
        vmax=bound,
    )
    if np.any(labels > 0):
        ax.contour(labels > 0, levels=[0.5], colors="#8a8a8a", linewidths=0.65)
    if selected_label_value is not None:
        selected_mask = labels == int(selected_label_value)
        if np.any(selected_mask):
            ax.contour(
                selected_mask,
                levels=[0.5],
                colors="cyan",
                linewidths=1.35,
            )

    labeled_feature_ids: list[str] = []
    label_margin_px = max(8.0, min(labels.shape) * 0.04)
    for record in region_records:
        label_value = int(record["label_value"])
        feature_id = str(record["feature_id"])
        if selected_label_value is not None and label_value != selected_label_value:
            continue
        center_y, center_x = record["centroid_yx"]
        label_x = float(
            np.clip(
                center_x,
                label_margin_px,
                labels.shape[1] - label_margin_px - 1.0,
            )
        )
        label_y = float(
            np.clip(
                center_y,
                label_margin_px,
                labels.shape[0] - label_margin_px - 1.0,
            )
        )
        change_mean = record.get("change_mean")
        label_text = feature_id
        if selected_label_value is None and change_mean is not None:
            label_text = f"{feature_id}\n{float(change_mean):+.3g}"
        ax.text(
            label_x,
            label_y,
            label_text,
            color="white",
            fontsize=5.5,
            ha="center",
            va="center",
            bbox={
                "facecolor": "black",
                "alpha": 0.55,
                "pad": 1,
                "linewidth": 0,
            },
        )
        labeled_feature_ids.append(feature_id)

    requested_title = (
        f"{display_channel} Full-Field Difference (B - A)"
        if selected_label_value is None
        else f"{display_channel} Selected-Region Difference (B - A)"
    )
    title = canonical_plot_title(requested_title)
    set_plot_title(ax, title)
    decorations = decorate_height_map(
        fig,
        ax,
        image,
        matrix_shape=difference_matrix.shape,
        physical_context=physical_context,
        value_label=f"{display_channel} Δ(B-A)",
        value_unit=value_unit,
    )
    decorations["plot_title"] = title
    decorations["feature_overlay"] = {
        "mode": (
            "all_catalog_regions"
            if selected_label_value is None
            else "selected_catalog_region"
        ),
        "label_format": "F###",
        "labeled_feature_ids": labeled_feature_ids,
        "selected_label_value": selected_label_value,
    }
    fig.tight_layout()
    atomic_write_path(output_path, lambda path: save_figure(fig, path, pad_inches=0.02))
    plt.close(fig)
    return decorations


def _catalog_tsv(region_records: list[dict[str, Any]]) -> str:
    columns = [
        "feature_id",
        "label_value",
        "area_px",
        "centroid_y_px",
        "centroid_x_px",
        "condition_a_mean",
        "condition_b_mean",
        "change_mean",
        "absolute_change",
        "paired_finite_coverage",
        "quality_passed",
    ]
    lines = ["\t".join(columns)]
    for record in region_records:
        values = [
            record["feature_id"],
            record["label_value"],
            record["area_px"],
            record["centroid_yx"][0],
            record["centroid_yx"][1],
            record["condition_a_mean"],
            record["condition_b_mean"],
            record["change_mean"],
            record["absolute_change"],
            record["paired_finite_coverage"],
            record["quality"]["passed"],
        ]
        lines.append("\t".join(str(value) for value in values))
    return "\n".join(lines) + "\n"


def build_paired_region_catalog(
    data_group_json_path: str,
    regions_json_path: str,
    output_dir: str,
    target_channel: str,
    mask_mode: str,
    erosion_px: int,
    interpolation: str,
    min_coverage: float,
    min_region_pixels: int,
    max_shift_fraction: float,
    render_change_atlas: bool,
) -> dict[str, Any]:
    """Build a registered full-field difference and catalog every Height region."""

    if mask_mode not in {"full", "eroded"}:
        raise ValueError("mask_mode must be full or eroded")
    if interpolation not in {"linear", "nearest"}:
        raise ValueError("interpolation must be linear or nearest")
    if int(erosion_px) < 0:
        raise ValueError("erosion_px must be non-negative")
    if int(min_region_pixels) < 1:
        raise ValueError("min_region_pixels must be positive")
    if not 0.0 < float(min_coverage) <= 1.0:
        raise ValueError("min_coverage must be in (0, 1]")
    if not 0.0 < float(max_shift_fraction) <= 1.0:
        raise ValueError("max_shift_fraction must be in (0, 1]")

    context = _load_catalog_pair_context(data_group_json_path, target_channel)
    first_matrix = context["first_matrix"]
    registered_second, in_bounds = _register_moving_field(
        context["second_matrix"],
        context["shift_yx_px"],
        interpolation,
    )
    full_sample = _field_region_sample(
        first_matrix,
        registered_second,
        in_bounds,
        np.ones(first_matrix.shape, dtype=bool),
    )
    if not full_sample["paired_count"]:
        raise ValueError("The registered target fields have no finite paired pixels")
    full_difference = np.full(first_matrix.shape, np.nan, dtype=float)
    full_difference[full_sample["valid_mask"]] = full_sample["differences"]

    regions_path, regions_payload = read_json_file(
        regions_json_path,
        expected_schema=VALUE_REGIONS_SCHEMA,
        required_keys=("status", "label_npy_path"),
    )
    if regions_payload.get("status") != "ok":
        raise ValueError(f"Value-region segmentation is not successful: {regions_path}")
    labels_path = resolve_readable_file(regions_payload["label_npy_path"])
    labels = np.load(labels_path)
    if labels.shape != first_matrix.shape:
        raise ValueError(
            f"Feature label shape {labels.shape} does not match target shape "
            f"{first_matrix.shape}"
        )
    label_values = sorted(int(value) for value in np.unique(labels) if int(value) > 0)
    if not label_values:
        raise ValueError("The region label matrix contains no feature regions")

    shift_magnitude_px = float(math.hypot(*context["shift_yx_px"]))
    shift_fraction = float(shift_magnitude_px / min(first_matrix.shape))
    shift_ok = shift_fraction <= float(max_shift_fraction)
    region_records: list[dict[str, Any]] = []
    for label_value in label_values:
        full_mask = labels == label_value
        analysis_mask, mask_mode_used, warnings = _region_mask(
            full_mask,
            mask_mode=mask_mode,
            erosion_px=int(erosion_px),
            min_region_pixels=int(min_region_pixels),
        )
        sample = _field_region_sample(
            first_matrix,
            registered_second,
            in_bounds,
            analysis_mask,
        )
        geometry = _region_geometry(full_mask)
        region_size_ok = int(sample["candidate_count"]) >= int(min_region_pixels)
        coverage_ok = float(sample["paired_coverage"]) >= float(min_coverage)
        if not region_size_ok:
            warnings.append(
                {
                    "code": "region_below_minimum_size",
                    "candidate_pixel_count": int(sample["candidate_count"]),
                    "minimum_region_pixels": int(min_region_pixels),
                }
            )
        if not coverage_ok:
            warnings.append(
                {
                    "code": "paired_coverage_below_threshold",
                    "paired_finite_coverage": float(sample["paired_coverage"]),
                    "minimum_coverage": float(min_coverage),
                }
            )
        if sample["paired_count"]:
            measurement = _paired_measurement(
                sample,
                first_member=context["first_member"],
                second_member=context["second_member"],
                value_unit=context["value_unit"],
            )
            condition_a_mean = measurement["condition_a"]["statistics"]["mean"]
            condition_b_mean = measurement["condition_b"]["statistics"]["mean"]
            change_statistics = measurement["paired_change_b_minus_a"]["statistics"]
            change_mean = change_statistics["mean"]
        else:
            measurement = None
            condition_a_mean = None
            condition_b_mean = None
            change_statistics = None
            change_mean = None
        region_records.append(
            {
                "feature_id": f"F{label_value:03d}",
                "label_value": label_value,
                **geometry,
                "analysis_mask_area_px": int(sample["candidate_count"]),
                "mask_mode_requested": mask_mode,
                "mask_mode_used": mask_mode_used,
                "condition_a_mean": condition_a_mean,
                "condition_b_mean": condition_b_mean,
                "change_mean": change_mean,
                "absolute_change": (
                    abs(float(change_mean)) if change_mean is not None else None
                ),
                "change_statistics": change_statistics,
                "paired_finite_pixel_count": int(sample["paired_count"]),
                "paired_finite_coverage": float(sample["paired_coverage"]),
                "measurement": measurement,
                "quality": {
                    "passed": bool(region_size_ok and coverage_ok and shift_ok),
                    "region_size_passed": bool(region_size_ok),
                    "paired_coverage_passed": bool(coverage_ok),
                    "registration_shift_passed": bool(shift_ok),
                },
                "warnings": warnings,
            }
        )

    out_dir = ensure_output_dir(output_dir)
    source_stem = safe_stem(
        f"{context['group'].get('source_stem', context['group_path'].stem)}_"
        f"{context['channel_key']}",
        fallback="paired_field",
    )
    registered_second_path = out_dir / f"{source_stem}_condition_b_registered.npy"
    registered_second_metadata_path = metadata_sidecar_path(registered_second_path)
    difference_path = out_dir / f"{source_stem}_full_field_difference.npy"
    difference_metadata_path = metadata_sidecar_path(difference_path)
    valid_mask_path = out_dir / f"{source_stem}_full_field_valid_mask.npy"
    catalog_path = out_dir / f"{source_stem}_paired_region_catalog.json"
    table_path = out_dir / f"{source_stem}_paired_region_catalog.tsv"
    atlas_path = (
        out_dir / f"{source_stem}_paired_change_atlas.png"
        if render_change_atlas
        else None
    )
    atomic_write_path(
        registered_second_path,
        lambda path: np.save(path, registered_second),
    )
    write_json_file(registered_second_metadata_path, context["first_metadata"])
    atomic_write_path(difference_path, lambda path: np.save(path, full_difference))
    write_json_file(difference_metadata_path, context["first_metadata"])
    atomic_write_path(
        valid_mask_path,
        lambda path: np.save(path, full_sample["valid_mask"].astype(bool)),
    )
    write_text_file(table_path, _catalog_tsv(region_records))

    physical_context = physical_area_context_from_metadata(context["first_metadata"])
    atlas_decorations: dict[str, Any] = {}
    if atlas_path is not None:
        atlas_decorations = _render_paired_change_overlay(
            output_path=atlas_path,
            difference_matrix=full_difference,
            labels=labels,
            region_records=region_records,
            physical_context=physical_context,
            value_unit=context["value_unit"],
            target_channel=context["channel_key"],
            selected_label_value=None,
        )

    quality_region_count = sum(
        bool(record["quality"]["passed"]) for record in region_records
    )
    artifacts: dict[str, Any] = {
        "data_group_json_path": context["group_path"],
        "regions_json_path": regions_path,
        "feature_labels_npy_path": labels_path,
        "condition_a_target_matrix_path": context["first_matrix_path"],
        "condition_b_target_matrix_path": context["second_matrix_path"],
        "registered_condition_b_target_npy_path": registered_second_path,
        "registered_condition_b_target_metadata_path": (
            registered_second_metadata_path
        ),
        "full_field_difference_npy_path": difference_path,
        "full_field_difference_metadata_path": difference_metadata_path,
        "full_field_valid_mask_npy_path": valid_mask_path,
        "paired_region_catalog_json_path": catalog_path,
        "paired_region_catalog_tsv_path": table_path,
    }
    if atlas_path is not None:
        artifacts["paired_change_atlas_png_path"] = atlas_path

    payload: dict[str, Any] = {
        "status": "ok",
        "step": "paired_region_catalog",
        "schema": PAIRED_REGION_CATALOG_SCHEMA,
        "source_stem": source_stem,
        "target_channel_requested": target_channel,
        "target_channel_key": context["channel_key"],
        "value_unit": context["value_unit"],
        "member_order": [
            {
                "member_id": context["first_id"],
                "label": context["first_member"]["label"],
                "role": "condition_a_reference",
            },
            {
                "member_id": context["second_id"],
                "label": context["second_member"]["label"],
                "role": "condition_b_moving",
            },
        ],
        "registration": {
            "source_channel": context["registration"].get("registration_channel"),
            "requested": context["requested_registration"],
            "applied": context["applied_registration"],
            "shift_yx_px": list(context["shift_yx_px"]),
            "shift_magnitude_px": shift_magnitude_px,
            "shift_fraction_of_short_dimension": shift_fraction,
            "registration_error": context["registration_error"],
            "phase_difference": context["phase_difference"],
            "coordinate_mapping": (
                "condition_b_source_yx = condition_a_reference_yx - shift_yx_px"
            ),
        },
        "full_field": {
            "coordinate_frame": "condition_a_reference",
            "condition_b_source": "original_channel_index",
            "interpolation": interpolation,
            "out_of_bounds_policy": "nan_omit",
            **_sampling_diagnostics(full_sample),
            "difference_operation": "condition_b_minus_condition_a",
            "difference_statistics": _distribution_statistics(
                full_sample["differences"]
            ),
        },
        "catalog_parameters": {
            "mask_mode": mask_mode,
            "erosion_px": int(erosion_px),
            "minimum_region_pixels": int(min_region_pixels),
            "minimum_coverage": float(min_coverage),
            "maximum_shift_fraction": float(max_shift_fraction),
        },
        "region_count": len(region_records),
        "quality_passed_region_count": quality_region_count,
        "regions": region_records,
        "render_change_atlas": bool(render_change_atlas),
        "atlas_plot_decorations": atlas_decorations,
        "physical_context": physical_context,
        "quality": {
            "passed": bool(shift_ok and quality_region_count > 0),
            "registration_shift_passed": bool(shift_ok),
            "has_quality_passed_region": bool(quality_region_count > 0),
        },
        "summary": {
            "target_channel": context["channel_key"],
            "value_unit": context["value_unit"],
            "region_count": len(region_records),
            "quality_passed_region_count": quality_region_count,
            "full_field_paired_finite_coverage": float(
                full_sample["paired_coverage"]
            ),
            "full_field_change_mean": float(np.mean(full_sample["differences"])),
        },
        "artifacts": artifacts,
    }
    write_json_file(catalog_path, payload)
    return payload


def _parse_feature_id(feature_id: str) -> int:
    match = re.fullmatch(r"[Ff]?(\d+)", str(feature_id).strip())
    if match is None or int(match.group(1)) < 1:
        raise ValueError("feature_id must be a positive label such as F003")
    return int(match.group(1))


def select_paired_region(
    paired_region_catalog_json_path: str,
    output_dir: str,
    feature_id: str,
    x_px: float | None,
    y_px: float | None,
    selection_metric: str,
    sort_order: str,
    rank: int,
    min_area_px: int,
    min_coverage: float,
    render_selected_overlay: bool,
) -> dict[str, Any]:
    """Select one Height-defined region from an existing paired catalog."""

    allowed_metrics = {
        "area_px",
        "condition_a_mean",
        "condition_b_mean",
        "change_mean",
        "absolute_change",
        "paired_finite_coverage",
    }
    if selection_metric not in allowed_metrics:
        raise ValueError(
            "selection_metric must be one of: " + ", ".join(sorted(allowed_metrics))
        )
    if sort_order not in {"ascending", "descending"}:
        raise ValueError("sort_order must be ascending or descending")
    if int(rank) < 1:
        raise ValueError("rank must be a positive one-based index")
    if int(min_area_px) < 0:
        raise ValueError("min_area_px must be non-negative")
    if not 0.0 <= float(min_coverage) <= 1.0:
        raise ValueError("min_coverage must be in [0, 1]")
    has_x = x_px is not None
    has_y = y_px is not None
    if has_x != has_y:
        raise ValueError("x_px and y_px must be supplied together")
    if feature_id and has_x:
        raise ValueError("Choose either feature_id or a coordinate, not both")

    catalog_path, catalog = read_json_file(
        paired_region_catalog_json_path,
        expected_schema=PAIRED_REGION_CATALOG_SCHEMA,
        required_keys=(
            "status",
            "regions",
            "artifacts.feature_labels_npy_path",
            "artifacts.full_field_difference_npy_path",
        ),
    )
    if catalog.get("status") != "ok":
        raise ValueError(f"Paired-region catalog is not successful: {catalog_path}")
    artifacts = catalog.get("artifacts") or {}
    labels_path = resolve_readable_file(artifacts["feature_labels_npy_path"])
    labels = np.load(labels_path)
    region_records = list(catalog.get("regions") or [])
    records_by_label = {
        int(record["label_value"]): record for record in region_records
    }
    if not records_by_label:
        raise ValueError("Paired-region catalog contains no regions")

    selection_mode: str
    candidate_ranking: list[dict[str, Any]] = []
    if feature_id:
        selected_label = _parse_feature_id(feature_id)
        selection_mode = "feature_id"
    elif has_x and has_y:
        row = int(round(float(y_px)))
        column = int(round(float(x_px)))
        if not (0 <= row < labels.shape[0] and 0 <= column < labels.shape[1]):
            raise ValueError("The requested coordinate lies outside the region map")
        selected_label = int(labels[row, column])
        if selected_label < 1:
            raise ValueError("The requested coordinate does not lie inside a feature")
        selection_mode = "coordinate"
    else:
        eligible = [
            record
            for record in region_records
            if int(record["area_px"]) >= int(min_area_px)
            and float(record["paired_finite_coverage"]) >= float(min_coverage)
            and record.get(selection_metric) is not None
        ]
        if not eligible:
            raise ValueError("No catalog region satisfies the selection filters")
        eligible.sort(key=lambda record: int(record["label_value"]))
        eligible.sort(
            key=lambda record: float(record[selection_metric]),
            reverse=sort_order == "descending",
        )
        candidate_ranking = [
            {
                "rank": index,
                "feature_id": record["feature_id"],
                "label_value": int(record["label_value"]),
                "selection_metric": selection_metric,
                "selection_metric_value": record[selection_metric],
            }
            for index, record in enumerate(eligible, start=1)
        ]
        if int(rank) > len(eligible):
            raise ValueError(
                f"rank {int(rank)} exceeds {len(eligible)} eligible catalog regions"
            )
        selected_label = int(eligible[int(rank) - 1]["label_value"])
        selection_mode = "rank"

    if selected_label not in records_by_label:
        raise ValueError(f"Feature F{selected_label:03d} is not present in the catalog")
    selected_record = records_by_label[selected_label]
    selected_mask = labels == selected_label
    if not np.any(selected_mask):
        raise ValueError("Selected feature mask is empty")

    out_dir = ensure_output_dir(output_dir)
    source_stem = safe_stem(catalog.get("source_stem", catalog_path.stem))
    selected_mask_path = out_dir / f"{source_stem}_selected_feature_mask.npy"
    selected_json_path = out_dir / f"{source_stem}_selected_feature.json"
    overlay_path = (
        out_dir / f"{source_stem}_selected_change_overlay.png"
        if render_selected_overlay
        else None
    )
    atomic_write_path(
        selected_mask_path,
        lambda path: np.save(path, selected_mask.astype(bool)),
    )

    overlay_decorations: dict[str, Any] = {}
    difference_path = resolve_readable_file(
        artifacts["full_field_difference_npy_path"]
    )
    if overlay_path is not None:
        difference_matrix = np.load(difference_path)
        overlay_decorations = _render_paired_change_overlay(
            output_path=overlay_path,
            difference_matrix=difference_matrix,
            labels=labels,
            region_records=region_records,
            physical_context=catalog.get("physical_context") or {},
            value_unit=str(catalog.get("value_unit") or ""),
            target_channel=str(catalog.get("target_channel_key") or "value"),
            selected_label_value=selected_label,
        )

    output_artifacts: dict[str, Any] = {
        "paired_region_catalog_json_path": catalog_path,
        "value_regions_json_path": artifacts["regions_json_path"],
        "feature_labels_npy_path": labels_path,
        "matrix_data_path": difference_path,
        "selected_feature_mask_npy_path": selected_mask_path,
        "selected_feature_json_path": selected_json_path,
    }
    if overlay_path is not None:
        output_artifacts["selected_change_overlay_png_path"] = overlay_path

    selected_feature = {
        **selected_record,
        "selection_mode": selection_mode,
        "selection_metric": (
            selection_metric if selection_mode == "rank" else selection_mode
        ),
        "selection_metric_value": (
            selected_record.get(selection_metric)
            if selection_mode == "rank"
            else None
        ),
        "rank": int(rank) if selection_mode == "rank" else None,
    }
    payload: dict[str, Any] = {
        "status": "ok",
        "step": "paired_region_select",
        "schema": SELECTED_FEATURE_SCHEMA,
        "source_stem": source_stem,
        "regions_json_path": artifacts["regions_json_path"],
        "matrix_data_path": difference_path,
        "selection": {
            "mode": selection_mode,
            "requested_feature_id": feature_id or None,
            "requested_coordinate_xy_px": (
                [float(x_px), float(y_px)] if has_x and has_y else None
            ),
            "selection_metric": selection_metric,
            "sort_order": sort_order,
            "rank": int(rank),
            "min_area_px": int(min_area_px),
            "min_coverage": float(min_coverage),
            "candidate_ranking": candidate_ranking,
        },
        "selected_feature": selected_feature,
        "plots_written": bool(render_selected_overlay),
        "render_selected_overlay": bool(render_selected_overlay),
        "selected_overlay_plot_decorations": overlay_decorations,
        "segmentation": {
            "source": "paired-region-select from existing Height region labels",
            "region_count": len(region_records),
            "label_npy_path": labels_path,
            "regions_json_path": artifacts["regions_json_path"],
        },
        "artifacts": output_artifacts,
    }
    write_json_file(selected_json_path, payload)
    return payload


def measure_paired_region(
    data_group_json_path: str,
    selected_feature_json_path: str,
    output_dir: str,
    target_channel: str,
    mask_mode: str,
    erosion_px: int,
    interpolation: str,
    min_coverage: float,
    min_region_pixels: int,
    max_shift_fraction: float,
    calculate_background_normalized: bool,
    background_ring_inner_px: int,
    background_ring_outer_px: int,
    write_difference_matrix: bool,
) -> dict[str, Any]:
    """Measure one reference-frame region across two ordered conditions."""

    if mask_mode not in {"full", "eroded"}:
        raise ValueError("mask_mode must be full or eroded")
    if interpolation not in {"linear", "nearest"}:
        raise ValueError("interpolation must be linear or nearest")
    if int(erosion_px) < 0:
        raise ValueError("erosion_px must be non-negative")
    if int(min_region_pixels) < 1:
        raise ValueError("min_region_pixels must be positive")
    if not 0.0 < float(min_coverage) <= 1.0:
        raise ValueError("min_coverage must be in (0, 1]")
    if not 0.0 < float(max_shift_fraction) <= 1.0:
        raise ValueError("max_shift_fraction must be in (0, 1]")
    if int(background_ring_inner_px) < 0:
        raise ValueError("background_ring_inner_px must be non-negative")
    if int(background_ring_outer_px) <= int(background_ring_inner_px):
        raise ValueError(
            "background_ring_outer_px must be greater than "
            "background_ring_inner_px"
        )

    group_path, group = read_json_file(
        data_group_json_path,
        expected_schema=DATA_GROUP_SCHEMA,
        required_keys=(
            "status",
            "relationship.value",
            "members",
            "original_channel_index",
            "registration",
        ),
    )
    if group.get("status") != "ok":
        raise ValueError(f"Data group is not successful: {group_path}")
    relationship = group.get("relationship", {}) or {}
    if relationship.get("value") != "ordered_conditions" or not relationship.get(
        "ordered"
    ):
        raise ValueError(
            "paired_region_measure requires an ordered_conditions data group"
        )
    members = list(group.get("members") or [])
    if len(members) != 2:
        raise ValueError("paired_region_measure requires exactly two group members")
    if int(group.get("reference_data_index", -1)) != 0:
        raise ValueError(
            "paired_region_measure requires condition A (member index 0) "
            "as the registration reference"
        )
    first_member, second_member = members
    first_id = str(first_member["member_id"])
    second_id = str(second_member["member_id"])
    if str(group.get("reference_member_id") or "") != first_id:
        raise ValueError("Data-group reference member does not match condition A")

    original_channel_index = group.get("original_channel_index") or {}
    channel_key = _resolve_channel_key(original_channel_index, target_channel)
    target_paths = original_channel_index[channel_key]
    if first_id not in target_paths or second_id not in target_paths:
        raise ValueError(
            f"Target channel {channel_key!r} is not present in both group members"
        )
    compatibility = (
        (group.get("compatibility") or {}).get("by_channel", {}).get(channel_key, {})
    )
    if compatibility.get("same_shape") is False:
        raise ValueError("Paired target-channel matrices must have the same shape")
    if compatibility.get("unit_compatible") is False:
        raise ValueError("Paired target-channel units are incompatible")
    if compatibility.get("same_scan_size") is False:
        raise ValueError("Paired target-channel scan sizes are incompatible")
    if compatibility.get("same_pixel_size") is False:
        raise ValueError("Paired target-channel pixel sizes are incompatible")

    records_by_member = (group.get("channel_records") or {}).get(channel_key, {})
    first_record = records_by_member.get(first_id, {}) or {}
    second_record = records_by_member.get(second_id, {}) or {}
    record_units = [
        str(record.get("unit") or "")
        for record in (first_record, second_record)
        if str(record.get("unit") or "").strip()
    ]
    if len({_normal_unit(unit) for unit in record_units}) > 1:
        raise ValueError("Paired target-channel units are incompatible")
    value_unit = record_units[0] if record_units else ""

    first_matrix, first_metadata, first_matrix_path = load_height_data_and_metadata(
        target_paths[first_id],
        str(first_record.get("metadata_path") or ""),
    )
    second_matrix_path = resolve_readable_file(target_paths[second_id])
    second_matrix = load_height_data(second_matrix_path)
    if first_matrix.shape != second_matrix.shape:
        raise ValueError(
            "Paired target-channel matrix shapes do not match: "
            f"{first_matrix.shape} and {second_matrix.shape}"
        )

    feature_path, feature = read_json_file(
        selected_feature_json_path,
        expected_schema=SELECTED_FEATURE_SCHEMA,
        required_keys=(
            "status",
            "selected_feature.label_value",
            "artifacts.selected_feature_mask_npy_path",
        ),
    )
    if feature.get("status") != "ok":
        raise ValueError(f"Selected feature is not successful: {feature_path}")
    feature_artifacts = feature.get("artifacts") or {}
    selected_mask_path = resolve_readable_file(
        feature_artifacts["selected_feature_mask_npy_path"]
    )
    selected_mask = np.load(selected_mask_path).astype(bool)
    if selected_mask.shape != first_matrix.shape:
        raise ValueError(
            f"Selected-region mask shape {selected_mask.shape} does not match "
            f"target shape {first_matrix.shape}"
        )
    if not np.any(selected_mask):
        raise ValueError("Selected-region mask is empty")

    registration = group.get("registration") or {}
    requested_registration = str(registration.get("requested") or "none")
    applied_registration = str(registration.get("applied") or "none")
    if requested_registration == "translation":
        if applied_registration != "phase_cross_correlation_translation":
            raise ValueError("Requested translation registration was not applied")
        if str(registration.get("reference_member_id") or "") != first_id:
            raise ValueError("Translation registration is not referenced to condition A")
        transform = (registration.get("transforms") or {}).get(second_id)
        if not isinstance(transform, dict):
            raise ValueError("Condition-B translation transform is missing")
        shift_values = list(transform.get("shift_yx_px") or [])
        if len(shift_values) != 2:
            raise ValueError("Condition-B translation must contain shift_yx_px")
        shift_yx_px = (float(shift_values[0]), float(shift_values[1]))
        registration_error = transform.get("registration_error")
        phase_difference = transform.get("phase_difference")
    elif requested_registration == "none" and applied_registration == "none":
        shift_yx_px = (0.0, 0.0)
        registration_error = None
        phase_difference = None
    else:
        raise ValueError(
            "Unsupported or inconsistent data-group registration contract"
        )
    if not all(math.isfinite(value) for value in shift_yx_px):
        raise ValueError("Condition-B translation contains a non-finite shift")

    region_mask, mask_mode_used, warnings = _region_mask(
        selected_mask,
        mask_mode=mask_mode,
        erosion_px=int(erosion_px),
        min_region_pixels=int(min_region_pixels),
    )
    region_sample = _sample_pair(
        first_matrix,
        second_matrix,
        region_mask,
        shift_yx_px,
        interpolation,
    )
    if not region_sample["paired_count"]:
        raise ValueError("Selected region has no finite paired target-channel pixels")
    region_measurement = _paired_measurement(
        region_sample,
        first_member=first_member,
        second_member=second_member,
        value_unit=value_unit,
    )

    shift_magnitude_px = float(math.hypot(*shift_yx_px))
    shift_fraction = float(shift_magnitude_px / min(first_matrix.shape))
    region_size_ok = int(region_sample["candidate_count"]) >= int(min_region_pixels)
    coverage_ok = float(region_sample["paired_coverage"]) >= float(min_coverage)
    shift_ok = shift_fraction <= float(max_shift_fraction)
    if not region_size_ok:
        warnings.append(
            {
                "code": "region_below_minimum_size",
                "message": "The selected region contains fewer than min_region_pixels.",
                "candidate_pixel_count": int(region_sample["candidate_count"]),
                "minimum_region_pixels": int(min_region_pixels),
            }
        )
    if not coverage_ok:
        warnings.append(
            {
                "code": "paired_coverage_below_threshold",
                "message": (
                    "The selected region has insufficient finite paired coverage."
                ),
                "paired_finite_coverage": float(region_sample["paired_coverage"]),
                "minimum_coverage": float(min_coverage),
            }
        )
    if not shift_ok:
        warnings.append(
            {
                "code": "registration_shift_out_of_range",
                "message": (
                    "The translation magnitude exceeds max_shift_fraction of the "
                    "shorter matrix dimension."
                ),
                "shift_fraction": shift_fraction,
                "maximum_shift_fraction": float(max_shift_fraction),
            }
        )

    background_result: dict[str, Any] = {
        "requested": bool(calculate_background_normalized),
        "available": False,
    }
    background_valid_mask: np.ndarray | None = None
    background_normalized_change: dict[str, Any] | None = None
    if calculate_background_normalized:
        labels_path_value = feature_artifacts.get("feature_labels_npy_path")
        if not labels_path_value:
            warnings.append(
                {
                    "code": "background_labels_missing",
                    "message": (
                        "Local-background normalization was skipped because the "
                        "selected-feature artifact has no feature label matrix."
                    ),
                }
            )
        else:
            labels_path = resolve_readable_file(labels_path_value)
            labels = np.load(labels_path)
            if labels.shape != selected_mask.shape:
                raise ValueError(
                    "Feature label shape does not match selected-region mask shape"
                )
            background_candidate = _background_ring(
                selected_mask,
                labels,
                inner_px=int(background_ring_inner_px),
                outer_px=int(background_ring_outer_px),
            )
            background_sample = _sample_pair(
                first_matrix,
                second_matrix,
                background_candidate,
                shift_yx_px,
                interpolation,
            )
            background_valid_mask = background_sample["valid_mask"]
            if background_sample["paired_count"]:
                background_measurement = _paired_measurement(
                    background_sample,
                    first_member=first_member,
                    second_member=second_member,
                    value_unit=value_unit,
                )
                background_coverage_ok = (
                    float(background_sample["paired_coverage"])
                    >= float(min_coverage)
                )
                region_change = region_measurement["paired_change_b_minus_a"][
                    "statistics"
                ]["mean"]
                background_change = background_measurement[
                    "paired_change_b_minus_a"
                ]["statistics"]["mean"]
                background_normalized_change = {
                    "operation": (
                        "region_change_b_minus_a_minus_local_background_change_b_minus_a"
                    ),
                    "mean": float(region_change - background_change),
                    "value_unit": value_unit,
                    "reliable": bool(background_coverage_ok),
                }
                background_result = {
                    "requested": True,
                    "available": True,
                    "ring_inner_px": int(background_ring_inner_px),
                    "ring_outer_px": int(background_ring_outer_px),
                    "sampling": _sampling_diagnostics(background_sample),
                    "quality_passed": bool(background_coverage_ok),
                    "measurement": background_measurement,
                    "background_normalized_change": background_normalized_change,
                }
                if not background_coverage_ok:
                    warnings.append(
                        {
                            "code": "background_coverage_below_threshold",
                            "message": (
                                "The local background has insufficient finite paired "
                                "coverage; its normalized change is not reliable."
                            ),
                            "paired_finite_coverage": float(
                                background_sample["paired_coverage"]
                            ),
                            "minimum_coverage": float(min_coverage),
                        }
                    )
            else:
                warnings.append(
                    {
                        "code": "background_pair_empty",
                        "message": (
                            "The local background ring contains no finite paired "
                            "target-channel pixels."
                        ),
                    }
                )

    out_dir = ensure_output_dir(output_dir)
    source_stem = safe_stem(
        f"{group.get('source_stem', group_path.stem)}_{channel_key}",
        fallback="paired_region",
    )
    region_mask_path = out_dir / f"{source_stem}_region_mask.npy"
    valid_mask_path = out_dir / f"{source_stem}_paired_valid_mask.npy"
    measurement_path = out_dir / f"{source_stem}_paired_region_measurement.json"
    atomic_write_path(
        region_mask_path,
        lambda path: np.save(path, region_mask.astype(bool)),
    )
    atomic_write_path(
        valid_mask_path,
        lambda path: np.save(path, region_sample["valid_mask"].astype(bool)),
    )

    artifacts: dict[str, Any] = {
        "data_group_json_path": group_path,
        "selected_feature_json_path": feature_path,
        "selected_feature_mask_npy_path": selected_mask_path,
        "condition_a_target_matrix_path": first_matrix_path,
        "condition_b_target_matrix_path": second_matrix_path,
        "region_mask_npy_path": region_mask_path,
        "paired_valid_mask_npy_path": valid_mask_path,
        "paired_region_measurement_json_path": measurement_path,
    }
    if background_valid_mask is not None:
        background_mask_path = out_dir / f"{source_stem}_background_valid_mask.npy"
        atomic_write_path(
            background_mask_path,
            lambda path: np.save(path, background_valid_mask.astype(bool)),
        )
        artifacts["background_valid_mask_npy_path"] = background_mask_path

    if write_difference_matrix:
        difference_matrix = np.full(first_matrix.shape, np.nan, dtype=float)
        difference_matrix[region_sample["valid_mask"]] = region_sample["differences"]
        difference_path = out_dir / f"{source_stem}_paired_region_difference.npy"
        difference_metadata_path = metadata_sidecar_path(difference_path)
        atomic_write_path(
            difference_path,
            lambda path: np.save(path, difference_matrix),
        )
        write_json_file(difference_metadata_path, first_metadata)
        artifacts["paired_difference_npy_path"] = difference_path
        artifacts["paired_difference_metadata_path"] = difference_metadata_path

    quality_passed = bool(region_size_ok and coverage_ok and shift_ok)
    payload: dict[str, Any] = {
        "status": "ok",
        "step": "paired_region_measure",
        "schema": PAIRED_REGION_MEASUREMENT_SCHEMA,
        "source_stem": source_stem,
        "data_group_json_path": group_path,
        "selected_feature_json_path": feature_path,
        "target_channel_requested": target_channel,
        "target_channel_key": channel_key,
        "value_unit": value_unit,
        "member_order": [
            {
                "member_id": first_id,
                "label": first_member["label"],
                "role": "condition_a_reference",
            },
            {
                "member_id": second_id,
                "label": second_member["label"],
                "role": "condition_b_moving",
            },
        ],
        "registration": {
            "source_channel": registration.get("registration_channel"),
            "requested": requested_registration,
            "applied": applied_registration,
            "shift_yx_px": list(shift_yx_px),
            "shift_magnitude_px": shift_magnitude_px,
            "shift_fraction_of_short_dimension": shift_fraction,
            "registration_error": registration_error,
            "phase_difference": phase_difference,
            "coordinate_mapping": (
                "condition_b_source_yx = condition_a_reference_yx - shift_yx_px"
            ),
        },
        "sampling": {
            "coordinate_frame": "condition_a_reference",
            "condition_b_source": "original_channel_index",
            "interpolation": interpolation,
            "out_of_bounds_policy": "nan_omit",
            **_sampling_diagnostics(region_sample),
        },
        "mask": {
            "requested_mode": mask_mode,
            "used_mode": mask_mode_used,
            "erosion_px": int(erosion_px),
            "minimum_region_pixels": int(min_region_pixels),
            "selected_feature_label": int(
                feature["selected_feature"]["label_value"]
            ),
            "selected_full_pixel_count": int(np.sum(selected_mask)),
            "analysis_mask_pixel_count": int(np.sum(region_mask)),
        },
        "measurement": region_measurement,
        "local_background": background_result,
        "background_normalized_change": background_normalized_change,
        "quality": {
            "passed": quality_passed,
            "minimum_coverage": float(min_coverage),
            "maximum_shift_fraction": float(max_shift_fraction),
            "region_size_passed": bool(region_size_ok),
            "paired_coverage_passed": bool(coverage_ok),
            "registration_shift_passed": bool(shift_ok),
            "warning_count": len(warnings),
        },
        "warnings": warnings,
        "difference_matrix_written": bool(write_difference_matrix),
        "summary": {
            "quality_passed": quality_passed,
            "target_channel": channel_key,
            "value_unit": value_unit,
            "paired_pixel_count": int(region_sample["paired_count"]),
            "paired_finite_coverage": float(region_sample["paired_coverage"]),
            "condition_a_mean": region_measurement["condition_a"]["statistics"][
                "mean"
            ],
            "condition_b_mean": region_measurement["condition_b"]["statistics"][
                "mean"
            ],
            "paired_change_b_minus_a_mean": region_measurement[
                "paired_change_b_minus_a"
            ]["statistics"]["mean"],
            "background_normalized_change_mean": (
                background_normalized_change["mean"]
                if background_normalized_change is not None
                else None
            ),
        },
        "artifacts": artifacts,
    }
    write_json_file(measurement_path, payload)
    return payload


__all__ = [
    "build_paired_region_catalog",
    "measure_paired_region",
    "select_paired_region",
]

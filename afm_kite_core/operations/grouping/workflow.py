"""Data-group compatibility, registration, and assembly workflow."""

from __future__ import annotations

import copy
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from skimage.registration import phase_cross_correlation

from afm_kite_core.algorithms.leveling import level_matrix
from afm_kite_core.contracts import (
    DATA_GROUP_SCHEMA,
    DATA_MANIFEST_SCHEMA,
    DATA_RELATIONSHIPS,
    RELATIONSHIP_SOURCES,
)
from afm_kite_core.artifacts.writers import (
    atomic_write_path,
    read_json_file,
    safe_stem,
    write_json_file,
)
from afm_kite_core.data_io.matrix import (
    metadata_sidecar_path,
    metadata_with_defaults,
)
from afm_kite_core.data_io.paths import ensure_output_dir, resolve_readable_file


def _normal_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _load_manifest(path: str) -> tuple[Path, dict[str, Any]]:
    manifest_path, payload = read_json_file(
        path,
        expected_schema=DATA_MANIFEST_SCHEMA,
        required_keys=("status", "channels"),
    )
    if payload.get("status") != "ok":
        raise ValueError(f"Data manifest is not successful: {manifest_path}")
    if int(payload.get("input_count", 1)) != 1:
        raise ValueError(
            "data_group_build accepts manifests produced from exactly one raw file"
        )
    channels = payload.get("channels")
    if not isinstance(channels, list) or not channels:
        raise ValueError(f"Data manifest has no channels: {manifest_path}")
    for channel in channels:
        matrix_path = channel.get("matrix_npy_path")
        if not matrix_path:
            raise ValueError(
                f"Manifest channel {channel.get('channel_id')!r} has no matrix_npy_path"
            )
        resolve_readable_file(matrix_path)
    return manifest_path, payload


def _default_labels(relationship: str, count: int) -> list[str]:
    if relationship == "ordered_conditions":
        return [f"condition_{chr(ord('a') + index)}" for index in range(count)]
    if relationship == "replicates":
        return [f"replicate_{index + 1}" for index in range(count)]
    if relationship == "time_series":
        return [f"timepoint_{index}" for index in range(count)]
    return [f"data_{index}" for index in range(count)]


def _channel_key_records(channels: list[dict[str, Any]]
                         ) -> tuple[dict[str, dict[str, Any]], list[str]]:
    base_names = [
        str(channel.get("normalized_name") or f"channel_{index + 1:02d}")
        for index, channel in enumerate(channels)
    ]
    totals = Counter(base_names)
    occurrences: Counter[str] = Counter()
    records: dict[str, dict[str, Any]] = {}
    duplicate_names = sorted(name for name, count in totals.items() if count > 1)
    for base_name, channel in zip(base_names, channels):
        occurrences[base_name] += 1
        key = (
            base_name
            if totals[base_name] == 1
            else f"{base_name}#{occurrences[base_name]}"
        )
        records[key] = channel
    return records, duplicate_names


def _float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _record_physical_values(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata_summary", {}) or {}
    x_size = _float(metadata.get("x_size"))
    y_size = _float(metadata.get("y_size"))
    x_resolution = _float(metadata.get("x_resolution"))
    y_resolution = _float(metadata.get("y_resolution"))
    pixel_width = (
        x_size / x_resolution if x_size is not None and x_resolution not in {None, 0.0} else None
    )
    pixel_height = (
        y_size / y_resolution if y_size is not None and y_resolution not in {None, 0.0} else None
    )
    return {
        "x_size": x_size,
        "y_size": y_size,
        "x_unit": str(metadata.get("x_unit") or ""),
        "y_unit": str(metadata.get("y_unit") or ""),
        "x_resolution": int(x_resolution) if x_resolution is not None else None,
        "y_resolution": int(y_resolution) if y_resolution is not None else None,
        "pixel_width": pixel_width,
        "pixel_height": pixel_height,
        "height_unit": str(record.get("unit") or ""),
    }


def _optional_equal(values: list[Any], *, numeric: bool = False) -> bool | None:
    if any(value in {None, ""} for value in values):
        return None
    if numeric:
        numeric_values = [float(value) for value in values]
        first = numeric_values[0]
        return all(math.isclose(value, first, rel_tol=1e-6, abs_tol=1e-12)
                   for value in numeric_values[1:])
    normalized = [str(value).strip().lower().replace("μ", "u").replace("µ", "u")
                  for value in values]
    return len(set(normalized)) == 1


def _compatibility_for_channel(
    channel_key: str,
    member_ids: list[str],
    records_by_member: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    present_ids = [member_id for member_id in member_ids if member_id in records_by_member]
    records = [records_by_member[member_id] for member_id in present_ids]
    present_in_all = len(present_ids) == len(member_ids)
    shapes = [tuple(record.get("shape", [])) for record in records]
    same_shape = len(set(shapes)) <= 1 if records else False
    units = [str(record.get("unit") or "") for record in records]
    known_units = [unit for unit in units if unit]
    unit_compatible = len(
        {unit.lower().replace("μ", "u").replace("µ", "u") for unit in known_units}
    ) <= 1
    physical = [_record_physical_values(record) for record in records]
    same_scan_size = None
    same_pixel_size = None
    if physical:
        x_size_equal = _optional_equal([item["x_size"] for item in physical], numeric=True)
        y_size_equal = _optional_equal([item["y_size"] for item in physical], numeric=True)
        x_unit_equal = _optional_equal([item["x_unit"] for item in physical])
        y_unit_equal = _optional_equal([item["y_unit"] for item in physical])
        checks = [x_size_equal, y_size_equal, x_unit_equal, y_unit_equal]
        same_scan_size = None if any(item is None for item in checks) else all(checks)

        pixel_width_equal = _optional_equal(
            [item["pixel_width"] for item in physical], numeric=True
        )
        pixel_height_equal = _optional_equal(
            [item["pixel_height"] for item in physical], numeric=True
        )
        pixel_checks = [pixel_width_equal, pixel_height_equal]
        same_pixel_size = (
            None if any(item is None for item in pixel_checks) else all(pixel_checks)
        )

    direct = (
        present_in_all
        and same_shape
        and unit_compatible
        and same_scan_size is not False
        and same_pixel_size is not False
    )
    return {
        "channel_key": channel_key,
        "present_in_all": present_in_all,
        "present_member_ids": present_ids,
        "missing_member_ids": [item for item in member_ids if item not in records_by_member],
        "shapes": [list(shape) for shape in sorted(set(shapes))],
        "same_shape": same_shape,
        "units": sorted(set(units)),
        "unit_compatible": unit_compatible,
        "same_scan_size": same_scan_size,
        "same_pixel_size": same_pixel_size,
        "direct_comparison_compatible": direct,
    }


def _finite_fill(matrix: np.ndarray) -> np.ndarray:
    finite = np.isfinite(matrix)
    if not np.any(finite):
        raise ValueError("Registration matrix has no finite values")
    return np.where(finite, matrix, float(np.nanmedian(matrix[finite])))


def _resolve_registration_key(channel_index: dict[str, Any], requested: str) -> str:
    if requested in channel_index:
        return requested
    normalized = _normal_name(requested)
    candidates = [
        key
        for key in channel_index
        if _normal_name(key.split("#", 1)[0]) == normalized
    ]
    if not candidates:
        raise ValueError(f"Registration channel {requested!r} is not present")
    if len(candidates) > 1:
        raise ValueError(
            f"Registration channel {requested!r} is ambiguous; choose one of {candidates}"
        )
    return candidates[0]


def _register_members(
    *,
    out_dir: Path,
    source_stem: str,
    members: list[dict[str, Any]],
    reference_data_index: int,
    registration_channel: str,
    channel_index: dict[str, dict[str, str]],
    channel_records: dict[str, dict[str, dict[str, Any]]],
    compatibility: dict[str, dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> tuple[
    dict[str, dict[str, str]],
    dict[str, Any],
    list[str],
    list[str],
]:
    registration_key = _resolve_registration_key(channel_index, registration_channel)
    channel_compatibility = compatibility[registration_key]
    if not channel_compatibility["present_in_all"]:
        raise ValueError(
            f"Registration channel {registration_key!r} is missing from one or more members"
        )
    if not channel_compatibility["same_shape"]:
        raise ValueError("Registration requires matching matrix shapes")
    if channel_compatibility["same_scan_size"] is False:
        raise ValueError("Registration requires matching physical scan sizes")
    if channel_compatibility["same_pixel_size"] is False:
        raise ValueError("Registration requires matching physical pixel sizes")

    registered_index = copy.deepcopy(channel_index)
    reference_member = members[reference_data_index]
    reference_id = reference_member["member_id"]
    reference_record = channel_records[registration_key][reference_id]
    reference_matrix = np.load(resolve_readable_file(channel_index[registration_key][reference_id]))
    reference_context = _record_physical_values(reference_record)
    reference_leveled, _ = level_matrix(reference_matrix, "plane", reference_context)
    reference_shape = tuple(reference_matrix.shape)
    transforms: dict[str, Any] = {
        reference_id: {
            "label": reference_member["label"],
            "shift_yx_px": [0.0, 0.0],
            "registration_error": 0.0,
            "phase_difference": 0.0,
            "applied_channel_keys": [],
            "skipped_channel_keys": [],
        }
    }
    registered_paths: list[str] = []
    registered_metadata_paths: list[str] = []

    for member in members:
        member_id = member["member_id"]
        if member_id == reference_id:
            continue
        moving_record = channel_records[registration_key][member_id]
        moving_matrix = np.load(
            resolve_readable_file(channel_index[registration_key][member_id])
        )
        moving_context = _record_physical_values(moving_record)
        moving_leveled, _ = level_matrix(moving_matrix, "plane", moving_context)
        shift_yx, error, phase = phase_cross_correlation(
            _finite_fill(reference_leveled),
            _finite_fill(moving_leveled),
            upsample_factor=10,
        )
        applied: list[str] = []
        skipped: list[str] = []
        for channel_key, member_paths in channel_index.items():
            matrix_path = member_paths.get(member_id)
            if not matrix_path:
                continue
            matrix = np.load(resolve_readable_file(matrix_path))
            if tuple(matrix.shape) != reference_shape:
                skipped.append(channel_key)
                warnings.append(
                    {
                        "code": "registration_channel_skipped",
                        "message": (
                            "A member channel was not registered because its shape "
                            "differs from the registration channel."
                        ),
                        "member_id": member_id,
                        "channel_key": channel_key,
                    }
                )
                continue
            shifted = ndi.shift(
                matrix,
                shift_yx,
                order=1,
                mode="constant",
                cval=np.nan,
                prefilter=False,
            )
            registered_path = out_dir / (
                f"{source_stem}_{member_id}_{safe_stem(channel_key)}_registered.npy"
            )
            atomic_write_path(registered_path, lambda path: np.save(path, shifted))
            registered_metadata_path = metadata_sidecar_path(registered_path)
            source_record = channel_records[channel_key][member_id]
            source_metadata = source_record.get("metadata") or {}
            if not source_metadata and source_record.get("metadata_path"):
                _, source_metadata = read_json_file(source_record["metadata_path"])
            write_json_file(
                registered_metadata_path,
                metadata_with_defaults(source_metadata, shifted.shape),
            )
            registered_index[channel_key][member_id] = str(registered_path)
            registered_paths.append(str(registered_path))
            registered_metadata_paths.append(str(registered_metadata_path))
            applied.append(channel_key)
        transforms[member_id] = {
            "label": member["label"],
            "shift_yx_px": [float(shift_yx[0]), float(shift_yx[1])],
            "registration_error": float(error),
            "phase_difference": float(phase),
            "applied_channel_keys": applied,
            "skipped_channel_keys": skipped,
        }

    registration = {
        "requested": "translation",
        "applied": "phase_cross_correlation_translation",
        "registration_channel": registration_key,
        "reference_data_index": int(reference_data_index),
        "reference_member_id": reference_id,
        "interpolation": "linear",
        "out_of_bounds_policy": "nan",
        "transforms": transforms,
    }
    return (
        registered_index,
        registration,
        registered_paths,
        registered_metadata_paths,
    )


def _normalise_group_options(
    data_manifest_paths: list[str],
    relationship: str,
    relationship_source: str,
    reference_data_index: int,
    registration: str,
) -> tuple[str, str, int, str]:
    """Validate and normalize data-group options before reading artifacts."""

    if len(data_manifest_paths) < 2:
        raise ValueError("data_group_build requires at least two data manifests")
    relationship = str(relationship).strip().lower()
    if relationship not in DATA_RELATIONSHIPS:
        raise ValueError(
            "relationship must be collection, ordered_conditions, replicates, or time_series"
        )
    relationship_source = str(relationship_source).strip().lower()
    if relationship_source not in RELATIONSHIP_SOURCES:
        raise ValueError(
            "relationship_source must be user_explicit, workflow_required, or agent_default"
        )
    registration = str(registration).strip().lower()
    if registration not in {"none", "translation"}:
        raise ValueError("registration must be none or translation")
    reference_index = int(reference_data_index)
    if not 0 <= reference_index < len(data_manifest_paths):
        raise ValueError("reference_data_index is out of range")
    return relationship, relationship_source, reference_index, registration


def _load_group_inputs(
    data_manifest_paths: list[str],
    labels: list[str] | None,
    relationship: str,
) -> tuple[list[tuple[Path, dict[str, Any]]], list[str]]:
    """Load unique manifests and establish validated member labels."""

    loaded = [_load_manifest(path) for path in data_manifest_paths]
    resolved_paths = [str(path) for path, _ in loaded]
    if len(set(resolved_paths)) != len(resolved_paths):
        raise ValueError("data_manifest_paths contains a duplicate manifest")

    member_labels = list(labels) if labels is not None else _default_labels(
        relationship, len(loaded)
    )
    if len(member_labels) != len(loaded):
        raise ValueError("labels count must match data_manifest_paths count")
    member_labels = [str(label).strip() for label in member_labels]
    if any(not label for label in member_labels):
        raise ValueError("labels must not contain empty values")
    if len(set(member_labels)) != len(member_labels):
        raise ValueError("labels must be unique")
    return loaded, member_labels


def _build_group_members(
    loaded: list[tuple[Path, dict[str, Any]]],
    member_labels: list[str],
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, dict[str, Any]]],
    list[dict[str, Any]],
]:
    """Build member records and collect duplicate-channel warnings."""

    members: list[dict[str, Any]] = []
    records_per_member: dict[str, dict[str, dict[str, Any]]] = {}
    warnings: list[dict[str, Any]] = []
    for index, ((manifest_path, manifest), label) in enumerate(
        zip(loaded, member_labels)
    ):
        member_id = f"member_{index:03d}"
        source = manifest.get("source", {}) or {}
        source_path = str(
            source.get("input_path")
            or (manifest.get("inputs") or [{}])[0].get("path", "")
        )
        channel_records, duplicates = _channel_key_records(list(manifest["channels"]))
        if duplicates:
            warnings.append(
                {
                    "code": "ambiguous_channel_match",
                    "message": (
                        "A member contains duplicate normalized channel names; "
                        "occurrence suffixes were assigned."
                    ),
                    "member_id": member_id,
                    "normalized_names": duplicates,
                }
            )
        records_per_member[member_id] = channel_records
        members.append(
            {
                "index": index,
                "member_id": member_id,
                "label": label,
                "data_id": manifest.get("data_id", manifest_path.stem),
                "data_manifest_path": str(manifest_path),
                "source_path": source_path,
                "channel_count": len(manifest["channels"]),
                "channel_keys": list(channel_records),
            }
        )
    return members, records_per_member, warnings


def _build_group_channel_indexes(
    members: list[dict[str, Any]],
    records_per_member: dict[str, dict[str, dict[str, Any]]],
    warnings: list[dict[str, Any]],
) -> tuple[
    dict[str, dict[str, dict[str, Any]]],
    dict[str, dict[str, str]],
    dict[str, dict[str, Any]],
    list[str],
    list[str],
]:
    """Index channels, compute compatibility, and append group-level warnings."""

    member_ids = [member["member_id"] for member in members]
    grouped_records: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for member_id, records in records_per_member.items():
        for channel_key, record in records.items():
            grouped_records[channel_key][member_id] = record
    channel_records = dict(grouped_records)

    original_channel_index = {
        channel_key: {
            member_id: str(record["matrix_npy_path"])
            for member_id, record in records.items()
        }
        for channel_key, records in channel_records.items()
    }
    compatibility = {
        channel_key: _compatibility_for_channel(channel_key, member_ids, records)
        for channel_key, records in channel_records.items()
    }
    common_channels = sorted(
        key for key, item in compatibility.items() if item["present_in_all"]
    )
    partial_channels = sorted(
        key for key, item in compatibility.items() if not item["present_in_all"]
    )
    for channel_key in partial_channels:
        warnings.append(
            {
                "code": "partial_channel",
                "message": "A channel is missing from one or more group members.",
                "channel_key": channel_key,
                "missing_member_ids": compatibility[channel_key]["missing_member_ids"],
            }
        )
    for channel_key, item in compatibility.items():
        if item["present_in_all"] and not item["direct_comparison_compatible"]:
            warnings.append(
                {
                    "code": "incompatible_common_channel",
                    "message": (
                        "A common channel is not directly comparable across all members."
                    ),
                    "channel_key": channel_key,
                }
            )
    if not common_channels:
        warnings.append(
            {
                "code": "no_common_channels",
                "message": "The group members have no channel key in common.",
            }
        )
    return (
        channel_records,
        original_channel_index,
        compatibility,
        common_channels,
        partial_channels,
    )


def build_data_group(
    data_manifest_paths: list[str],
    output_dir: str,
    relationship: str,
    labels: list[str] | None,
    reference_data_index: int,
    registration: str,
    registration_channel: str,
    relationship_source: str,
) -> dict[str, Any]:
    """Build one group manifest from two or more single-file ingest manifests."""

    (
        relationship,
        relationship_source,
        reference_data_index,
        registration,
    ) = _normalise_group_options(
        data_manifest_paths,
        relationship,
        relationship_source,
        reference_data_index,
        registration,
    )
    loaded, member_labels = _load_group_inputs(
        data_manifest_paths,
        labels,
        relationship,
    )
    members, records_per_member, warnings = _build_group_members(
        loaded,
        member_labels,
    )
    (
        channel_records,
        original_channel_index,
        compatibility,
        common_channels,
        partial_channels,
    ) = _build_group_channel_indexes(
        members,
        records_per_member,
        warnings,
    )

    out_dir = ensure_output_dir(output_dir)
    if len(member_labels) == 2 and relationship == "ordered_conditions":
        source_stem = safe_stem(
            f"{member_labels[0]}_vs_{member_labels[1]}", fallback="data"
        )
    else:
        source_stem = safe_stem(
            f"{member_labels[0]}_plus_{len(member_labels) - 1}", fallback="data"
        )

    if registration == "translation":
        (
            channel_index,
            registration_payload,
            registered_paths,
            registered_metadata_paths,
        ) = _register_members(
            out_dir=out_dir,
            source_stem=source_stem,
            members=members,
            reference_data_index=int(reference_data_index),
            registration_channel=registration_channel,
            channel_index=original_channel_index,
            channel_records=channel_records,
            compatibility=compatibility,
            warnings=warnings,
        )
    else:
        channel_index = copy.deepcopy(original_channel_index)
        registered_paths = []
        registered_metadata_paths = []
        reference_member_id = members[int(reference_data_index)]["member_id"]
        registration_payload = {
            "requested": "none",
            "applied": "none",
            "registration_channel": registration_channel,
            "reference_data_index": int(reference_data_index),
            "reference_member_id": reference_member_id,
            "transforms": {},
        }

    ordered = relationship in {"ordered_conditions", "time_series"}
    group_manifest_path = out_dir / f"{source_stem}_data_group.json"
    payload: dict[str, Any] = {
        "status": "ok",
        "step": "data_group_build",
        "schema": DATA_GROUP_SCHEMA,
        "source_stem": source_stem,
        "relationship": {
            "value": relationship,
            "source": relationship_source,
            "ordered": ordered,
            "labels": member_labels,
        },
        "members": members,
        "reference_data_index": int(reference_data_index),
        "reference_member_id": members[int(reference_data_index)]["member_id"],
        "original_channel_index": original_channel_index,
        "channel_index": channel_index,
        "channel_records": channel_records,
        "compatibility": {
            "common_channels": common_channels,
            "partial_channels": partial_channels,
            "by_channel": compatibility,
        },
        "registration": registration_payload,
        "warnings": warnings,
        "summary": {
            "member_count": len(members),
            "common_channel_count": len(common_channels),
            "partial_channel_count": len(partial_channels),
            "warning_count": len(warnings),
        },
        "artifacts": {
            "data_group_manifest_path": group_manifest_path,
            "registered_matrix_paths": registered_paths,
            "registered_metadata_paths": registered_metadata_paths,
        },
    }
    write_json_file(group_manifest_path, payload)
    return payload


__all__ = ["build_data_group"]

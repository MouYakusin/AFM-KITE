"""Line-profile measurement and ordered-pair comparison stage."""

from __future__ import annotations

from typing import Any

import numpy as np

from afm_kite_core.algorithms.line_profiles import measure_profile_windows
from afm_kite_core.contracts import (
    LINE_PROFILE_MEASUREMENT_SCHEMA,
    LINE_PROFILE_SCHEMA,
)
from afm_kite_core.artifacts.writers import (
    read_json_file,
    write_json_file,
)
from afm_kite_core.data_io.paths import ensure_output_dir

from .contracts import (
    safe_profile_name,
    validate_compatible_profiles,
    window_indices,
)


def measure_line_profiles(
    line_profile_json_paths: list[str],
    output_dir: str,
) -> dict[str, Any]:
    """Measure compatible line profiles and compare an ordered pair."""

    if not line_profile_json_paths:
        raise ValueError("line_profile_json_paths must contain at least one path")
    profiles = [
        read_json_file(
            path,
            expected_schema=LINE_PROFILE_SCHEMA,
            required_keys=("line_id", "coordinates", "windows", "values"),
        )
        for path in line_profile_json_paths
    ]
    validate_compatible_profiles(profiles)
    first_payload = profiles[0][1]
    left, feature, right = window_indices(first_payload)

    measurements: dict[str, Any] = {}
    profile_records: list[dict[str, Any]] = []
    for path, payload in profiles:
        label = safe_profile_name(payload.get("profile_label", path.stem))
        if label in measurements:
            raise ValueError(f"Duplicate profile_label: {label}")
        values = np.asarray(payload["values"], dtype=float)
        distance = np.asarray(payload["coordinates"]["distance"], dtype=float)
        measurement = measure_profile_windows(
            values,
            left,
            feature,
            right,
            distance,
        )
        measurement["value_unit"] = payload.get("value_unit", "")
        measurement["distance_unit"] = payload["coordinates"].get(
            "distance_unit", "μm"
        )
        measurements[label] = measurement
        profile_records.append(
            {
                "profile_label": label,
                "line_profile_json_path": path,
                "matrix_data_path": payload.get("matrix_data_path", ""),
                "value_unit": payload.get("value_unit", ""),
            }
        )

    comparisons: dict[str, Any] = {}
    if len(profile_records) == 2:
        first_label = profile_records[0]["profile_label"]
        second_label = profile_records[1]["profile_label"]
        first_unit = str(profile_records[0].get("value_unit") or "")
        second_unit = str(profile_records[1].get("value_unit") or "")
        if first_unit and second_unit and first_unit != second_unit:
            raise ValueError(
                "Cannot compare profiles with different units: "
                f"{first_unit}, {second_unit}"
            )
        first_measurement = measurements[first_label]
        second_measurement = measurements[second_label]
        comparisons[f"{second_label}_minus_{first_label}"] = {
            "feature_minus_substrate_median": float(
                second_measurement["feature_minus_substrate_median"]
                - first_measurement["feature_minus_substrate_median"]
            ),
            "absolute_feature_substrate_contrast": float(
                second_measurement["absolute_feature_substrate_contrast"]
                - first_measurement["absolute_feature_substrate_contrast"]
            ),
            "feature_robust_extreme_minus_substrate": float(
                second_measurement["feature_robust_extreme_minus_substrate"]
                - first_measurement["feature_robust_extreme_minus_substrate"]
            ),
            "value_unit": second_unit or first_unit,
        }

    labels = [record["profile_label"] for record in profile_records]
    source_stem = safe_profile_name(
        f"{first_payload.get('source_stem', profiles[0][0].stem)}_"
        + "_vs_".join(labels)
    )
    out_dir = ensure_output_dir(output_dir)
    measurement_json = out_dir / f"{source_stem}_line_measurement.json"
    payload: dict[str, Any] = {
        "status": "ok",
        "step": "line_profile_measure",
        "schema": LINE_PROFILE_MEASUREMENT_SCHEMA,
        "line_id": first_payload.get("line_id", ""),
        "line": first_payload.get("line", {}),
        "windows": first_payload["windows"],
        "profiles": profile_records,
        "measurements": measurements,
        "comparisons": comparisons,
        "artifacts": {
            "line_profile_measurement_json": measurement_json,
            "line_profile_json_paths": [path for path, _ in profiles],
        },
    }
    write_json_file(measurement_json, payload)
    return payload


__all__ = ["measure_line_profiles"]

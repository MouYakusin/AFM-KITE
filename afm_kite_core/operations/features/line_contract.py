"""Feature-crossing line-contract construction stage."""

from __future__ import annotations

from typing import Any

import numpy as np

from afm_kite_core.algorithms.feature_selection import (
    line_direction,
    line_windows,
    rotate_direction,
    segment_labels,
)
from afm_kite_core.contracts import (
    FEATURE_LINE_SCHEMA,
    SELECTED_FEATURE_SCHEMA,
    coordinate_line_id,
)
from afm_kite_core.artifacts.writers import (
    read_json_file,
    safe_stem,
    write_json_file,
)
from afm_kite_core.data_io.paths import ensure_output_dir, resolve_readable_file


def feature_line_build(
    feature_json_path: str,
    output_dir: str,
    background_margin_px: int,
    direction_mode: str,
    angle_degrees: float | None,
    min_substrate_samples: int,
) -> dict[str, Any]:
    """Build one substrate-feature-substrate line from selected-feature JSON."""

    feature_path = resolve_readable_file(feature_json_path)
    feature_payload = read_json_file(
        feature_path,
        expected_schema=SELECTED_FEATURE_SCHEMA,
        required_keys=(
            "selected_feature",
            "artifacts.selected_feature_mask_npy_path",
            "artifacts.feature_labels_npy_path",
        ),
    )[1]
    artifacts = feature_payload.get("artifacts", {})
    selected_mask_path = artifacts.get("selected_feature_mask_npy_path")
    if not selected_mask_path:
        raise ValueError(
            "Selected-feature JSON must contain "
            "artifacts.selected_feature_mask_npy_path"
        )
    selected_mask = np.load(resolve_readable_file(selected_mask_path)).astype(bool)
    labels_path = artifacts.get("feature_labels_npy_path")
    if not labels_path:
        raise ValueError(
            "Selected-feature JSON must contain artifacts.feature_labels_npy_path"
        )
    region_labels = np.load(resolve_readable_file(labels_path)).astype(np.int32)
    if region_labels.shape != selected_mask.shape:
        raise ValueError(
            "Feature label shape does not match selected-feature mask shape"
        )

    selected_feature = feature_payload.get("selected_feature")
    if selected_feature and selected_feature.get("centroid_yx"):
        center_y, center_x = selected_feature["centroid_yx"]
        center_y, center_x = float(center_y), float(center_x)
    else:
        coords = np.argwhere(selected_mask)
        if coords.size == 0:
            raise ValueError("Selected-feature mask is empty")
        center_y, center_x = [float(item) for item in np.mean(coords, axis=0)]
        selected_feature = {
            "centroid_yx": [center_y, center_x],
            "area_px": int(selected_mask.sum()),
        }

    selected_label = int(selected_feature.get("label_value", 0))
    if selected_label <= 0:
        selected_values = region_labels[selected_mask]
        selected_values = selected_values[selected_values > 0]
        if selected_values.size == 0:
            raise ValueError("Could not resolve the selected positive region label")
        selected_label = int(np.bincount(selected_values).argmax())

    base_dx, base_dy = line_direction(
        selected_mask,
        direction_mode,
        angle_degrees,
    )
    offsets = (
        (0.0, -15.0, 15.0, -30.0, 30.0)
        if direction_mode in {"minor_axis", "major_axis"}
        else (0.0,)
    )
    valid_candidates: list[tuple[float, float, float, dict[str, Any]]] = []
    failures: list[str] = []
    for offset in offsets:
        dx, dy = rotate_direction(base_dx, base_dy, offset)
        try:
            candidate = line_windows(
                selected_mask,
                region_labels,
                selected_label,
                center_x,
                center_y,
                dx,
                dy,
                background_margin_px,
            )
            if (
                candidate["left_indices"].size < int(min_substrate_samples)
                or candidate["right_indices"].size < int(min_substrate_samples)
            ):
                raise ValueError(
                    "Line does not meet the minimum substrate sample count"
                )
            score = float(
                min(
                    candidate["left_indices"].size,
                    candidate["right_indices"].size,
                )
                - (0.01 * candidate["feature_indices"].size)
            )
            valid_candidates.append((score, dx, dy, candidate))
        except ValueError as exc:
            failures.append(f"offset {offset:g}: {exc}")
    if not valid_candidates:
        raise ValueError(
            "Could not construct a clean substrate-feature-substrate line; "
            + "; ".join(failures)
        )

    _, dx, dy, windows = max(valid_candidates, key=lambda item: item[0])
    labels = segment_labels(
        windows["x_px"].size,
        windows["left_indices"],
        windows["right_indices"],
        windows["feature_indices"],
    )
    source_stem = safe_stem(
        f"{feature_payload.get('source_stem', feature_path.stem)}_feature_line",
        fallback="matrix",
    )
    out_dir = ensure_output_dir(output_dir)
    line_json = out_dir / f"{source_stem}_line_contract.json"
    current_line_id = coordinate_line_id(
        selected_mask.shape,
        windows["x_px"],
        windows["y_px"],
    )
    line_record = {
        "line_id": current_line_id,
        "start_xy_px": [float(windows["x_px"][0]), float(windows["y_px"][0])],
        "end_xy_px": [float(windows["x_px"][-1]), float(windows["y_px"][-1])],
        "center_xy_px": [center_x, center_y],
        "direction_xy": [dx, dy],
        "direction_mode": direction_mode,
        "feature_window_indices": [
            int(windows["feature_start_index"]),
            int(windows["feature_end_index"]),
        ],
        "left_substrate_indices": [
            int(windows["left_indices"][0]),
            int(windows["left_indices"][-1]),
        ],
        "right_substrate_indices": [
            int(windows["right_indices"][0]),
            int(windows["right_indices"][-1]),
        ],
    }
    payload: dict[str, Any] = {
        "status": "ok",
        "step": "feature_line_build",
        "schema": FEATURE_LINE_SCHEMA,
        "line_id": current_line_id,
        "feature_json_path": feature_path,
        "source_stem": source_stem,
        "shape": list(selected_mask.shape),
        "selected_feature": selected_feature,
        "line": line_record,
        "validation": windows["validation"],
        "windows": {
            "line_start_index": int(windows["line_start_index"]),
            "line_end_index": int(windows["line_end_index"]),
            "feature_window_indices": [
                int(item) for item in windows["feature_indices"]
            ],
            "left_substrate_indices": [
                int(item) for item in windows["left_indices"]
            ],
            "right_substrate_indices": [
                int(item) for item in windows["right_indices"]
            ],
        },
        "coordinates": {
            "t_values": windows["t_values"],
            "x_px": windows["x_px"],
            "y_px": windows["y_px"],
            "source_indices": windows["indices"],
            "segment": labels.tolist(),
        },
        "artifacts": {
            "line_contract_json_path": line_json,
            "selected_feature_mask_npy_path": resolve_readable_file(
                selected_mask_path
            ),
            "feature_labels_npy_path": resolve_readable_file(labels_path),
        },
    }
    write_json_file(line_json, payload)
    return payload


__all__ = ["feature_line_build"]

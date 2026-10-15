"""Residual-feature candidate segmentation stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import closing, disk

from afm_kite_core.algorithms.morphology import remove_objects_smaller_than
from afm_kite_core.algorithms.statistics import robust_location_scale
from afm_kite_core.contracts import (
    RESIDUAL_CANDIDATES_SCHEMA,
    RESIDUAL_PREPARATION_SCHEMA,
)
from afm_kite_core.data_io.matrix import load_height_data
from afm_kite_core.artifacts.plotting import PLOT_DPI, plt, save_figure
from afm_kite_core.artifacts.writers import (
    atomic_write_path,
    read_json_file,
    write_json_file,
)
from afm_kite_core.data_io.paths import ensure_output_dir, resolve_readable_file


def segment_defect_candidates(
    prepared_json_path: str,
    residual_data_path: str,
    leveled_data_path: str,
    output_dir: str,
    polarity: str,
    robust_sigma_threshold: float,
    min_area_px: int,
    min_depth: float,
    min_global_range_fraction: float,
) -> dict[str, Any]:
    """Segment reusable defect candidate components from a residual height map."""

    if polarity not in {"dark", "bright", "both"}:
        raise ValueError("polarity must be dark, bright, or both")

    prepared_payload: dict[str, Any] = {}
    if prepared_json_path:
        _prepared_path, prepared_payload = read_json_file(
            prepared_json_path,
            expected_schema=RESIDUAL_PREPARATION_SCHEMA,
            required_keys=(
                "artifacts.leveled_npy_path",
                "artifacts.residual_npy_path",
            ),
        )
        artifacts = prepared_payload.get("artifacts", {})
        residual_data_path = residual_data_path or artifacts.get(
            "residual_npy_path", ""
        )
        leveled_data_path = leveled_data_path or artifacts.get(
            "leveled_npy_path", ""
        )

    if not residual_data_path:
        raise ValueError("Use prepared_json_path or residual_data_path")

    residual = load_height_data(residual_data_path)
    leveled = load_height_data(leveled_data_path) if leveled_data_path else residual
    out_dir = ensure_output_dir(output_dir)
    stem = prepared_payload.get("source_stem") or Path(residual_data_path).stem.replace(
        "_defect_depth_residual", ""
    )

    finite = np.isfinite(residual)
    if not np.any(finite):
        raise ValueError("Cannot segment an all-NaN residual matrix")

    residual_median, residual_sigma = robust_location_scale(residual[finite])
    leveled_finite = leveled[np.isfinite(leveled)]
    global_range = (
        float(np.max(leveled_finite) - np.min(leveled_finite))
        if leveled_finite.size
        else 0.0
    )
    depth_threshold = max(
        float(min_depth),
        float(robust_sigma_threshold) * residual_sigma,
        global_range * max(float(min_global_range_fraction), 0.0),
    )

    candidate_specs: list[tuple[str, np.ndarray]] = []
    if polarity in {"dark", "both"}:
        candidate_specs.append(
            ("dark", residual <= residual_median - depth_threshold)
        )
    if polarity in {"bright", "both"}:
        candidate_specs.append(
            ("bright", residual >= residual_median + depth_threshold)
        )

    defect_label = np.zeros(residual.shape, dtype=np.int32)
    candidates: list[dict[str, Any]] = []
    candidate_index = 1
    min_component_area = max(int(min_area_px), 1)
    for defect_type, mask in candidate_specs:
        mask = np.asarray(mask & finite, dtype=bool)
        mask = closing(mask, disk(1))
        mask = remove_objects_smaller_than(mask, min_component_area)
        component_labels = label(mask)
        for prop in regionprops(component_labels):
            component_mask = component_labels == prop.label
            area_px = int(np.sum(component_mask))
            if area_px < min_component_area:
                continue
            min_row, min_col, max_row, max_col = prop.bbox
            defect_label[component_mask] = candidate_index
            candidates.append(
                {
                    "candidate_id": f"candidate_{candidate_index:03d}",
                    "label_value": candidate_index,
                    "type": defect_type,
                    "bbox_xyxy": [
                        int(min_col),
                        int(min_row),
                        int(max_col),
                        int(max_row),
                    ],
                    "centroid_yx": [
                        float(prop.centroid[0]),
                        float(prop.centroid[1]),
                    ],
                    "area_px": area_px,
                }
            )
            candidate_index += 1

    label_npy = out_dir / f"{stem}_defects_label.npy"
    mask_png = out_dir / f"{stem}_defects_mask.png"
    candidates_json = out_dir / f"{stem}_defect_candidates.json"
    atomic_write_path(
        label_npy,
        lambda path: np.save(path, defect_label.astype(np.int32)),
    )

    fig, ax = plt.subplots(figsize=(5, 5), dpi=PLOT_DPI)
    ax.imshow(defect_label, cmap="magma")
    ax.set_axis_off()
    fig.tight_layout()
    atomic_write_path(
        mask_png,
        lambda path: save_figure(fig, path, pad_inches=0.02),
    )
    plt.close(fig)

    defect_type_counts = {
        defect_type: sum(1 for item in candidates if item["type"] == defect_type)
        for defect_type in ("dark", "bright")
    }
    payload: dict[str, Any] = {
        "status": "ok",
        "step": "segment_defect_candidates",
        "schema": RESIDUAL_CANDIDATES_SCHEMA,
        "prepared_json_path": (
            str(resolve_readable_file(prepared_json_path))
            if prepared_json_path
            else ""
        ),
        "residual_data_path": str(resolve_readable_file(residual_data_path)),
        "leveled_data_path": (
            str(resolve_readable_file(leveled_data_path))
            if leveled_data_path
            else ""
        ),
        "source_stem": stem,
        "polarity": polarity,
        "robust_sigma_threshold": float(robust_sigma_threshold),
        "residual_median": residual_median,
        "residual_robust_sigma": residual_sigma,
        "global_height_range": global_range,
        "depth_threshold": depth_threshold,
        "min_area_px": min_component_area,
        "min_depth": float(min_depth),
        "min_global_range_fraction": float(min_global_range_fraction),
        "candidate_count": len(candidates),
        "defect_type_counts": defect_type_counts,
        "candidates": candidates,
        "artifacts": {
            "defect_label_npy_path": label_npy,
            "defect_mask_png_path": mask_png,
            "candidates_json_path": candidates_json,
        },
    }
    write_json_file(candidates_json, payload)
    return payload


__all__ = ["segment_defect_candidates"]

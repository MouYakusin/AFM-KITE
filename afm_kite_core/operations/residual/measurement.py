"""Residual-feature local-reference measurement stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from afm_kite_core.algorithms.statistics import image_clip_limits
from afm_kite_core.contracts import (
    RESIDUAL_CANDIDATES_SCHEMA,
    RESIDUAL_MEASUREMENT_SCHEMA,
    RESIDUAL_PREPARATION_SCHEMA,
)
from afm_kite_core.data_io.matrix import load_height_data
from afm_kite_core.artifacts.plotting import (
    PLOT_DPI,
    PSEUDOCOLOR_FIGURE_SIZE,
    decorate_height_map,
    plt,
    render_height_image,
    resolve_afm_colormap,
    save_figure,
    set_plot_title,
)
from afm_kite_core.serialization import to_jsonable
from afm_kite_core.artifacts.writers import (
    atomic_write_path,
    read_json_file,
    write_json_file,
)
from afm_kite_core.data_io.paths import ensure_output_dir, resolve_readable_file

from .reports import write_defect_depth_report


def _render_residual_candidate_overlay(
    *,
    output_path: Path,
    leveled: np.ndarray,
    defect_label: np.ndarray,
    all_candidates: list[dict[str, Any]],
    selected_label_values: set[int],
    physical_context: dict[str, Any],
    height_unit: str,
    colormap: str,
    mode: str,
) -> dict[str, Any]:
    """Render either the gray candidate review or selected-candidate overlay."""

    if mode not in {"all_candidates_gray", "selected_candidate"}:
        raise ValueError(
            "overlay mode must be all_candidates_gray or selected_candidate"
        )

    vmin, vmax = image_clip_limits(leveled)
    fig, ax = plt.subplots(figsize=PSEUDOCOLOR_FIGURE_SIZE, dpi=PLOT_DPI)
    image = render_height_image(
        ax,
        leveled,
        cmap=resolve_afm_colormap(colormap),
        vmin=vmin,
        vmax=vmax,
    )
    outlined_candidate_ids: list[str] = []
    labeled_candidate_ids: list[str] = []
    for candidate in all_candidates:
        label_value = int(candidate["label_value"])
        component_mask = defect_label == label_value
        if not np.any(component_mask):
            continue
        candidate_id = str(
            candidate.get("candidate_id") or f"candidate_{label_value:03d}"
        )
        outlined_candidate_ids.append(candidate_id)
        is_selected = label_value in selected_label_values
        if mode == "all_candidates_gray":
            outline_color = "#707070"
            outline_width = 0.7
            should_label = True
        else:
            outline_color = "cyan" if is_selected else "#8A8A8A"
            outline_width = 1.4 if is_selected else 0.45
            should_label = is_selected
        ax.contour(
            component_mask,
            levels=[0.5],
            colors=outline_color,
            linewidths=outline_width,
            alpha=1.0 if is_selected else 0.9,
        )
        centroid = candidate.get("centroid_yx")
        if not should_label or not centroid:
            continue
        labeled_candidate_ids.append(candidate_id)
        y, x = centroid
        ax.text(
            x,
            y,
            f"C{label_value:03d}",
            color="black" if is_selected and mode != "all_candidates_gray" else "#404040",
            fontsize=6 if is_selected and mode != "all_candidates_gray" else 4.5,
            fontweight=(
                "bold" if is_selected and mode != "all_candidates_gray" else "normal"
            ),
            ha="center",
            va="center",
            bbox={
                "facecolor": (
                    "cyan"
                    if is_selected and mode != "all_candidates_gray"
                    else "#B8B8B8"
                ),
                "alpha": 0.78 if is_selected else 0.68,
                "pad": 1.0 if is_selected else 0.55,
                "linewidth": 0,
            },
        )

    set_plot_title(ax, "Height")
    decorations = decorate_height_map(
        fig,
        ax,
        image,
        matrix_shape=leveled.shape,
        physical_context=physical_context,
        value_label="Height",
        value_unit=height_unit,
    )
    decorations["candidate_overlay"] = {
        "mode": mode,
        "candidate_count": len(all_candidates),
        "label_format": "C###",
        "outlined_candidate_ids": outlined_candidate_ids,
        "labeled_candidate_ids": labeled_candidate_ids,
        "selected_label_values": sorted(selected_label_values),
        "unselected_outline": {
            "color": "#707070" if mode == "all_candidates_gray" else "#8A8A8A",
            "linewidth": 0.7 if mode == "all_candidates_gray" else 0.45,
        },
        "selected_outline": (
            {
                "color": "cyan",
                "linewidth": 1.4,
            }
            if mode == "selected_candidate"
            else None
        ),
    }
    fig.tight_layout()
    atomic_write_path(
        output_path,
        lambda path: save_figure(fig, path, pad_inches=0.02),
    )
    plt.close(fig)
    return decorations


def measure_residual_features(
    prepared_json_path: str,
    candidates_json_path: str,
    output_dir: str,
    annulus_radius_px: int,
    min_depth: float,
    colormap: str,
    selection_mode: str,
    render_overlay: bool,
    render_candidates_overlay: bool,
    render_selected_overlay: bool,
) -> dict[str, Any]:
    """Measure local-reference depth for segmented residual-feature candidates."""

    if selection_mode not in {"all", "largest_area"}:
        raise ValueError("selection_mode must be all or largest_area")

    prepared_path, prepared_payload = read_json_file(
        prepared_json_path,
        expected_schema=RESIDUAL_PREPARATION_SCHEMA,
        required_keys=("artifacts.leveled_npy_path",),
    )
    candidates_path, candidates_payload = read_json_file(
        candidates_json_path,
        expected_schema=RESIDUAL_CANDIDATES_SCHEMA,
        required_keys=("artifacts.defect_label_npy_path", "candidates"),
    )

    artifacts = prepared_payload.get("artifacts", {})
    leveled = load_height_data(artifacts["leveled_npy_path"])
    defect_label = np.load(
        resolve_readable_file(
            candidates_payload["artifacts"]["defect_label_npy_path"]
        )
    )
    out_dir = ensure_output_dir(output_dir)
    stem = prepared_payload.get("source_stem") or Path(
        artifacts["leveled_npy_path"]
    ).stem.replace("_defect_depth_leveled", "")

    finite = np.isfinite(leveled)
    if not np.any(finite):
        raise ValueError("Cannot measure defects on an all-NaN leveled matrix")

    physical = prepared_payload.get("physical_context", {}) or {}
    pixel_area = physical.get("pixel_area")
    pixel_width = physical.get("pixel_width")
    pixel_height = physical.get("pixel_height")
    height_unit = prepared_payload.get("height_unit", "μm")
    area_unit = prepared_payload.get("area_unit", "μm^2")

    defects: list[dict[str, Any]] = []
    candidate_union = defect_label > 0
    min_reference_px = max(8, int(candidates_payload.get("min_area_px", 1)))
    annulus_iterations = max(int(annulus_radius_px), 1)
    all_candidates = list(candidates_payload.get("candidates", []))
    selected_candidates = all_candidates
    if selection_mode == "largest_area" and all_candidates:
        selected_candidates = [
            min(
                all_candidates,
                key=lambda item: (
                    -int(item.get("area_px", 0)),
                    int(item["label_value"]),
                ),
            )
        ]
    candidates_by_label = {
        int(item["label_value"]): item for item in selected_candidates
    }
    selected_label_values = set(candidates_by_label)
    selected_candidate_ids = [
        str(item.get("candidate_id", "")) for item in selected_candidates
    ]

    for label_value, candidate in candidates_by_label.items():
        component_mask = defect_label == label_value
        area_px = int(np.sum(component_mask))
        if area_px == 0:
            continue
        component_values = leveled[component_mask]
        component_values = component_values[np.isfinite(component_values)]
        if component_values.size == 0:
            continue

        dilated = ndi.binary_dilation(
            component_mask,
            iterations=annulus_iterations,
        )
        reference_mask = dilated & ~component_mask & ~candidate_union & finite
        if int(np.sum(reference_mask)) < min_reference_px:
            reference_mask = (~candidate_union) & finite
        if int(np.sum(reference_mask)) < min_reference_px:
            reference_mask = (~component_mask) & finite
        reference_values = leveled[reference_mask]
        reference_values = reference_values[np.isfinite(reference_values)]
        if reference_values.size == 0:
            reference_values = leveled[finite]

        defect_type = str(candidate.get("type", "dark"))
        local_reference_median = float(np.median(reference_values))
        local_reference_mean = float(np.mean(reference_values))
        coords = np.argwhere(component_mask)
        if defect_type == "dark":
            extremum_index = int(np.argmin(leveled[component_mask]))
            extremum_height = float(np.min(component_values))
            depth_values = local_reference_median - component_values
            max_depth = float(local_reference_median - extremum_height)
        else:
            extremum_index = int(np.argmax(leveled[component_mask]))
            extremum_height = float(np.max(component_values))
            depth_values = component_values - local_reference_median
            max_depth = float(extremum_height - local_reference_median)

        if max_depth < max(float(min_depth), 0.0):
            continue

        extremum_y, extremum_x = coords[extremum_index]
        record: dict[str, Any] = {
            "defect_id": f"defect_{len(defects) + 1:03d}",
            "candidate_id": candidate.get("candidate_id"),
            "candidate_label_value": label_value,
            "type": defect_type,
            "bbox_xyxy": candidate.get("bbox_xyxy"),
            "centroid_yx": candidate.get("centroid_yx"),
            "extremum_yx": [int(extremum_y), int(extremum_x)],
            "area_px": area_px,
            "area_physical": float(area_px * pixel_area) if pixel_area else None,
            "area_unit": area_unit,
            "local_reference_median": local_reference_median,
            "local_reference_mean": local_reference_mean,
            "defect_mean_height": float(np.mean(component_values)),
            "defect_median_height": float(np.median(component_values)),
            "defect_min_height": float(np.min(component_values)),
            "defect_max_height": float(np.max(component_values)),
            "extremum_height": extremum_height,
            "max_depth": max_depth,
            "mean_depth": float(np.mean(depth_values)),
            "median_depth": float(np.median(depth_values)),
            "p95_depth": float(np.percentile(depth_values, 95.0)),
            "height_unit": height_unit,
            "reference_pixel_count": int(reference_values.size),
        }
        if pixel_width and pixel_height and candidate.get("centroid_yx"):
            centroid_y, centroid_x = candidate["centroid_yx"]
            record["centroid_xy_physical"] = [
                float(centroid_x * pixel_width),
                float(centroid_y * pixel_height),
            ]
            record["extremum_xy_physical"] = [
                float(extremum_x * pixel_width),
                float(extremum_y * pixel_height),
            ]
        defects.append(record)

    defects.sort(key=lambda item: item["max_depth"], reverse=True)
    analysis_json = out_dir / f"{stem}_defect_depth_analysis.json"
    report_md = out_dir / f"{stem}_defect_depth_report.md"
    candidates_overlay_png: Path | None = None
    selected_overlay_png: Path | None = None
    candidates_overlay_decorations: dict[str, Any] = {}
    selected_overlay_decorations: dict[str, Any] = {}
    should_render_candidates = bool(
        render_overlay or render_candidates_overlay
    )
    should_render_selected = bool(render_overlay or render_selected_overlay)
    if should_render_candidates:
        candidates_overlay_png = (
            out_dir / f"{stem}_defect_candidates_overlay.png"
        )
        candidates_overlay_decorations = _render_residual_candidate_overlay(
            output_path=candidates_overlay_png,
            leveled=leveled,
            defect_label=defect_label,
            all_candidates=all_candidates,
            selected_label_values=selected_label_values,
            physical_context=physical,
            height_unit=height_unit,
            colormap=colormap,
            mode="all_candidates_gray",
        )
    if should_render_selected:
        selected_overlay_png = out_dir / f"{stem}_defects_overlay.png"
        selected_overlay_decorations = _render_residual_candidate_overlay(
            output_path=selected_overlay_png,
            leveled=leveled,
            defect_label=defect_label,
            all_candidates=all_candidates,
            selected_label_values=selected_label_values,
            physical_context=physical,
            height_unit=height_unit,
            colormap=colormap,
            mode="selected_candidate",
        )

    defect_type_counts = {
        defect_type: sum(1 for item in defects if item["type"] == defect_type)
        for defect_type in ("dark", "bright")
    }
    output_artifacts = {
        **artifacts,
        **candidates_payload.get("artifacts", {}),
        "analysis_json_path": analysis_json,
        "report_md_path": report_md,
    }
    if candidates_overlay_png is not None:
        output_artifacts["candidate_overlay_png_path"] = candidates_overlay_png
    if selected_overlay_png is not None:
        output_artifacts["defect_overlay_png_path"] = selected_overlay_png

    payload: dict[str, Any] = {
        "status": "ok",
        "step": "measure_residual_features",
        "schema": RESIDUAL_MEASUREMENT_SCHEMA,
        "input_path": prepared_payload.get("input_path", ""),
        "prepared_json_path": str(prepared_path),
        "candidates_json_path": str(candidates_path),
        "matrix_shape": prepared_payload.get("matrix_shape", list(leveled.shape)),
        "metadata_summary": prepared_payload.get("metadata_summary", {}),
        "polarity": candidates_payload.get("polarity"),
        "leveling": prepared_payload.get("leveling", {}),
        "height_unit": height_unit,
        "area_unit": area_unit,
        "background_sigma_px": prepared_payload.get("background_sigma_px"),
        "annulus_radius_px": annulus_iterations,
        "robust_sigma_threshold": candidates_payload.get(
            "robust_sigma_threshold"
        ),
        "residual_median": candidates_payload.get("residual_median"),
        "residual_robust_sigma": candidates_payload.get("residual_robust_sigma"),
        "global_height_range": candidates_payload.get("global_height_range"),
        "depth_threshold": candidates_payload.get("depth_threshold"),
        "min_area_px": candidates_payload.get("min_area_px"),
        "candidate_count": candidates_payload.get("candidate_count"),
        "selection_mode": selection_mode,
        "selected_candidate_count": len(selected_candidates),
        "selected_candidate_ids": selected_candidate_ids,
        "render_overlay": bool(
            should_render_candidates or should_render_selected
        ),
        "render_candidates_overlay": should_render_candidates,
        "render_selected_overlay": should_render_selected,
        "defect_count": len(defects),
        "defect_type_counts": defect_type_counts,
        "defects": defects,
        "artifacts": output_artifacts,
        "candidate_overlay_plot_decorations": candidates_overlay_decorations,
        "selected_overlay_plot_decorations": selected_overlay_decorations,
        "overlay_plot_decorations": (
            selected_overlay_decorations or candidates_overlay_decorations
        ),
    }
    write_json_file(analysis_json, payload)
    write_defect_depth_report(report_md, to_jsonable(payload))
    return payload


__all__ = ["measure_residual_features"]

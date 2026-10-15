"""Ranked feature-selection stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from skimage.measure import regionprops

from afm_kite_core.algorithms.feature_selection import (
    feature_candidate_record,
    rank_feature_candidates,
)
from afm_kite_core.algorithms.statistics import image_clip_limits
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
from afm_kite_core.contracts import SELECTED_FEATURE_SCHEMA, VALUE_REGIONS_SCHEMA
from afm_kite_core.data_io.matrix import (
    height_unit,
    load_height_data_and_metadata,
    physical_area_context_from_metadata,
)
from afm_kite_core.artifacts.writers import (
    atomic_write_path,
    read_json_file,
    safe_stem,
    write_json_file,
)
from afm_kite_core.data_io.paths import ensure_output_dir, resolve_readable_file


def _render_feature_overlay(
    *,
    output_path: Path,
    matrix: np.ndarray,
    labels: np.ndarray,
    candidate_label_values: list[int],
    selected_label_value: int,
    physical_context: dict[str, Any],
    value_unit: str,
    colormap: str,
    mode: str,
) -> dict[str, Any]:
    """Render all candidate regions or emphasize the selected feature."""

    if mode not in {"all_candidates_gray", "selected_feature"}:
        raise ValueError(
            "feature overlay mode must be all_candidates_gray or selected_feature"
        )
    vmin, vmax = image_clip_limits(matrix)
    fig, ax = plt.subplots(figsize=PSEUDOCOLOR_FIGURE_SIZE, dpi=PLOT_DPI)
    image = render_height_image(
        ax,
        matrix,
        cmap=resolve_afm_colormap(colormap),
        vmin=vmin,
        vmax=vmax,
    )
    outlined_feature_ids: list[str] = []
    labeled_feature_ids: list[str] = []
    for label_value in candidate_label_values:
        component_mask = labels == int(label_value)
        if not np.any(component_mask):
            continue
        feature_id = f"F{int(label_value):03d}"
        outlined_feature_ids.append(feature_id)
        is_selected = int(label_value) == int(selected_label_value)
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
        if not should_label:
            continue
        coordinates = np.argwhere(component_mask)
        center_y, center_x = np.mean(coordinates, axis=0)
        labeled_feature_ids.append(feature_id)
        ax.text(
            center_x,
            center_y,
            feature_id,
            color=(
                "black"
                if is_selected and mode == "selected_feature"
                else "#404040"
            ),
            fontsize=6 if is_selected and mode == "selected_feature" else 4.5,
            fontweight=(
                "bold"
                if is_selected and mode == "selected_feature"
                else "normal"
            ),
            ha="center",
            va="center",
            bbox={
                "facecolor": (
                    "cyan"
                    if is_selected and mode == "selected_feature"
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
        matrix_shape=matrix.shape,
        physical_context=physical_context,
        value_label="Height",
        value_unit=value_unit,
    )
    decorations["feature_overlay"] = {
        "mode": mode,
        "candidate_count": len(candidate_label_values),
        "label_format": "F###",
        "outlined_feature_ids": outlined_feature_ids,
        "labeled_feature_ids": labeled_feature_ids,
        "selected_label_value": int(selected_label_value),
        "unselected_outline": {
            "color": "#707070" if mode == "all_candidates_gray" else "#8A8A8A",
            "linewidth": 0.7 if mode == "all_candidates_gray" else 0.45,
        },
        "selected_outline": (
            {"color": "cyan", "linewidth": 1.4}
            if mode == "selected_feature"
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


def feature_select(
    regions_json_path: str,
    matrix_data_path: str,
    output_dir: str,
    selection_metric: str,
    sort_order: str,
    rank: int,
    polarity: str,
    min_area_px: int,
    max_area_px: int,
    min_edge_margin_px: int,
    prominence_reference: str,
    local_ring_inner_px: int,
    local_ring_outer_px: int,
    min_local_background_samples: int,
    render_candidates_overlay: bool,
    render_selected_overlay: bool,
    colormap: str,
) -> dict[str, Any]:
    """Select one ranked target feature from a labeled-region artifact."""

    if prominence_reference not in {"local_ring", "global_background"}:
        raise ValueError(
            "prominence_reference must be local_ring or global_background"
        )
    if int(local_ring_inner_px) < 0:
        raise ValueError("local_ring_inner_px must be non-negative")
    if int(local_ring_outer_px) <= int(local_ring_inner_px):
        raise ValueError(
            "local_ring_outer_px must be greater than local_ring_inner_px"
        )
    if int(min_local_background_samples) < 1:
        raise ValueError("min_local_background_samples must be positive")

    regions_path = resolve_readable_file(regions_json_path)
    regions_payload = read_json_file(
        regions_path,
        expected_schema=VALUE_REGIONS_SCHEMA,
        required_keys=("label_npy_path",),
    )[1]
    label_path = regions_payload.get("label_npy_path")
    if not label_path:
        raise ValueError("regions JSON must contain label_npy_path")

    matrix_path = matrix_data_path or str(regions_payload.get("input_path", ""))
    if not matrix_path:
        raise ValueError(
            "feature-select requires --matrix-data-path when regions JSON "
            "has no input_path"
        )

    labels_file = resolve_readable_file(label_path)
    matrix, metadata, matrix_file = load_height_data_and_metadata(
        matrix_path,
        str(regions_payload.get("metadata_path") or ""),
    )
    physical_context = (
        regions_payload.get("physical_context")
        or physical_area_context_from_metadata(metadata)
    )
    value_unit = height_unit(metadata)
    labels = np.load(labels_file).astype(np.int32)
    if labels.shape != matrix.shape:
        raise ValueError(
            f"Label shape {labels.shape} does not match matrix shape {matrix.shape}"
        )

    props = regionprops(labels, intensity_image=matrix)
    if not props:
        raise ValueError("No labeled features found in regions JSON")

    finite = np.isfinite(matrix)
    global_background_values = matrix[(labels == 0) & finite]
    if global_background_values.size:
        global_background_median = float(np.median(global_background_values))
        global_background_sample_count = int(global_background_values.size)
    else:
        finite_values = matrix[finite]
        global_background_median = (
            float(np.median(finite_values)) if finite_values.size else 0.0
        )
        global_background_sample_count = int(finite_values.size)

    candidate_records = [
        feature_candidate_record(
            labels,
            matrix,
            int(prop.label),
            global_background_median=global_background_median,
            global_background_sample_count=global_background_sample_count,
            prominence_reference=prominence_reference,
            local_ring_inner_px=local_ring_inner_px,
            local_ring_outer_px=local_ring_outer_px,
            min_local_background_samples=min_local_background_samples,
        )
        for prop in props
    ]
    props_by_label = {int(prop.label): prop for prop in props}
    selected_diag, candidate_ranking = rank_feature_candidates(
        candidate_records,
        selection_metric=selection_metric,
        sort_order=sort_order,
        rank=rank,
        polarity=polarity,
        min_area_px=min_area_px,
        max_area_px=max_area_px,
        min_edge_margin_px=min_edge_margin_px,
    )
    selected_diag["selection_rule"] = (
        f"{selection_metric} {sort_order}, one-based rank {int(rank)}"
    )
    selection_request = {
        "selection_metric": selection_metric,
        "sort_order": sort_order,
        "rank": int(rank),
        "polarity": polarity,
        "min_area_px": int(min_area_px),
        "max_area_px": int(max_area_px),
        "min_edge_margin_px": int(min_edge_margin_px),
        "prominence_reference": prominence_reference,
        "local_ring_inner_px": int(local_ring_inner_px),
        "local_ring_outer_px": int(local_ring_outer_px),
        "min_local_background_samples": int(min_local_background_samples),
    }

    selected = props_by_label[int(selected_diag["label_value"])]
    selected_mask = labels == int(selected.label)
    min_row, min_col, max_row, max_col = selected.bbox
    center_y, center_x = float(selected.centroid[0]), float(selected.centroid[1])
    source_stem = safe_stem(
        regions_path.stem.replace("_regions", ""),
        fallback="matrix",
    )
    out_dir = ensure_output_dir(output_dir)
    selected_mask_npy = out_dir / f"{source_stem}_selected_feature_mask.npy"
    selected_json = out_dir / f"{source_stem}_selected_feature.json"
    atomic_write_path(
        selected_mask_npy,
        lambda path: np.save(path, selected_mask.astype(bool)),
    )
    candidate_overlay_png = (
        out_dir / f"{source_stem}_feature_candidates_overlay.png"
        if render_candidates_overlay
        else None
    )
    selected_overlay_png = (
        out_dir / f"{source_stem}_selected_feature_overlay.png"
        if render_selected_overlay
        else None
    )
    candidate_overlay_decorations: dict[str, Any] = {}
    selected_overlay_decorations: dict[str, Any] = {}
    candidate_label_values = [
        int(candidate["label_value"]) for candidate in candidate_ranking
    ]
    if candidate_overlay_png is not None:
        candidate_overlay_decorations = _render_feature_overlay(
            output_path=candidate_overlay_png,
            matrix=matrix,
            labels=labels,
            candidate_label_values=candidate_label_values,
            selected_label_value=int(selected.label),
            physical_context=physical_context,
            value_unit=value_unit,
            colormap=colormap,
            mode="all_candidates_gray",
        )
    if selected_overlay_png is not None:
        selected_overlay_decorations = _render_feature_overlay(
            output_path=selected_overlay_png,
            matrix=matrix,
            labels=labels,
            candidate_label_values=candidate_label_values,
            selected_label_value=int(selected.label),
            physical_context=physical_context,
            value_unit=value_unit,
            colormap=colormap,
            mode="selected_feature",
        )

    selected_feature = {
        "label_value": int(selected.label),
        "bbox_xyxy": [int(min_col), int(min_row), int(max_col), int(max_row)],
        "centroid_yx": [center_y, center_x],
        "area_px": int(selected.area),
        "mean_value": selected_diag["mean_value"],
        "median_value": selected_diag["median_value"],
        "background_median": selected_diag["background_median"],
        "background_reference_requested": selected_diag[
            "background_reference_requested"
        ],
        "background_reference_used": selected_diag["background_reference_used"],
        "background_sample_count": selected_diag["background_sample_count"],
        "background_fallback_reason": selected_diag["background_fallback_reason"],
        "global_background_median": selected_diag["global_background_median"],
        "global_background_sample_count": selected_diag[
            "global_background_sample_count"
        ],
        "local_background_median": selected_diag["local_background_median"],
        "local_background_sample_count": selected_diag[
            "local_background_sample_count"
        ],
        "local_background_mad": selected_diag["local_background_mad"],
        "median_prominence": selected_diag["median_prominence"],
        "absolute_prominence": abs(float(selected_diag["median_prominence"])),
        "rank": int(selected_diag["rank"]),
        "selection_metric": selected_diag["selection_metric"],
        "selection_metric_value": selected_diag["selection_metric_value"],
        "selection_rule": selected_diag["selection_rule"],
        "edge_margin_px": selected_diag["edge_margin_px"],
        "edge_penalty": selected_diag.get("edge_penalty"),
    }
    output_artifacts: dict[str, Any] = {
        "matrix_data_path": matrix_file,
        "feature_labels_npy_path": labels_file,
        "value_regions_json_path": regions_path,
        "selected_feature_mask_npy_path": selected_mask_npy,
        "selected_feature_json_path": selected_json,
    }
    if candidate_overlay_png is not None:
        output_artifacts[
            "feature_candidates_overlay_png_path"
        ] = candidate_overlay_png
    if selected_overlay_png is not None:
        output_artifacts["selected_feature_overlay_png_path"] = selected_overlay_png

    payload: dict[str, Any] = {
        "status": "ok",
        "step": "feature_select",
        "schema": SELECTED_FEATURE_SCHEMA,
        "regions_json_path": regions_path,
        "matrix_data_path": matrix_file,
        "source_stem": source_stem,
        "selection": {
            **selection_request,
            "candidate_count": len(candidate_records),
            "eligible_count": sum(
                bool(candidate.get("eligible"))
                for candidate in candidate_ranking
            ),
            "candidate_ranking": candidate_ranking,
        },
        "selected_feature": selected_feature,
        "plots_written": bool(
            render_candidates_overlay or render_selected_overlay
        ),
        "render_candidates_overlay": bool(render_candidates_overlay),
        "render_selected_overlay": bool(render_selected_overlay),
        "candidate_overlay_plot_decorations": candidate_overlay_decorations,
        "selected_overlay_plot_decorations": selected_overlay_decorations,
        "segmentation": {
            "source": "feature-select from existing region labels",
            "method": selected_feature["selection_rule"],
            "value_region_segmentation": {
                "step": regions_payload.get("step"),
                "basis": regions_payload.get("basis"),
                "feature_type": regions_payload.get("feature_type"),
                "method": regions_payload.get("method"),
                "selected_polarity": regions_payload.get("selected_polarity"),
                "threshold": regions_payload.get("threshold"),
                "threshold_percentile": regions_payload.get(
                    "threshold_percentile"
                ),
                "region_count": regions_payload.get("region_count"),
                "label_npy_path": labels_file,
                "regions_json_path": regions_path,
            },
        },
        "artifacts": output_artifacts,
    }
    write_json_file(selected_json, payload)
    return payload


__all__ = ["feature_select"]

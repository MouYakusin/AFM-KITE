"""Value-region and local-outlier segmentation workflow."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from skimage.filters import threshold_otsu

from afm_kite_core.algorithms.outliers import detect_local_outlier_components
from afm_kite_core.algorithms.segmentation import (
    build_detrended_value_candidate,
    build_value_candidate_labels,
    build_value_candidate_record,
    regions_from_label_image,
)
from afm_kite_core.algorithms.statistics import image_clip_limits
from afm_kite_core.contracts import LOCAL_OUTLIERS_SCHEMA, VALUE_REGIONS_SCHEMA
from afm_kite_core.data_io.matrix import height_unit as get_height_unit
from afm_kite_core.data_io.matrix import (
    load_height_data_and_metadata,
    physical_area_context_from_metadata,
    resolve_matrix_metadata_path,
)
from afm_kite_core.artifacts.plotting import (
    COLORMAP,
    PLOT_DPI,
    PSEUDOCOLOR_FIGURE_SIZE,
    decorate_height_map,
    plt,
    render_height_image,
    save_figure,
    set_plot_title,
)
from afm_kite_core.artifacts.writers import (
    atomic_write_path,
    read_json_file,
    write_json_file,
)
from afm_kite_core.data_io.paths import ensure_output_dir, resolve_readable_file


def segment_value_regions(
    matrix_data_path: str,
    output_dir: str,
    polarity: str,
    min_area_fraction: float,
    expected_count: int,
    write_plots: bool,
    threshold_percentile: float | None,
    min_area_px: int,
    metadata_path: str,
) -> dict[str, Any]:
    """Segment special AFM regions with Otsu thresholding and morphology.

    ``polarity`` can be ``auto``, ``bright``, or ``dark``.
    In auto mode, the tool scores bright and dark components and keeps the more plausible mask.
    """

    matrix, metadata, resolved_input = load_height_data_and_metadata(
        matrix_data_path,
        metadata_path,
    )
    physical = physical_area_context_from_metadata(metadata)
    value_unit = get_height_unit(metadata)
    out_dir = ensure_output_dir(output_dir)
    stem = resolved_input.stem
    threshold = float(threshold_otsu(matrix[np.isfinite(matrix)]))
    if int(min_area_px) > 0:
        min_area_px = int(min_area_px)
    else:
        min_area_px = max(8, int(matrix.size * float(min_area_fraction)))
    detrend_sigma_px = max(6.0, min(matrix.shape) / 16.0)
    candidates: list[dict[str, Any]] = []

    polarities = ["bright", "dark"] if polarity == "auto" else [polarity]
    if any(item not in {"bright", "dark"} for item in polarities):
        raise ValueError("polarity must be auto, bright, or dark")

    for candidate_polarity in polarities:
        label_image = build_value_candidate_labels(
            matrix, candidate_polarity, threshold, min_area_px)
        candidates.append(
            build_value_candidate_record(
                label_image,
                method="otsu_threshold_morphology",
                polarity=candidate_polarity,
                threshold=threshold,
                expected_count=expected_count,
            )
        )

    if threshold_percentile is not None:
        finite_values = matrix[np.isfinite(matrix)]
        percentiles = [
            float(threshold_percentile),
            max(70.0, float(threshold_percentile) - 8.0),
            max(60.0, float(threshold_percentile) - 16.0),
        ]
        for candidate_polarity in polarities:
            for percentile in percentiles:
                percentile_threshold = max(
                    threshold,
                    float(np.percentile(finite_values, percentile)),
                )
                label_image = build_value_candidate_labels(
                    matrix,
                    candidate_polarity,
                    percentile_threshold,
                    min_area_px,
                )
                candidates.append(
                    build_value_candidate_record(
                        label_image,
                        method="percentile_otsu_threshold_morphology",
                        polarity=candidate_polarity,
                        threshold=percentile_threshold,
                        expected_count=expected_count,
                    )
                )

    for candidate_polarity in polarities:
        detrended_label, detrended_threshold = build_detrended_value_candidate(
            matrix,
            candidate_polarity,
            min_area_px,
            detrend_sigma_px,
        )
        candidates.append(
            build_value_candidate_record(
                detrended_label,
                method="detrended_otsu_threshold_morphology",
                polarity=candidate_polarity,
                threshold=detrended_threshold,
                expected_count=expected_count,
                detrend_sigma_px=detrend_sigma_px,
            )
        )

    passing_candidates = [
        item for item in candidates if item["quality"]["passed"]
    ]
    selected = max(
        passing_candidates or candidates,
        key=lambda item: item["score"],
    )
    selected_polarity = str(selected["polarity"])
    selected_score = float(selected["score"])
    selected_method = str(selected["method"])
    selected_threshold = float(selected["threshold"])
    label_image = np.asarray(selected["label_image"])
    regions = regions_from_label_image(label_image, matrix)
    label_npy = out_dir / f"{stem}_regions_label.npy"
    mask_png = out_dir / f"{stem}_regions_mask.png" if write_plots else None
    overlay_png = out_dir / f"{stem}_regions_overlay.png" if write_plots else None
    regions_json = out_dir / f"{stem}_regions.json"
    atomic_write_path(
        label_npy,
        lambda path: np.save(path, label_image.astype(np.int32)),
    )

    if write_plots:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=PLOT_DPI)
        ax.imshow(label_image, cmap="tab20")
        ax.set_axis_off()
        fig.tight_layout()
        atomic_write_path(
            mask_png,
            lambda path: save_figure(fig, path, pad_inches=0.02),
        )
        plt.close(fig)

        vmin, vmax = image_clip_limits(matrix)
        fig, ax = plt.subplots(figsize=PSEUDOCOLOR_FIGURE_SIZE, dpi=PLOT_DPI)
        image = render_height_image(
            ax,
            matrix,
            cmap=COLORMAP,
            vmin=vmin,
            vmax=vmax,
        )
        if np.any(label_image > 0):
            ax.contour(label_image > 0, levels=[0.5], colors="red", linewidths=0.8)
            for region in regions:
                y, x = region["centroid_yx"]
                ax.text(
                    x,
                    y,
                    str(int(region["label_value"])),
                    color="white",
                    fontsize=6,
                    ha="center",
                    va="center",
                    bbox={"facecolor": "black", "alpha": 0.45, "pad": 1, "linewidth": 0},
                )
        set_plot_title(ax, "Height")
        overlay_decorations = decorate_height_map(
            fig,
            ax,
            image,
            matrix_shape=matrix.shape,
            physical_context=physical,
            value_label="Height",
            value_unit=value_unit,
        )
        fig.tight_layout()
        atomic_write_path(
            overlay_png,
            lambda path: save_figure(fig, path, pad_inches=0.02),
        )
        plt.close(fig)

    payload = {
        "status": "ok",
        "step": "segment_value_regions",
        "schema": VALUE_REGIONS_SCHEMA,
        "basis": "value",
        "feature_type": "value_region",
        "input_path": str(resolved_input),
        "metadata_path": str(
            resolve_matrix_metadata_path(resolved_input, metadata_path) or ""
        ),
        "physical_context": physical,
        "method": selected_method,
        "threshold": selected_threshold,
        "selected_polarity": selected_polarity,
        "selected_score": selected_score,
        "score_diagnostics": selected["diagnostics"],
        "quality": selected["quality"],
        "candidate_scores": [
            {
                "method": item["method"],
                "polarity": item["polarity"],
                "threshold": item["threshold"],
                "detrend_sigma_px": item["detrend_sigma_px"],
                "score": item["score"],
                "diagnostics": item["diagnostics"],
                "quality": item["quality"],
            }
            for item in candidates
        ],
        "min_area_px": min_area_px,
        "threshold_percentile": threshold_percentile,
        "region_count": len(regions),
        "regions": regions,
        "label_npy_path": label_npy,
        "mask_png_path": mask_png,
        "overlay_png_path": overlay_png,
        "overlay_plot_decorations": (
            overlay_decorations if write_plots else None
        ),
        "regions_json_path": regions_json,
        "plots_written": bool(write_plots),
    }
    write_json_file(regions_json, payload)
    return payload


def detect_local_outliers(
    height_data_path: str,
    regions_json_path: str,
    output_dir: str,
    metadata_path: str,
    polarity: str,
    robust_sigma_threshold: float,
    min_area_px: int,
    min_height_contrast: float,
    min_global_range_fraction: float,
    scope: str,
) -> dict[str, Any]:
    """Detect contaminant-like local outliers at the requested analysis scope.

    Scopes:
    - ``regions``: detect inside each segmented region independently.
    - ``background``: detect only outside segmented regions.
    - ``full``: detect across the whole finite image as one area.
    - ``all``: detect segmented regions and background separately.
    """

    if polarity not in {"bright", "dark", "both"}:
        raise ValueError("polarity must be bright, dark, or both")
    if scope not in {"regions", "background", "full", "all"}:
        raise ValueError("scope must be regions, background, full, or all")

    matrix, metadata, resolved_input = load_height_data_and_metadata(
        height_data_path,
        metadata_path,
    )
    regions_path: Path | None = None
    regions_payload: dict[str, Any] = {}
    label_image: np.ndarray | None = None
    if scope != "full":
        if not regions_json_path:
            raise ValueError("regions_json_path is required unless scope is full")
        regions_path, regions_payload = read_json_file(
            regions_json_path,
            expected_schema=VALUE_REGIONS_SCHEMA,
            required_keys=("label_npy_path", "regions"),
        )
        label_image = np.load(resolve_readable_file(regions_payload["label_npy_path"]))

    out_dir = ensure_output_dir(output_dir)
    stem = resolved_input.stem
    physical = physical_area_context_from_metadata(metadata)
    pixel_area = physical.get("pixel_area")
    pixel_width = physical.get("pixel_width")
    pixel_height = physical.get("pixel_height")
    height_unit = physical.get("height_unit") or "μm"
    area_unit = "μm^2"
    if physical.get("x_unit") and physical.get(
            "y_unit") and physical.get("x_unit") == physical.get("y_unit"):
        area_unit = f"{physical['x_unit']}^2"

    detection = detect_local_outlier_components(
        matrix,
        scope=scope,
        regions=list(regions_payload.get("regions", [])),
        label_image=label_image,
        polarity=polarity,
        robust_sigma_threshold=robust_sigma_threshold,
        min_area_px=min_area_px,
        min_height_contrast=min_height_contrast,
        min_global_range_fraction=min_global_range_fraction,
        pixel_area=pixel_area,
        pixel_width=pixel_width,
        pixel_height=pixel_height,
        height_unit=height_unit,
        area_unit=area_unit,
    )
    global_range = float(detection["global_range"])
    dynamic_min_contrast = float(detection["dynamic_min_contrast"])
    contaminant_label = np.asarray(detection["label_image"], dtype=np.int32)
    contaminants = list(detection["contaminants"])
    analysis_region_summaries = list(detection["analysis_regions"])
    analysis_region_count = int(detection["analysis_region_count"])

    label_npy = out_dir / f"{stem}_contaminants_label.npy"
    mask_png = out_dir / f"{stem}_contaminants_mask.png"
    overlay_png = out_dir / f"{stem}_contaminants_overlay.png"
    contaminants_json = out_dir / f"{stem}_contaminants.json"
    atomic_write_path(label_npy, lambda path: np.save(path, contaminant_label))

    fig, ax = plt.subplots(figsize=(5, 5), dpi=PLOT_DPI)
    ax.imshow(contaminant_label, cmap="magma")
    ax.set_axis_off()
    fig.tight_layout()
    atomic_write_path(
        mask_png,
        lambda path: save_figure(fig, path, pad_inches=0.02),
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=PSEUDOCOLOR_FIGURE_SIZE, dpi=PLOT_DPI)
    image = render_height_image(ax, matrix, cmap=COLORMAP)
    if np.any(contaminant_label > 0):
        ax.contour(contaminant_label > 0, levels=[0.5], colors="magenta", linewidths=1.0)
        for contaminant in contaminants:
            y, x = contaminant["centroid_yx"]
            label_text = contaminant["contaminant_id"].replace("contaminant_", "P")
            ax.text(
                x,
                y,
                label_text,
                color="white",
                fontsize=6,
                ha="center",
                va="center",
                bbox={"facecolor": "magenta", "alpha": 0.65, "pad": 1, "linewidth": 0},
            )
    set_plot_title(ax, "Height")
    overlay_decorations = decorate_height_map(
        fig,
        ax,
        image,
        matrix_shape=matrix.shape,
        physical_context=physical,
        value_label="Height",
        value_unit=height_unit,
    )
    fig.tight_layout()
    atomic_write_path(
        overlay_png,
        lambda path: save_figure(fig, path, pad_inches=0.02),
    )
    plt.close(fig)

    per_region: dict[str, dict[str, Any]] = {}
    for contaminant in contaminants:
        item = per_region.setdefault(
            contaminant["region_id"],
            {
                "region_id": contaminant["region_id"],
                "region_label_value": contaminant.get("region_label_value"),
                "analysis_scope": contaminant.get("analysis_scope"),
                "contaminant_count": 0,
                "contaminant_area_px": 0,
                "contaminant_area_physical": 0.0 if pixel_area else None,
            },
        )
        item["contaminant_count"] += 1
        item["contaminant_area_px"] += contaminant["area_px"]
        if pixel_area:
            item["contaminant_area_physical"] += contaminant["area_px"] * pixel_area

    payload = {
        "status": "ok",
        "step": "detect_local_outliers",
        "schema": LOCAL_OUTLIERS_SCHEMA,
        "input_path": str(resolved_input),
        "metadata_path": str(
            resolve_matrix_metadata_path(resolved_input, metadata_path) or ""
        ),
        "regions_json_path": str(regions_path) if regions_path else "",
        "method": "scope_mad_outlier_components",
        "scope": scope,
        "polarity": polarity,
        "robust_sigma_threshold": robust_sigma_threshold,
        "min_area_px": min_area_px,
        "min_height_contrast": min_height_contrast,
        "min_global_range_fraction": min_global_range_fraction,
        "global_height_range": global_range,
        "dynamic_min_contrast": dynamic_min_contrast,
        "height_unit": height_unit,
        "area_unit": area_unit,
        "analysis_region_count": analysis_region_count,
        "analysis_regions": analysis_region_summaries,
        "contaminant_count": len(contaminants),
        "contaminants": contaminants,
        "per_region": list(per_region.values()),
        "contaminant_label_npy_path": label_npy,
        "contaminant_mask_png_path": mask_png,
        "contaminant_overlay_png_path": overlay_png,
        "overlay_plot_decorations": overlay_decorations,
        "contaminants_json_path": contaminants_json,
    }
    write_json_file(contaminants_json, payload)
    return payload

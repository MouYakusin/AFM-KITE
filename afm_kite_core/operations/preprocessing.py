"""Matrix preprocessing and rendering stage implementations."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from afm_kite_core.algorithms.filtering import filter_matrix
from afm_kite_core.algorithms.leveling import auto_level_matrix, level_matrix
from afm_kite_core.algorithms.statistics import finite_matrix_summary, image_clip_limits
from afm_kite_core.data_io.matrix import (
    height_unit,
    load_height_data,
    load_height_data_and_metadata,
    metadata_sidecar_path,
    metadata_summary,
    model_info,
    physical_area_context_from_metadata,
    resolve_matrix_metadata_path,
)
from afm_kite_core.artifacts.plotting import (
    PLOT_DPI,
    PSEUDOCOLOR_FIGURE_SIZE,
    canonical_plot_title,
    canonical_plot_value_label,
    decorate_height_map,
    plt,
    render_height_image,
    resolve_afm_colormap,
    save_figure,
    set_plot_title,
)
from afm_kite_core.artifacts.writers import (
    atomic_write_path,
    write_json_file,
)
from afm_kite_core.data_io.paths import ensure_output_dir, resolve_readable_file


def level_height_data(
    height_data_path: str,
    output_dir: str,
    mode: str,
    metadata_path: str,
) -> dict[str, Any]:
    """Apply common AFM leveling to a height matrix.

    Supported modes are ``plane``, ``line``, ``none``, and ``auto``.
    """

    matrix, metadata, resolved_input = load_height_data_and_metadata(
        height_data_path,
        metadata_path,
    )
    physical_context = physical_area_context_from_metadata(metadata) if metadata else {}
    out_dir = ensure_output_dir(output_dir)
    stem = resolved_input.stem
    if mode == "auto":
        leveled, diagnostics, foreground_mask, background_mask = auto_level_matrix(
            matrix,
            physical_context,
        )
    else:
        leveled, diagnostics = level_matrix(matrix, mode, physical_context)
        foreground_mask = None
        background_mask = None

    output_npy = out_dir / f"{stem}_leveled_{mode}.npy"
    output_tsv = out_dir / f"{stem}_leveled_{mode}.tsv"
    output_metadata = metadata_sidecar_path(output_npy)
    atomic_write_path(output_npy, lambda path: np.save(path, leveled))
    atomic_write_path(
        output_tsv,
        lambda path: np.savetxt(path, leveled, delimiter="\t", fmt="%.12g"),
    )
    write_json_file(output_metadata, metadata)

    payload: dict[str, Any] = {
        "status": "ok",
        "step": "level_height_data",
        "input_path": str(resolved_input),
        "leveled_npy_path": output_npy,
        "leveled_txt_path": output_tsv,
        "metadata_path": output_metadata,
        "shape": list(leveled.shape),
        "metadata_summary": metadata_summary(metadata) if metadata else {},
        "physical_context": physical_context,
        "diagnostics": diagnostics,
    }

    if mode == "auto" and foreground_mask is not None and background_mask is not None:
        foreground_npy = out_dir / f"{stem}_level_auto_foreground_mask.npy"
        background_npy = out_dir / f"{stem}_level_auto_background_mask.npy"
        foreground_png = out_dir / f"{stem}_level_auto_foreground_mask.png"
        background_png = out_dir / f"{stem}_level_auto_background_mask.png"
        diagnostics_json = out_dir / f"{stem}_level_auto_diagnostics.json"

        atomic_write_path(
            foreground_npy,
            lambda path: np.save(path, foreground_mask.astype(bool)),
        )
        atomic_write_path(
            background_npy,
            lambda path: np.save(path, background_mask.astype(bool)),
        )
        atomic_write_path(
            foreground_png,
            lambda path: plt.imsave(
                path,
                foreground_mask.astype(float),
                cmap="gray",
                vmin=0,
                vmax=1,
                dpi=PLOT_DPI,
            ),
        )
        atomic_write_path(
            background_png,
            lambda path: plt.imsave(
                path,
                background_mask.astype(float),
                cmap="gray",
                vmin=0,
                vmax=1,
                dpi=PLOT_DPI,
            ),
        )

        auto_artifacts = {
            "foreground_mask_npy_path": foreground_npy,
            "foreground_mask_png_path": foreground_png,
            "background_mask_npy_path": background_npy,
            "background_mask_png_path": background_png,
            "diagnostics_json_path": diagnostics_json,
        }
        diagnostics["artifacts"] = auto_artifacts
        payload.update(auto_artifacts)
        write_json_file(diagnostics_json, diagnostics)

    return payload


def apply_matrix_filter(
    height_data_path: str,
    output_dir: str,
    filter_name: str,
    sigma: float,
    size: int,
    metadata_path: str,
) -> dict[str, Any]:
    """Apply a common image-processing filter to AFM height data.

    Supported filters are ``gaussian``, ``median``, ``mean``, ``sobel``, and ``laplace``.
    """

    matrix, metadata, resolved_input = load_height_data_and_metadata(
        height_data_path,
        metadata_path,
    )
    filtered, diagnostics = filter_matrix(matrix, filter_name, sigma, size)
    filter_key = diagnostics["filter"]
    out_dir = ensure_output_dir(output_dir)
    stem = resolved_input.stem
    output_npy = out_dir / f"{stem}_{filter_key}.npy"
    output_tsv = out_dir / f"{stem}_{filter_key}.tsv"
    output_metadata = metadata_sidecar_path(output_npy)
    atomic_write_path(output_npy, lambda path: np.save(path, filtered))
    atomic_write_path(
        output_tsv,
        lambda path: np.savetxt(path, filtered, delimiter="\t", fmt="%.12g"),
    )
    write_json_file(output_metadata, metadata)

    return {
        "status": "ok",
        "step": "apply_matrix_filter",
        "filter": filter_key,
        "diagnostics": diagnostics,
        "filtered_npy_path": output_npy,
        "filtered_txt_path": output_tsv,
        "metadata_path": output_metadata,
        "physical_context": physical_area_context_from_metadata(metadata),
        "shape": list(filtered.shape),
    }


def render_pseudocolor(
    matrix_data_path: str,
    output_dir: str,
    colormap: str,
    percentile_clip: float,
    value_label: str,
    metadata_path: str,
    plot_title: str,
    color_scale_mode: str,
    color_min: float | None,
    color_max: float | None,
) -> dict[str, Any]:
    """Render a 2D numeric matrix with an AFM colorbar and scale bar."""

    if color_scale_mode not in {"auto", "symmetric_zero"}:
        raise ValueError("color_scale_mode must be auto or symmetric_zero")
    has_color_min = color_min is not None
    has_color_max = color_max is not None
    if has_color_min != has_color_max:
        raise ValueError("color_min and color_max must be supplied together")
    if has_color_min and color_scale_mode != "auto":
        raise ValueError(
            "explicit color limits require color_scale_mode='auto'"
        )
    if has_color_min:
        color_min = float(color_min)
        color_max = float(color_max)
        if not np.isfinite(color_min) or not np.isfinite(color_max):
            raise ValueError("color_min and color_max must be finite")
        if color_min >= color_max:
            raise ValueError("color_min must be less than color_max")

    matrix, metadata, resolved_input = load_height_data_and_metadata(
        matrix_data_path,
        metadata_path,
    )
    physical_context = (
        physical_area_context_from_metadata(metadata) if metadata else {}
    )
    value_unit = height_unit(metadata) if metadata else "μm"
    out_dir = ensure_output_dir(output_dir)
    stem = resolved_input.stem
    output_png = out_dir / f"{stem}_pseudocolor.png"

    if has_color_min:
        vmin, vmax = float(color_min), float(color_max)
        color_limits_source = "explicit_shared"
    else:
        vmin, vmax = image_clip_limits(matrix, percentile_clip)
        color_limits_source = "percentile_clip"
        if color_scale_mode == "symmetric_zero":
            limit = max(abs(float(vmin)), abs(float(vmax)))
            if limit <= 0.0:
                limit = 1.0
            vmin, vmax = -limit, limit

    selected_colormap = resolve_afm_colormap(colormap)
    fig, ax = plt.subplots(figsize=PSEUDOCOLOR_FIGURE_SIZE, dpi=PLOT_DPI)
    image = render_height_image(
        ax,
        matrix,
        cmap=selected_colormap,
        vmin=vmin,
        vmax=vmax,
    )
    resolved_value_label = canonical_plot_value_label(value_label)
    image_mode = str(model_info(metadata).get("ImageShowMode") or "")
    normalized_mode = "".join(
        character
        for character in image_mode.split("(", 1)[0].casefold()
        if character.isalnum()
    )
    if (
        resolved_value_label.casefold() == "value"
        and normalized_mode in {"height", "heightsensor"}
    ):
        resolved_value_label = "Height"
    plot_decorations = decorate_height_map(
        fig,
        ax,
        image,
        matrix_shape=matrix.shape,
        physical_context=physical_context,
        value_label=resolved_value_label,
        value_unit=value_unit,
    )
    is_height_map = resolved_value_label.casefold() == "height"
    requested_plot_title = (
        "Height"
        if is_height_map
        else str(plot_title).strip() or resolved_value_label
    )
    resolved_plot_title = canonical_plot_title(
        requested_plot_title,
        channel_hint=f"{resolved_value_label} {image_mode}",
    )
    title_fontweight = "bold"
    set_plot_title(
        ax,
        resolved_plot_title,
    )
    fig.tight_layout()
    atomic_write_path(
        output_png,
        lambda path: save_figure(fig, path, pad_inches=0.02),
    )
    plt.close(fig)
    display_vmin = float(plot_decorations["colorbar"]["minimum"])
    display_vmax = float(plot_decorations["colorbar"]["maximum"])
    display_value_unit = str(plot_decorations["colorbar"]["value_unit"])

    return {
        "status": "ok",
        "step": "render_pseudocolor",
        "input_path": str(resolved_input),
        "matrix_data_path": str(resolved_input),
        "pseudocolor_path": output_png,
        "colormap": getattr(selected_colormap, "name", str(selected_colormap)),
        "value_label": resolved_value_label,
        "plot_title": resolved_plot_title,
        "title_fontweight": title_fontweight,
        "color_scale_mode": color_scale_mode,
        "color_limits_source": color_limits_source,
        "value_unit": value_unit,
        "display_value_unit": display_value_unit,
        "metadata_path": str(
            resolve_matrix_metadata_path(resolved_input, metadata_path) or ""
        ),
        "physical_context": physical_context,
        "plot_decorations": plot_decorations,
        "vmin": vmin,
        "vmax": vmax,
        "display_vmin": display_vmin,
        "display_vmax": display_vmax,
        "shape": list(matrix.shape),
    }


def subtract_matrices(
    minuend_data_path: str,
    subtrahend_data_path: str,
    output_dir: str,
    output_stem: str,
    nan_policy: str,
    metadata_path: str,
) -> dict[str, Any]:
    """Subtract one matrix from another and write reusable residual artifacts."""

    if nan_policy not in {"propagate", "left"}:
        raise ValueError("nan_policy must be propagate or left")

    minuend, metadata, _resolved_minuend = load_height_data_and_metadata(
        minuend_data_path,
        metadata_path,
    )
    subtrahend = load_height_data(subtrahend_data_path)
    if minuend.shape != subtrahend.shape:
        raise ValueError(
            f"Matrix shapes must match, got {minuend.shape} and {subtrahend.shape}"
        )

    residual = minuend - subtrahend
    if nan_policy == "left":
        residual = np.where(np.isfinite(minuend), residual, np.nan)

    out_dir = ensure_output_dir(output_dir)
    left_stem = Path(minuend_data_path).stem
    right_stem = Path(subtrahend_data_path).stem
    stem = output_stem.strip() if output_stem else f"{left_stem}_minus_{right_stem}"
    result_npy = out_dir / f"{stem}.npy"
    result_tsv = out_dir / f"{stem}.tsv"
    result_metadata = metadata_sidecar_path(result_npy)

    atomic_write_path(result_npy, lambda path: np.save(path, residual))
    atomic_write_path(
        result_tsv,
        lambda path: np.savetxt(path, residual, delimiter="\t", fmt="%.12g"),
    )
    write_json_file(result_metadata, metadata)

    return {
        "status": "ok",
        "step": "subtract_matrices",
        "minuend_data_path": str(resolve_readable_file(minuend_data_path)),
        "subtrahend_data_path": str(resolve_readable_file(subtrahend_data_path)),
        "output_dir": out_dir,
        "output_stem": stem,
        "nan_policy": nan_policy,
        "shape": list(residual.shape),
        "result_summary": finite_matrix_summary(residual),
        "artifacts": {
            "result_npy_path": result_npy,
            "result_tsv_path": result_tsv,
            "metadata_path": result_metadata,
        },
        "result_npy_path": result_npy,
        "result_tsv_path": result_tsv,
        "metadata_path": result_metadata,
        "physical_context": physical_area_context_from_metadata(metadata),
    }

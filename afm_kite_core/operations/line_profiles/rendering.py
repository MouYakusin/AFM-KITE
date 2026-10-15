"""Line-profile and feature-crossing overlay rendering stage."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from afm_kite_core.algorithms.statistics import image_clip_limits
from afm_kite_core.contracts import (
    FEATURE_LINE_SCHEMA,
    LINE_PROFILE_MEASUREMENT_SCHEMA,
    LINE_PROFILE_SCHEMA,
)
from afm_kite_core.data_io.matrix import (
    load_height_data_and_metadata,
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
    read_json_file,
    write_json_file,
)
from afm_kite_core.data_io.paths import ensure_output_dir, resolve_readable_file

from .contracts import (
    line_id,
    safe_profile_name,
    sampling_contract,
    validate_compatible_profiles,
    window_indices,
)


def normalise_plot_kinds(plot_kinds: Sequence[str] | None) -> list[str]:
    selected = list(plot_kinds or ["profile", "overlay"])
    if not selected:
        raise ValueError("plot_kinds must contain profile, overlay, or both")
    invalid = sorted(set(selected) - {"profile", "overlay"})
    if invalid:
        raise ValueError("Unsupported plot kinds: " + ", ".join(invalid))
    return list(dict.fromkeys(selected))


def _render_profile_plot(
    profiles: list[tuple[Any, dict[str, Any]]],
    output_path: Any,
) -> str:
    figure, axis = plt.subplots(
        figsize=(8.2, 4.8),
        dpi=PLOT_DPI,
        constrained_layout=True,
    )
    for _, payload in profiles:
        distance = np.asarray(payload["coordinates"]["distance"], dtype=float)
        values = np.asarray(payload["values"], dtype=float)
        axis.plot(distance, values, linewidth=1.6, label=payload["profile_label"])
        sampling = payload.get("sampling", {}) or {}
        band_std = np.asarray(
            sampling.get("standard_deviation_per_position", []),
            dtype=float,
        )
        if (
            int(sampling.get("line_width_px", 1)) > 1
            and band_std.shape == values.shape
        ):
            axis.fill_between(
                distance,
                values - band_std,
                values + band_std,
                alpha=0.12,
            )
    first_payload = profiles[0][1]
    distance = np.asarray(first_payload["coordinates"]["distance"], dtype=float)
    left, feature, right = window_indices(first_payload)
    axis.axvspan(distance[left[0]], distance[left[-1]], alpha=0.1, color="gray")
    axis.axvspan(
        distance[feature[0]],
        distance[feature[-1]],
        alpha=0.13,
        color="orange",
    )
    axis.axvspan(distance[right[0]], distance[right[-1]], alpha=0.1, color="gray")
    axis.set_xlabel(
        f"Distance ({first_payload['coordinates'].get('distance_unit', 'μm')})"
    )
    axis.set_ylabel("Value")
    channel_hint = " ".join(
        str(payload.get(key) or "")
        for _, payload in profiles
        for key in ("profile_label", "matrix_data_path", "metadata_path")
    )
    plot_title = canonical_plot_title(
        "Shared-line profiles",
        channel_hint=channel_hint,
    )
    set_plot_title(axis, plot_title)
    axis.legend(loc="best")
    axis.grid(alpha=0.2)
    atomic_write_path(
        output_path,
        lambda path: save_figure(
            figure,
            path,
            dpi=PLOT_DPI,
            pad_inches=0.08,
        ),
    )
    plt.close(figure)
    return plot_title


def _render_overlay_plot(
    line_payload: dict[str, Any],
    first_profile: dict[str, Any],
    matrix: np.ndarray,
    output_path: Any,
    colormap: str,
    physical_context: dict[str, Any],
    value_label: str,
    value_unit: str,
) -> dict[str, object]:
    vmin, vmax = image_clip_limits(matrix)
    figure, axis = plt.subplots(
        figsize=PSEUDOCOLOR_FIGURE_SIZE,
        dpi=PLOT_DPI,
    )
    image = render_height_image(
        axis,
        matrix,
        cmap=resolve_afm_colormap(colormap),
        vmin=vmin,
        vmax=vmax,
    )
    coordinates = line_payload["coordinates"]
    x_px = np.asarray(coordinates["x_px"], dtype=float)
    y_px = np.asarray(coordinates["y_px"], dtype=float)
    offsets_px = np.asarray(sampling_contract(first_profile)["offsets_px"], dtype=float)
    if offsets_px.size > 1:
        tangent_x = np.gradient(x_px)
        tangent_y = np.gradient(y_px)
        tangent_norm = np.hypot(tangent_x, tangent_y)
        normal_x = -tangent_y / tangent_norm
        normal_y = tangent_x / tangent_norm
        for offset_px in (float(offsets_px[0]), float(offsets_px[-1])):
            axis.plot(
                x_px + offset_px * normal_x,
                y_px + offset_px * normal_y,
                color="yellow",
                linewidth=0.8,
                linestyle="--",
                alpha=0.9,
            )
    axis.plot(x_px, y_px, color="white", linewidth=2.4)
    axis.plot(x_px, y_px, color="red", linewidth=1.1)
    axis.scatter(
        [x_px[0], x_px[-1]],
        [y_px[0], y_px[-1]],
        c=["cyan", "yellow"],
        s=28,
        edgecolors="black",
        linewidths=0.5,
    )
    mask_path = line_payload.get("artifacts", {}).get(
        "selected_feature_mask_npy_path"
    )
    if mask_path:
        selected_mask = np.load(resolve_readable_file(mask_path)).astype(bool)
        if selected_mask.shape == matrix.shape:
            axis.contour(
                selected_mask.astype(float),
                levels=[0.5],
                colors="white",
                linewidths=0.8,
            )
    plot_title = canonical_plot_title(
        canonical_plot_value_label(value_label, default="Height"),
        channel_hint=value_label,
    )
    set_plot_title(axis, plot_title)
    plot_decorations = decorate_height_map(
        figure,
        axis,
        image,
        matrix_shape=matrix.shape,
        physical_context=physical_context,
        value_label=value_label,
        value_unit=value_unit,
    )
    plot_decorations["plot_title"] = plot_title
    figure.tight_layout()
    atomic_write_path(
        output_path,
        lambda path: save_figure(figure, path, dpi=PLOT_DPI),
    )
    plt.close(figure)
    return plot_decorations


def render_line_profiles(
    line_contract_json_path: str,
    line_profile_json_paths: list[str],
    output_dir: str,
    measurement_json_path: str,
    overlay_matrix_path: str,
    plot_kinds: Sequence[str] | None,
    colormap: str,
    overlay_metadata_path: str,
    overlay_value_label: str,
) -> dict[str, Any]:
    """Render optional profile and position-overlay plots without measuring data."""

    kinds = normalise_plot_kinds(plot_kinds)
    line_path, line_payload = read_json_file(
        line_contract_json_path,
        expected_schema=FEATURE_LINE_SCHEMA,
        required_keys=("line_id", "coordinates", "windows"),
    )
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
    current_line_id = line_id(line_payload)
    if any(
        str(payload.get("line_id", "")) != current_line_id
        for _, payload in profiles
    ):
        raise ValueError("Profile line_id does not match line contract")

    measurement_path = ""
    if measurement_json_path:
        measurement_path, measurement_payload = read_json_file(
            measurement_json_path,
            expected_schema=LINE_PROFILE_MEASUREMENT_SCHEMA,
            required_keys=("line_id", "measurements"),
        )
        if str(measurement_payload.get("line_id", "")) != current_line_id:
            raise ValueError("Measurement line_id does not match line contract")

    labels = [
        safe_profile_name(payload.get("profile_label", path.stem))
        for path, payload in profiles
    ]
    source_stem = safe_profile_name(
        f"{line_payload.get('source_stem', line_path.stem)}_"
        + "_vs_".join(labels)
    )
    out_dir = ensure_output_dir(output_dir)
    artifacts: dict[str, Any] = {
        "line_contract_json_path": line_path,
        "line_profile_json_paths": [path for path, _ in profiles],
    }
    plot_decorations: dict[str, Any] = {}
    plot_titles: dict[str, str] = {}

    if "profile" in kinds:
        profile_png = out_dir / f"{source_stem}_line_profile.png"
        plot_titles["profile"] = _render_profile_plot(profiles, profile_png)
        artifacts["line_profile_png"] = profile_png

    if "overlay" in kinds:
        if not overlay_matrix_path:
            raise ValueError("overlay_matrix_path is required for an overlay plot")
        matrix, overlay_metadata, overlay_path = load_height_data_and_metadata(
            overlay_matrix_path,
            overlay_metadata_path,
        )
        expected_shape = tuple(int(item) for item in line_payload.get("shape", []))
        if expected_shape and tuple(matrix.shape) != expected_shape:
            raise ValueError("Overlay matrix shape does not match line contract")
        overlay_png = out_dir / f"{source_stem}_line_overlay.png"
        metadata_physical = physical_area_context_from_metadata(overlay_metadata)
        overlay_physical = dict(metadata_physical)
        if not overlay_physical:
            profile_physical = profiles[0][1].get("physical_context", {}) or {}
            overlay_physical = {
                key: profile_physical[key]
                for key in ("pixel_width", "pixel_height", "x_unit", "y_unit")
                if profile_physical.get(key) not in {None, ""}
            }
        overlay_value_unit = str(metadata_physical.get("height_unit") or "μm")
        first_profile_matrix = str(profiles[0][1].get("matrix_data_path") or "")
        if not overlay_value_unit and first_profile_matrix:
            try:
                same_matrix = (
                    resolve_readable_file(first_profile_matrix) == overlay_path
                )
            except FileNotFoundError:
                same_matrix = False
            if same_matrix:
                overlay_value_unit = str(
                    profiles[0][1].get("value_unit") or "μm"
                )
        overlay_value_unit = overlay_value_unit or "μm"
        plot_decorations["overlay"] = _render_overlay_plot(
            line_payload,
            profiles[0][1],
            matrix,
            overlay_png,
            colormap,
            overlay_physical,
            overlay_value_label or "Height",
            overlay_value_unit,
        )
        plot_titles["overlay"] = str(
            plot_decorations["overlay"]["plot_title"]
        )
        artifacts["line_overlay_png"] = overlay_png
        artifacts["overlay_matrix_path"] = overlay_path
        artifacts["overlay_metadata_path"] = str(
            resolve_matrix_metadata_path(overlay_path, overlay_metadata_path) or ""
        )

    render_json = out_dir / f"{source_stem}_line_render.json"
    artifacts["line_render_json"] = render_json
    payload = {
        "status": "ok",
        "step": "line_profile_render",
        "schema": "afm.line-profile-render/v1",
        "line_id": current_line_id,
        "plot_kinds": kinds,
        "measurement_json_path": measurement_path,
        "plot_titles": plot_titles,
        "plot_decorations": plot_decorations,
        "artifacts": artifacts,
    }
    write_json_file(render_json, payload)
    return payload


__all__ = ["normalise_plot_kinds", "render_line_profiles"]

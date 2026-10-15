"""Rendering helpers for periodic-orientation and profile workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from afm_kite_core.algorithms.statistics import image_clip_limits
from afm_kite_core.artifacts.plotting import (
    COLORMAP,
    COLORBAR_EXTREME_FONT_SIZE,
    COLORBAR_FRACTION,
    COLORBAR_PAD,
    PLOT_DPI,
    PSEUDOCOLOR_FIGURE_SIZE,
    decorate_height_map,
    plt,
    render_height_image,
    save_figure,
    set_plot_title,
    style_colorbar_extreme_labels,
)
from afm_kite_core.artifacts.writers import atomic_write_path


PROFILE_NORMAL_DIRECTION_FONT_SIZE = 12
PROFILE_NORMAL_DIRECTION_LABEL = "profile-normal direction"
HEIGHT_MAP_TITLE = "Height"
MEASUREMENT_DIRECTION_LABEL_POSITION = (0.5, 0.2)
FFT_DIRECTION_LABEL_POSITION = (0.5, 0.12)
FFT_FREQUENCY_UNIT = "cycles/μm"
FFT_COLORBAR_LABEL = "ln(1 + FFT power)"
FFT_COLORBAR_SIZE = f"{100.0 * COLORBAR_FRACTION:g}%"
FFT_COLORBAR_PAD = f"{100.0 * COLORBAR_PAD:g}%"
FFT_COLORBAR_EXTREME_FONTWEIGHT = "bold"
DIRECTION_LABEL_ROTATION = 0
DIRECTION_ARROW_COLOR = "#ff0000"
DIRECTION_ARROW_STYLE = (
    "Simple,head_length=1.0,head_width=0.75,tail_width=0.18"
)
DIRECTION_ARROW_MUTATION_SCALE = 22
FFT_DIRECTION_ARROW_LENGTH = 0.618
FFT_DIRECTION_ARROW_CENTER_OFFSET = 0.10
FFT_IMAGE_BOX_ASPECT = 1.0
PERIODIC_PROFILE_LEGEND_LOCATION = "upper right"


def _direction_arrowprops() -> dict[str, Any]:
    """Return the shared FFT/height-map profile-direction arrow style."""

    return {
        "arrowstyle": DIRECTION_ARROW_STYLE,
        "color": DIRECTION_ARROW_COLOR,
        "linewidth": 0,
        "mutation_scale": DIRECTION_ARROW_MUTATION_SCALE,
        "shrinkA": 0,
        "shrinkB": 0,
    }


def _measurement_direction_arrow_geometry(
    rows: int,
    cols: int,
    unit_x: float,
    unit_y: float,
) -> tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    float,
]:
    """Return an FFT-length direction arrow centered on the height image."""

    center_x = (float(cols) - 1.0) / 2.0
    center_y = (float(rows) - 1.0) / 2.0
    arrow_length = FFT_DIRECTION_ARROW_LENGTH * float(min(rows, cols))
    half_length = 0.5 * arrow_length
    arrow_start = (
        center_x - float(unit_x) * half_length,
        center_y - float(unit_y) * half_length,
    )
    arrow_end = (
        center_x + float(unit_x) * half_length,
        center_y + float(unit_y) * half_length,
    )
    return (center_x, center_y), arrow_start, arrow_end, arrow_length


def _frequency_bin_extent(
    freq_x: np.ndarray,
    freq_y: np.ndarray,
) -> tuple[float, float, float, float]:
    """Return image boundaries around frequency-bin center coordinates."""

    x_values = np.asarray(freq_x, dtype=float)
    y_values = np.asarray(freq_y, dtype=float)
    if x_values.size < 2 or y_values.size < 2:
        raise ValueError("FFT frequency axes must contain at least two bins")
    x_step = float(np.median(np.diff(x_values)))
    y_step = float(np.median(np.diff(y_values)))
    return (
        float(x_values[0] - (x_step / 2.0)),
        float(x_values[-1] + (x_step / 2.0)),
        float(y_values[0] - (y_step / 2.0)),
        float(y_values[-1] + (y_step / 2.0)),
    )


def _positive_pixel_size(value: object) -> float:
    """Return a finite positive μm/pixel value, defaulting to 1."""

    try:
        pixel_size = float(value)
    except (TypeError, ValueError):
        return 1.0
    return pixel_size if np.isfinite(pixel_size) and pixel_size > 0 else 1.0


def render_frequency_spectrum(
    log_power: np.ndarray,
    freq_x: np.ndarray,
    freq_y: np.ndarray,
    *,
    direction_fx: float,
    direction_fy: float,
    peak_fx: float,
    peak_fy: float,
    opposite_fx: float,
    opposite_fy: float,
    profile_angle: float,
    output_path: Path,
    pixel_width_um: float = 1.0,
    pixel_height_um: float = 1.0,
) -> dict[str, object]:
    """Render the FFT spectrum and selected measurement direction."""

    resolved_pixel_width = _positive_pixel_size(pixel_width_um)
    resolved_pixel_height = _positive_pixel_size(pixel_height_um)
    display_freq_x = np.asarray(freq_x, dtype=float) / resolved_pixel_width
    display_freq_y = np.asarray(freq_y, dtype=float) / resolved_pixel_height
    display_direction_fx = float(direction_fx) / resolved_pixel_width
    display_direction_fy = float(direction_fy) / resolved_pixel_height
    display_peak_fx = float(peak_fx) / resolved_pixel_width
    display_peak_fy = float(peak_fy) / resolved_pixel_height
    display_opposite_fx = float(opposite_fx) / resolved_pixel_width
    display_opposite_fy = float(opposite_fy) / resolved_pixel_height
    if np.hypot(display_direction_fx, display_direction_fy) > 0:
        display_angle = float(
            np.degrees(
                np.arctan2(display_direction_fy, display_direction_fx)
            )
            % 180.0
        )
    else:
        display_angle = float(profile_angle)

    direction_color = DIRECTION_ARROW_COLOR
    arrow_length = FFT_DIRECTION_ARROW_LENGTH
    arrow_center_offset = FFT_DIRECTION_ARROW_CENTER_OFFSET
    angle_radians = np.deg2rad(display_angle)
    arrow_direction = np.array(
        [np.cos(angle_radians), np.sin(angle_radians)],
        dtype=float,
    )
    arrow_normal = np.array(
        [-arrow_direction[1], arrow_direction[0]],
        dtype=float,
    )
    if arrow_normal[0] < 0:
        arrow_normal *= -1.0
    image_center = np.array([0.5, 0.5], dtype=float)
    arrow_center = image_center + arrow_center_offset * arrow_normal
    arrow_start = arrow_center - 0.5 * arrow_length * arrow_direction
    arrow_end = arrow_center + 0.5 * arrow_length * arrow_direction

    fig, ax = plt.subplots(figsize=(5, 5), dpi=PLOT_DPI)
    frequency_extent = _frequency_bin_extent(display_freq_x, display_freq_y)
    image = ax.imshow(
        log_power,
        cmap="magma",
        origin="lower",
        extent=frequency_extent,
        aspect="equal",
    )
    ax.set_box_aspect(FFT_IMAGE_BOX_ASPECT)
    ax.scatter(
        [display_direction_fx],
        [display_direction_fy],
        s=52,
        facecolors="none",
        edgecolors=direction_color,
        linewidths=1.8,
    )
    ax.scatter(
        [display_peak_fx],
        [display_peak_fy],
        s=18,
        facecolors="none",
        edgecolors="#ffffff",
        linewidths=0.8,
    )
    ax.scatter(
        [display_opposite_fx],
        [display_opposite_fy],
        s=22,
        facecolors="none",
        edgecolors="#ffffff",
        linewidths=1.0,
    )
    ax.annotate(
        "",
        xy=tuple(arrow_end),
        xytext=tuple(arrow_start),
        xycoords="axes fraction",
        textcoords="axes fraction",
        arrowprops=_direction_arrowprops(),
    )
    direction_annotation = (
        f"{PROFILE_NORMAL_DIRECTION_LABEL} {display_angle:.2f}°"
    )
    ax.text(
        *FFT_DIRECTION_LABEL_POSITION,
        direction_annotation,
        transform=ax.transAxes,
        color=direction_color,
        fontsize=PROFILE_NORMAL_DIRECTION_FONT_SIZE,
        weight="bold",
        rotation=DIRECTION_LABEL_ROTATION,
        va="center",
        ha="center",
    )
    ax.set_xlabel(f"frequency x ({FFT_FREQUENCY_UNIT})")
    ax.set_ylabel(f"frequency y ({FFT_FREQUENCY_UNIT})")
    set_plot_title(
        ax,
        FFT_COLORBAR_LABEL,
    )
    divider = make_axes_locatable(ax)
    colorbar_axis = divider.append_axes(
        "right",
        size=FFT_COLORBAR_SIZE,
        pad=FFT_COLORBAR_PAD,
    )
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar_minimum, colorbar_maximum = image.get_clim()
    colorbar_minimum_label = f"{float(colorbar_minimum):.3f}"
    colorbar_maximum_label = f"{float(colorbar_maximum):.3f}"
    style_colorbar_extreme_labels(
        colorbar,
        minimum_label=colorbar_minimum_label,
        maximum_label=colorbar_maximum_label,
    )
    fig.tight_layout()
    atomic_write_path(
        output_path,
        lambda path: save_figure(fig, path, pad_inches=0.04),
    )
    plt.close(fig)
    return {
        "title": FFT_COLORBAR_LABEL,
        "title_fontweight": "bold",
        "direction_label": PROFILE_NORMAL_DIRECTION_LABEL,
        "direction_annotation": direction_annotation,
        "direction_angle_degrees": display_angle,
        "direction_arrow_center_offset_axes_fraction": arrow_center_offset,
        "axes": {
            "x_unit": FFT_FREQUENCY_UNIT,
            "y_unit": FFT_FREQUENCY_UNIT,
            "extent": [float(value) for value in frequency_extent],
            "pixel_width_um": resolved_pixel_width,
            "pixel_height_um": resolved_pixel_height,
        },
        "colorbar": {
            "label": FFT_COLORBAR_LABEL,
            "visible": True,
            "visible_side_label": False,
            "ticks": [],
            "minimum": float(colorbar_minimum),
            "maximum": float(colorbar_maximum),
            "minimum_label": colorbar_minimum_label,
            "maximum_label": colorbar_maximum_label,
            "extreme_fontweight": FFT_COLORBAR_EXTREME_FONTWEIGHT,
            "fraction": COLORBAR_FRACTION,
            "pad": COLORBAR_PAD,
            "extreme_font_size": COLORBAR_EXTREME_FONT_SIZE,
        },
    }


def render_measurement_direction(
    matrix: np.ndarray,
    *,
    unit_x: float,
    unit_y: float,
    profile_angle: float,
    physical_context: dict[str, Any] | None,
    height_unit: str,
    output_path: Path,
) -> dict[str, object]:
    """Render the selected profile-normal direction over the height map."""

    rows, cols = matrix.shape
    direction_color = DIRECTION_ARROW_COLOR
    direction_label = PROFILE_NORMAL_DIRECTION_LABEL
    direction_annotation = f"{direction_label} {profile_angle:.2f}°"
    _center, arrow_start, arrow_end, _arrow_length = (
        _measurement_direction_arrow_geometry(
            rows,
            cols,
            unit_x,
            unit_y,
        )
    )
    start_x, start_y = arrow_start
    end_x, end_y = arrow_end
    vmin, vmax = image_clip_limits(matrix)

    fig, ax = plt.subplots(figsize=PSEUDOCOLOR_FIGURE_SIZE, dpi=PLOT_DPI)
    image = render_height_image(
        ax,
        matrix,
        cmap=COLORMAP,
        vmin=vmin,
        vmax=vmax,
    )
    ax.annotate(
        "",
        xy=(end_x, end_y),
        xytext=(start_x, start_y),
        arrowprops=_direction_arrowprops(),
    )
    label_x, label_y = MEASUREMENT_DIRECTION_LABEL_POSITION
    ax.text(
        label_x,
        label_y,
        direction_annotation,
        transform=ax.transAxes,
        color=direction_color,
        fontsize=PROFILE_NORMAL_DIRECTION_FONT_SIZE,
        weight="bold",
        rotation=DIRECTION_LABEL_ROTATION,
        va="center",
        ha="center",
    )
    set_plot_title(
        ax,
        HEIGHT_MAP_TITLE,
    )
    plot_decorations = decorate_height_map(
        fig,
        ax,
        image,
        matrix_shape=matrix.shape,
        physical_context=physical_context,
        value_label="Height",
        value_unit=height_unit,
    )
    plot_decorations.update(
        {
            "title": HEIGHT_MAP_TITLE,
            "title_fontweight": "bold",
            "direction_label": direction_label,
            "direction_annotation": direction_annotation,
            "direction_angle_degrees": float(profile_angle),
        }
    )
    fig.tight_layout()
    atomic_write_path(
        output_path,
        lambda path: save_figure(fig, path, pad_inches=0.04),
    )
    plt.close(fig)
    return plot_decorations


def render_periodic_profile(
    coordinates: np.ndarray,
    profile: np.ndarray,
    threshold: float,
    extrema: dict[str, Any],
    *,
    axis_unit: str,
    height_unit: str,
    output_path: Path,
) -> None:
    """Render one measured periodic profile and its detected extrema."""

    fig, ax = plt.subplots(figsize=(7, 4), dpi=PLOT_DPI)
    ax.plot(coordinates, profile, color="#1f77b4", linewidth=1.1, label="profile")
    ax.axhline(
        threshold,
        color="#d62728",
        linewidth=0.8,
        linestyle="--",
        label="threshold",
    )
    peaks = extrema.get("peaks", [])
    valleys = extrema.get("valleys", [])
    if peaks:
        ax.scatter(
            coordinates[peaks],
            profile[peaks],
            s=14,
            color="#2ca02c",
            label="high centers",
        )
    if valleys:
        ax.scatter(
            coordinates[valleys],
            profile[valleys],
            s=14,
            color="#ff7f0e",
            label="low centers",
        )
    ax.set_xlabel(f"coordinate ({axis_unit})")
    ax.set_ylabel(f"height ({height_unit})")
    ax.legend(loc=PERIODIC_PROFILE_LEGEND_LOCATION, fontsize=7)
    fig.tight_layout()
    atomic_write_path(
        output_path,
        lambda path: save_figure(fig, path, pad_inches=0.04),
    )
    plt.close(fig)

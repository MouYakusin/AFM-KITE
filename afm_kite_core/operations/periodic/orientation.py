"""Periodic orientation workflow orchestration."""

from __future__ import annotations

from typing import Any

import numpy as np

from afm_kite_core.algorithms.periodic import (
    centroid_fft_peak,
    refine_periodic_angle_by_projection,
    wrap_angle_0_180,
)
from afm_kite_core.data_io.matrix import (
    height_unit,
    load_height_data_and_metadata,
    metadata_summary,
    physical_area_context_from_metadata,
    resolve_matrix_metadata_path,
)
from afm_kite_core.artifacts.writers import (
    atomic_write_path,
    write_json_file,
)
from afm_kite_core.data_io.paths import ensure_output_dir

from .rendering import render_frequency_spectrum, render_measurement_direction


def _pixel_size_micrometres(
    physical_context: dict[str, Any],
    axis: str,
) -> float:
    """Return one lateral pixel size in μm, defaulting to 1 μm/pixel."""

    unit = str(physical_context.get(f"{axis}_unit") or "μm").strip()
    unit = unit.replace("µ", "μ")
    unit_scale = {
        "pm": 1.0e-6,
        "nm": 1.0e-3,
        "μm": 1.0,
        "mm": 1.0e3,
        "m": 1.0e6,
    }.get(unit, 1.0)
    try:
        pixel_size = float(
            physical_context.get(
                "pixel_width" if axis == "x" else "pixel_height",
                1.0,
            )
        )
    except (TypeError, ValueError):
        pixel_size = 1.0
    if not np.isfinite(pixel_size) or pixel_size <= 0:
        pixel_size = 1.0
    return float(pixel_size * unit_scale)


def analyze_periodic_orientation(
    height_data_path: str,
    output_dir: str,
    min_period_px: float,
    max_period_px: float,
    metadata_path: str,
) -> dict[str, Any]:
    """Estimate grating orientation from the dominant 2D FFT peak pair."""

    matrix, metadata, resolved_input = load_height_data_and_metadata(
        height_data_path,
        metadata_path,
    )
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("height_data_path must contain a 2D matrix")
    rows, cols = matrix.shape
    if rows < 4 or cols < 4:
        raise ValueError("matrix must be at least 4x4 for FFT angle analysis")

    finite = np.isfinite(matrix)
    finite_count = int(np.sum(finite))
    if finite_count < 16:
        raise ValueError("matrix must contain at least 16 finite pixels")

    filled = matrix.copy()
    median_value = float(np.nanmedian(filled[finite]))
    filled[~finite] = median_value
    centered = filled - float(np.mean(filled))
    if float(np.ptp(centered)) <= 1e-12:
        raise ValueError(
            "matrix has no measurable height contrast for FFT angle analysis"
        )

    min_period = max(float(min_period_px), 2.0)
    effective_max_period = (
        float(max_period_px)
        if float(max_period_px) > 0
        else max(float(max(rows, cols)) / 2.0, min_period)
    )
    effective_max_period = max(effective_max_period, min_period)

    window = np.outer(np.hanning(rows), np.hanning(cols))
    power = np.abs(np.fft.fftshift(np.fft.fft2(centered * window))) ** 2
    log_power = np.log1p(power)
    freq_y = np.fft.fftshift(np.fft.fftfreq(rows, d=1.0))
    freq_x = np.fft.fftshift(np.fft.fftfreq(cols, d=1.0))
    grid_fx, grid_fy = np.meshgrid(freq_x, freq_y)
    radial_frequency = np.hypot(grid_fx, grid_fy)
    periods = np.full_like(radial_frequency, np.inf, dtype=float)
    periods[radial_frequency > 0] = 1.0 / radial_frequency[radial_frequency > 0]
    valid = (
        (radial_frequency > 0)
        & (periods >= min_period)
        & (periods <= effective_max_period)
        & np.isfinite(power)
    )
    if not np.any(valid):
        raise ValueError(
            "no FFT frequency bins remain after period filtering; "
            "adjust min/max period"
        )

    masked_power = np.where(valid, power, -np.inf)
    peak_flat_index = int(np.argmax(masked_power))
    peak_row, peak_col = np.unravel_index(peak_flat_index, power.shape)
    peak_fx = float(grid_fx[peak_row, peak_col])
    peak_fy = float(grid_fy[peak_row, peak_col])
    peak_power = float(power[peak_row, peak_col])
    peak_period_px = float(periods[peak_row, peak_col])
    discrete_peak_angle = wrap_angle_0_180(
        np.degrees(np.arctan2(peak_fy, peak_fx))
    )
    centroid_frequency = centroid_fft_peak(
        power,
        valid,
        grid_fx,
        grid_fy,
        peak_row,
        peak_col,
    )
    centroid_fx = float(centroid_frequency["fx_cycles_per_px"])
    centroid_fy = float(centroid_frequency["fy_cycles_per_px"])
    fft_centroid_angle = wrap_angle_0_180(
        np.degrees(np.arctan2(centroid_fy, centroid_fx))
    )
    angle_refinement = refine_periodic_angle_by_projection(
        filled,
        fft_centroid_angle,
        min_period,
        effective_max_period,
    )
    angle_refinement["used_for_profile_angle"] = False
    profile_angle = fft_centroid_angle
    direction_period_px = float(
        centroid_frequency.get("period_px") or peak_period_px
    )
    direction_frequency = (
        1.0 / direction_period_px if direction_period_px > 0 else 0.0
    )
    direction_fx = float(
        np.cos(np.deg2rad(profile_angle)) * direction_frequency
    )
    direction_fy = float(
        np.sin(np.deg2rad(profile_angle)) * direction_frequency
    )
    stripe_angle = wrap_angle_0_180(profile_angle + 90.0)

    opposite_distance = np.square(grid_fx + peak_fx) + np.square(grid_fy + peak_fy)
    opposite_flat_index = int(np.argmin(opposite_distance))
    opposite_row, opposite_col = np.unravel_index(opposite_flat_index, power.shape)
    opposite_power = float(power[opposite_row, opposite_col])
    pair_balance = (
        float(min(peak_power, opposite_power) / max(peak_power, opposite_power))
        if max(peak_power, opposite_power) > 0
        else None
    )

    valid_power = power[valid]
    total_power = float(np.sum(valid_power))
    median_power = float(np.median(valid_power))
    mean_power = float(np.mean(valid_power))
    spectral_strength = float(peak_power / total_power) if total_power > 0 else 0.0
    peak_to_median = float(peak_power / median_power) if median_power > 0 else None
    peak_to_mean = float(peak_power / mean_power) if mean_power > 0 else None

    quality_flags: list[str] = []
    if spectral_strength < 0.03:
        quality_flags.append("dominant_fft_peak_is_weak")
    if peak_to_median is not None and peak_to_median < 8.0:
        quality_flags.append("dominant_fft_peak_has_low_contrast")
    if peak_period_px > max(rows, cols) / 3.0:
        quality_flags.append("dominant_period_has_few_cycles_in_image")

    out_dir = ensure_output_dir(output_dir)
    stem = resolved_input.stem
    spectrum_npy = out_dir / f"{stem}_grating_fft_log_power.npy"
    spectrum_png = out_dir / f"{stem}_grating_fft_angle.png"
    direction_overlay_png = out_dir / f"{stem}_grating_measurement_direction.png"
    analysis_json = out_dir / f"{stem}_grating_angle.json"
    atomic_write_path(spectrum_npy, lambda path: np.save(path, log_power))
    opposite_fx = float(grid_fx[opposite_row, opposite_col])
    opposite_fy = float(grid_fy[opposite_row, opposite_col])
    physical_context = (
        physical_area_context_from_metadata(metadata) if metadata else {}
    )
    pixel_width_um = _pixel_size_micrometres(physical_context, "x")
    pixel_height_um = _pixel_size_micrometres(physical_context, "y")
    spectrum_plot_decorations = render_frequency_spectrum(
        log_power,
        freq_x,
        freq_y,
        direction_fx=direction_fx,
        direction_fy=direction_fy,
        peak_fx=peak_fx,
        peak_fy=peak_fy,
        opposite_fx=opposite_fx,
        opposite_fy=opposite_fy,
        profile_angle=profile_angle,
        output_path=spectrum_png,
        pixel_width_um=pixel_width_um,
        pixel_height_um=pixel_height_um,
    )

    unit_x = float(np.cos(np.deg2rad(profile_angle)))
    unit_y = float(np.sin(np.deg2rad(profile_angle)))
    direction_plot_decorations = render_measurement_direction(
        matrix,
        unit_x=unit_x,
        unit_y=unit_y,
        profile_angle=profile_angle,
        physical_context=physical_context,
        height_unit=height_unit(metadata) if metadata else "μm",
        output_path=direction_overlay_png,
    )

    payload: dict[str, Any] = {
        "status": "ok",
        "step": "analyze_periodic_orientation",
        "input_path": str(resolved_input),
        "matrix_shape": [int(rows), int(cols)],
        "metadata_summary": metadata_summary(metadata) if metadata else {},
        "physical_context": physical_context,
        "angle_convention": (
            "profile_angle_degrees is the height-variation direction normal "
            "to grating stripes; 0 degrees follows image X and 90 degrees "
            "follows image Y"
        ),
        "profile_angle_degrees": profile_angle,
        "stripe_angle_degrees": stripe_angle,
        "fft_discrete_angle_degrees": discrete_peak_angle,
        "fft_centroid_angle_degrees": fft_centroid_angle,
        "angle_refinement": angle_refinement,
        "spectrum_plot_decorations": spectrum_plot_decorations,
        "measurement_direction_vector_xy": [unit_x, unit_y],
        "measurement_direction_frequency": {
            "fx_cycles_per_px": direction_fx,
            "fy_cycles_per_px": direction_fy,
            "period_px": direction_period_px,
        },
        "dominant_frequency": {
            "source": "fft_peak_centroid",
            **centroid_frequency,
        },
        "discrete_peak_frequency": {
            "fx_cycles_per_px": peak_fx,
            "fy_cycles_per_px": peak_fy,
            "radial_frequency_cycles_per_px": float(
                radial_frequency[peak_row, peak_col]
            ),
            "period_px": peak_period_px,
        },
        "period_search": {
            "min_period_px": min_period,
            "max_period_px": effective_max_period,
        },
        "peak": {
            "row_index": int(peak_row),
            "col_index": int(peak_col),
            "power": peak_power,
            "log_power": float(log_power[peak_row, peak_col]),
        },
        "opposite_peak": {
            "row_index": int(opposite_row),
            "col_index": int(opposite_col),
            "fx_cycles_per_px": float(grid_fx[opposite_row, opposite_col]),
            "fy_cycles_per_px": float(grid_fy[opposite_row, opposite_col]),
            "power": opposite_power,
            "log_power": float(log_power[opposite_row, opposite_col]),
        },
        "quality": {
            "finite_pixel_count": finite_count,
            "candidate_frequency_bin_count": int(np.sum(valid)),
            "spectral_strength": spectral_strength,
            "peak_to_median_power_ratio": peak_to_median,
            "peak_to_mean_power_ratio": peak_to_mean,
            "opposite_peak_power_balance": pair_balance,
            "flags": quality_flags,
        },
        "measurement_direction_plot_decorations": direction_plot_decorations,
        "artifacts": {
            "analysis_json_path": analysis_json,
            "spectrum_plot_path": spectrum_png,
            "measurement_direction_overlay_path": direction_overlay_png,
            "spectrum_log_power_npy_path": spectrum_npy,
            "metadata_path": str(
                resolve_matrix_metadata_path(resolved_input, metadata_path) or ""
            ),
        },
    }
    write_json_file(analysis_json, payload)
    return payload


__all__ = ["analyze_periodic_orientation"]

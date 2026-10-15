"""Periodic profile measurement workflow orchestration."""

from __future__ import annotations

from typing import Any

import numpy as np

from afm_kite_core.algorithms.periodic import (
    dominant_period,
    median_diff,
    profile_extrema,
    profile_segments,
    segment_summary,
)
from afm_kite_core.serialization import to_jsonable
from afm_kite_core.artifacts.writers import (
    write_json_file,
)
from afm_kite_core.data_io.paths import ensure_output_dir

from .profile_io import load_profile_data
from .rendering import render_periodic_profile
from .reports import write_periodic_profile_report, write_segments_table


def measure_periodic_profile(
    profile_data_path: str,
    output_dir: str,
    high_percentile: float,
    low_percentile: float,
    min_period_px: float,
    max_period_px: float,
    pixel_size: float,
    axis_unit: str,
    height_unit: str,
    coordinate_column: str,
    height_column: str,
) -> dict[str, Any]:
    """Measure period, widths, duty cycle, and depth from a 1D profile."""

    coordinates, profile, resolved_input, profile_metadata = load_profile_data(
        profile_data_path,
        coordinate_column,
        height_column,
    )
    if coordinates.shape != profile.shape:
        raise ValueError(
            "profile coordinates and height values must have the same length"
        )
    finite = np.isfinite(coordinates) & np.isfinite(profile)
    if int(np.sum(finite)) < 4:
        raise ValueError("profile must contain at least four finite samples")

    coordinates = np.asarray(coordinates[finite], dtype=float)
    profile = np.asarray(profile[finite], dtype=float)
    order = np.argsort(coordinates)
    coordinates = coordinates[order]
    profile = profile[order]

    period = dominant_period(profile, min_period_px, max_period_px)
    extrema = profile_extrema(profile, period.get("period_px"))
    high_q = float(np.nanpercentile(profile, float(high_percentile)))
    low_q = float(np.nanpercentile(profile, float(low_percentile)))
    threshold = (high_q + low_q) / 2.0
    high_mask = profile >= threshold
    segments = profile_segments(
        profile,
        high_mask,
        float(pixel_size) if pixel_size > 0 else None,
    )
    high_summary = segment_summary(segments, "high")
    low_summary = segment_summary(segments, "low")

    high_period = median_diff(high_summary["centers_px"])
    low_period = median_diff(low_summary["centers_px"])
    period_values = [
        value
        for value in (high_period, low_period)
        if value is not None and value > 0
    ]
    if period_values:
        period_px = float(np.median(period_values))
    else:
        period_guess = period.get("period_px")
        period_px = float(period_guess) if period_guess else None

    high_level = high_summary.get("median_height")
    low_level = low_summary.get("median_height")
    depth_median = (
        float(high_level) - float(low_level)
        if high_level is not None and low_level is not None
        else None
    )
    high_width_px = high_summary.get("median_width_px")
    low_width_px = low_summary.get("median_width_px")
    high_width = high_summary.get("median_width_physical")
    low_width = low_summary.get("median_width_physical")
    period_physical = (
        float(period_px * pixel_size) if period_px and pixel_size > 0 else None
    )
    duty_cycle_high = (
        float(high_width_px / period_px)
        if high_width_px is not None and period_px
        else None
    )
    duty_cycle_low = (
        float(low_width_px / period_px)
        if low_width_px is not None and period_px
        else None
    )

    out_dir = ensure_output_dir(output_dir)
    stem = resolved_input.stem
    analysis_json = out_dir / f"{stem}_periodic_profile_analysis.json"
    report_md = out_dir / f"{stem}_periodic_profile_report.md"
    segments_tsv = out_dir / f"{stem}_periodic_profile_segments.tsv"
    plot_png = out_dir / f"{stem}_periodic_profile.png"

    write_segments_table(segments_tsv, segments)
    render_periodic_profile(
        coordinates,
        profile,
        threshold,
        extrema,
        axis_unit=axis_unit,
        height_unit=height_unit,
        output_path=plot_png,
    )

    payload: dict[str, Any] = {
        "status": "ok",
        "step": "measure_periodic_profile",
        "profile_data_path": str(resolved_input),
        "profile_metadata": profile_metadata,
        "profile_length": int(profile.size),
        "axis_unit": axis_unit,
        "height_unit": height_unit,
        "pixel_size": float(pixel_size),
        "profile_high_percentile": high_q,
        "profile_low_percentile": low_q,
        "profile_threshold": threshold,
        "dominant_period": period,
        "extrema": extrema,
        "measurement_summary": {
            "period_px": period_px,
            "period": period_physical,
            "period_unit": axis_unit,
            "period_from_high_centers_px": high_period,
            "period_from_low_centers_px": low_period,
            "high_width_px": high_width_px,
            "high_width": high_width,
            "low_width_px": low_width_px,
            "low_width": low_width,
            "depth_median_high_low": depth_median,
            "depth_percentile_high_low": float(high_q - low_q),
            "peak_to_valley_profile_range": float(
                np.nanmax(profile) - np.nanmin(profile)
            ),
            "high_level_median": high_level,
            "low_level_median": low_level,
            "high_segment_count": high_summary["count"],
            "low_segment_count": low_summary["count"],
            "duty_cycle_high": duty_cycle_high,
            "duty_cycle_low": duty_cycle_low,
        },
        "segments": segments,
        "artifacts": {
            "analysis_json_path": analysis_json,
            "report_md_path": report_md,
            "segments_tsv_path": segments_tsv,
            "profile_plot_path": plot_png,
        },
    }
    write_json_file(analysis_json, payload)
    write_periodic_profile_report(report_md, to_jsonable(payload))
    return payload


__all__ = ["measure_periodic_profile"]

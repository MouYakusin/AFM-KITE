"""Pure algorithms for periodic orientation and profile measurements."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage as ndi
from scipy.signal import find_peaks

from .profiles import projected_median_profile


def dominant_period(
    profile: np.ndarray,
    min_period_px: float,
    max_period_px: float,
) -> dict[str, Any]:
    """Estimate the strongest allowed one-dimensional FFT period."""

    values = np.asarray(profile, dtype=float)
    finite = np.isfinite(values)
    if int(np.sum(finite)) < 4:
        return {
            "period_px": None,
            "spectral_strength": 0.0,
            "frequency_cycles_per_px": None,
        }

    if not np.all(finite):
        x_idx = np.arange(values.size)
        values = values.copy()
        values[~finite] = np.interp(x_idx[~finite], x_idx[finite], values[finite])

    values = values - float(np.mean(values))
    if float(np.ptp(values)) <= 1e-12:
        return {
            "period_px": None,
            "spectral_strength": 0.0,
            "frequency_cycles_per_px": None,
        }

    n = values.size
    min_period = max(float(min_period_px), 2.0)
    max_period = float(max_period_px) if float(max_period_px) > 0 else n / 2.0
    max_period = min(max(max_period, min_period), n / 2.0)

    windowed = values * np.hanning(n)
    power = np.abs(np.fft.rfft(windowed)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0)
    periods = np.full_like(freqs, np.inf, dtype=float)
    periods[freqs > 0] = 1.0 / freqs[freqs > 0]
    mask = (freqs > 0) & (periods >= min_period) & (periods <= max_period)
    if not np.any(mask):
        return {
            "period_px": None,
            "spectral_strength": 0.0,
            "frequency_cycles_per_px": None,
        }

    masked_power = np.where(mask, power, -np.inf)
    best_index = int(np.argmax(masked_power))
    total_power = float(np.sum(power[mask]))
    best_power = float(power[best_index])
    return {
        "period_px": float(periods[best_index]),
        "spectral_strength": float(best_power / total_power) if total_power > 0 else 0.0,
        "frequency_cycles_per_px": float(freqs[best_index]),
    }


def wrap_angle_0_180(angle_degrees: float) -> float:
    """Normalize an undirected orientation angle to ``[0, 180)`` degrees."""

    wrapped = float(angle_degrees) % 180.0
    if np.isclose(wrapped, 180.0):
        return 0.0
    return wrapped


def centroid_fft_peak(
    power: np.ndarray,
    valid: np.ndarray,
    grid_fx: np.ndarray,
    grid_fy: np.ndarray,
    peak_row: int,
    peak_col: int,
    radius_px: int = 3,
) -> dict[str, Any]:
    """Refine a discrete two-dimensional FFT peak by local power centroiding."""

    rows, cols = power.shape
    radius = max(int(radius_px), 1)
    row_start = max(0, int(peak_row) - radius)
    row_stop = min(rows, int(peak_row) + radius + 1)
    col_start = max(0, int(peak_col) - radius)
    col_stop = min(cols, int(peak_col) + radius + 1)

    local_power = power[row_start:row_stop, col_start:col_stop]
    local_valid = valid[row_start:row_stop, col_start:col_stop]
    if not np.any(local_valid):
        fx = float(grid_fx[peak_row, peak_col])
        fy = float(grid_fy[peak_row, peak_col])
        radial = float(np.hypot(fx, fy))
        return {
            "fx_cycles_per_px": fx,
            "fy_cycles_per_px": fy,
            "radial_frequency_cycles_per_px": radial,
            "period_px": float(1.0 / radial) if radial > 0 else None,
            "window_radius_px": radius,
            "weighted_bin_count": 0,
        }

    baseline = float(np.percentile(local_power[local_valid], 25.0))
    weights = np.where(local_valid, np.maximum(local_power - baseline, 0.0), 0.0)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0:
        weights = np.where(local_valid, local_power, 0.0)
        weight_sum = float(np.sum(weights))
    if weight_sum <= 0:
        weights = local_valid.astype(float)
        weight_sum = float(np.sum(weights))

    local_fx = grid_fx[row_start:row_stop, col_start:col_stop]
    local_fy = grid_fy[row_start:row_stop, col_start:col_stop]
    fx = float(np.sum(local_fx * weights) / weight_sum)
    fy = float(np.sum(local_fy * weights) / weight_sum)
    radial = float(np.hypot(fx, fy))
    return {
        "fx_cycles_per_px": fx,
        "fy_cycles_per_px": fy,
        "radial_frequency_cycles_per_px": radial,
        "period_px": float(1.0 / radial) if radial > 0 else None,
        "window_radius_px": radius,
        "weighted_bin_count": int(np.sum(weights > 0)),
    }


def projected_angle_score(
    matrix: np.ndarray,
    angle_degrees: float,
    min_period_px: float,
    max_period_px: float,
) -> dict[str, Any]:
    """Score one projected-profile angle by contrast and spectral strength."""

    _, profile, _ = projected_median_profile(matrix, float(angle_degrees), 1.0)
    profile = np.asarray(profile, dtype=float)
    finite = np.isfinite(profile)
    if int(np.sum(finite)) < 4:
        return {
            "angle_degrees": float(angle_degrees),
            "score": 0.0,
            "dynamic_range": 0.0,
            "dominant_period": {
                "period_px": None,
                "spectral_strength": 0.0,
                "frequency_cycles_per_px": None,
            },
        }

    smoothed = ndi.gaussian_filter1d(profile[finite], sigma=1.0)
    dynamic_range = float(np.percentile(smoothed, 95.0) - np.percentile(smoothed, 5.0))
    period = dominant_period(smoothed, min_period_px, max_period_px)
    spectral_strength = float(period.get("spectral_strength") or 0.0)
    score = dynamic_range * max(spectral_strength, 0.02)
    return {
        "angle_degrees": float(angle_degrees),
        "score": float(score),
        "dynamic_range": dynamic_range,
        "dominant_period": period,
        "profile_length": int(smoothed.size),
    }


def refine_periodic_angle_by_projection(
    matrix: np.ndarray,
    initial_angle_degrees: float,
    min_period_px: float,
    max_period_px: float,
    search_half_width_degrees: float = 8.0,
    search_step_degrees: float = 0.25,
) -> dict[str, Any]:
    """Refine an initial orientation using a local projected-profile search."""

    half_width = max(float(search_half_width_degrees), 0.0)
    step = max(float(search_step_degrees), 0.05)
    offsets = np.arange(-half_width, half_width + (0.5 * step), step)
    candidates = [
        projected_angle_score(
            matrix,
            wrap_angle_0_180(float(initial_angle_degrees) + float(offset)),
            min_period_px,
            max_period_px,
        )
        for offset in offsets
    ]
    best = max(candidates, key=lambda item: float(item["score"]))
    top_candidates = sorted(
        candidates,
        key=lambda item: float(item["score"]),
        reverse=True,
    )[:5]
    return {
        "method": "local_projected_profile_search",
        "initial_angle_degrees": float(initial_angle_degrees),
        "selected_angle_degrees": float(best["angle_degrees"]),
        "search_half_width_degrees": half_width,
        "search_step_degrees": step,
        "best": best,
        "top_candidates": top_candidates,
    }


def profile_extrema(
    smoothed_profile: np.ndarray,
    period_guess_px: float | None,
) -> dict[str, Any]:
    """Locate robust profile peaks and valleys using adaptive prominence."""

    dynamic_range = float(np.nanmax(smoothed_profile) - np.nanmin(smoothed_profile))
    if dynamic_range <= 1e-12:
        return {"peaks": [], "valleys": [], "prominence": None, "distance_px": None}

    if period_guess_px and period_guess_px > 0:
        distance_px = max(2, int(round(0.45 * period_guess_px)))
    else:
        distance_px = max(2, int(round(smoothed_profile.size / 40.0)))

    selected_peaks: np.ndarray = np.asarray([], dtype=int)
    selected_valleys: np.ndarray = np.asarray([], dtype=int)
    selected_prominence: float | None = None
    for prominence_fraction in (0.15, 0.10, 0.06, 0.03):
        prominence = dynamic_range * prominence_fraction
        peaks, _ = find_peaks(smoothed_profile, prominence=prominence, distance=distance_px)
        valleys, _ = find_peaks(-smoothed_profile, prominence=prominence, distance=distance_px)
        selected_peaks = peaks
        selected_valleys = valleys
        selected_prominence = float(prominence)
        if len(peaks) >= 2 or len(valleys) >= 2:
            break

    return {
        "peaks": [int(item) for item in selected_peaks],
        "valleys": [int(item) for item in selected_valleys],
        "prominence": selected_prominence,
        "distance_px": int(distance_px),
    }


def profile_segments(
    profile: np.ndarray,
    high_mask: np.ndarray,
    pixel_size: float | None,
) -> list[dict[str, Any]]:
    """Summarize contiguous high and low runs in a thresholded profile."""

    segments: list[dict[str, Any]] = []
    n = int(high_mask.size)
    start = 0
    segment_index = 1

    while start < n:
        state = bool(high_mask[start])
        end = start + 1
        while end < n and bool(high_mask[end]) == state:
            end += 1

        values = profile[start:end]
        center = (start + end - 1) / 2.0
        width = end - start
        item: dict[str, Any] = {
            "segment_id": f"segment_{segment_index:03d}",
            "type": "high" if state else "low",
            "start_px": int(start),
            "end_px_exclusive": int(end),
            "center_px": float(center),
            "width_px": int(width),
            "edge_touched": bool(start == 0 or end == n),
            "median_height": float(np.nanmedian(values)),
            "mean_height": float(np.nanmean(values)),
            "min_height": float(np.nanmin(values)),
            "max_height": float(np.nanmax(values)),
        }
        if pixel_size:
            item.update(
                {
                    "start_physical": float(start * pixel_size),
                    "end_physical": float(end * pixel_size),
                    "center_physical": float(center * pixel_size),
                    "width_physical": float(width * pixel_size),
                }
            )
        segments.append(item)
        segment_index += 1
        start = end

    return segments


def median_diff(values: list[float]) -> float | None:
    """Return the median adjacent spacing, or ``None`` for fewer than two values."""

    if len(values) < 2:
        return None
    return float(np.median(np.diff(np.asarray(values, dtype=float))))


def segment_summary(
    segments: list[dict[str, Any]],
    segment_type: str,
) -> dict[str, Any]:
    """Aggregate widths, heights, and centers for one segment type."""

    candidates = [
        item
        for item in segments
        if item.get("type") == segment_type and not item.get("edge_touched")
    ]
    if not candidates:
        candidates = [item for item in segments if item.get("type") == segment_type]
    if not candidates:
        return {
            "count": 0,
            "median_width_px": None,
            "median_width_physical": None,
            "median_height": None,
            "centers_px": [],
        }

    widths = [float(item["width_px"]) for item in candidates]
    physical_widths = [
        float(item["width_physical"])
        for item in candidates
        if item.get("width_physical") is not None
    ]
    heights = [float(item["median_height"]) for item in candidates]
    centers = [float(item["center_px"]) for item in candidates]
    return {
        "count": len(candidates),
        "median_width_px": float(np.median(widths)),
        "median_width_physical": float(np.median(physical_widths)) if physical_widths else None,
        "median_height": float(np.median(heights)),
        "centers_px": centers,
    }


__all__ = [
    "centroid_fft_peak",
    "dominant_period",
    "median_diff",
    "profile_extrema",
    "profile_segments",
    "projected_angle_score",
    "refine_periodic_angle_by_projection",
    "segment_summary",
    "wrap_angle_0_180",
]

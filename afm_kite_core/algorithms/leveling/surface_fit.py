"""Coordinate-aware plane and polynomial surface fitting."""

from __future__ import annotations

from typing import Any

import numpy as np


def mask_values(matrix: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    finite = np.isfinite(matrix)
    if mask is not None:
        finite &= np.asarray(mask, dtype=bool)
    return matrix[finite]


def _positive_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(result) or result <= 0:
        return None
    return result


def surface_coordinates(
    shape: tuple[int, int],
    physical_context: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    y_idx, x_idx = np.indices(shape, dtype=float)
    context = physical_context or {}
    pixel_width = _positive_float(context.get("pixel_width"))
    pixel_height = _positive_float(context.get("pixel_height"))

    if pixel_width is not None and pixel_height is not None:
        x_coord = x_idx * pixel_width
        y_coord = y_idx * pixel_height
        coordinate_context = {
            "coordinate_system": "physical",
            "x_unit": context.get("x_unit", ""),
            "y_unit": context.get("y_unit", ""),
            "height_unit": context.get("height_unit", ""),
            "x_step": pixel_width,
            "y_step": pixel_height,
            "x_min": float(np.min(x_coord)),
            "x_max": float(np.max(x_coord)),
            "y_min": float(np.min(y_coord)),
            "y_max": float(np.max(y_coord)),
            "x_resolution": context.get("x_resolution"),
            "y_resolution": context.get("y_resolution"),
            "x_size": context.get("x_size"),
            "y_size": context.get("y_size"),
        }
        return y_coord, x_coord, coordinate_context

    coordinate_context = {
        "coordinate_system": "pixel_index",
        "reason": "missing_or_invalid_physical_pixel_size",
        "x_unit": "px",
        "y_unit": "px",
        "height_unit": context.get("height_unit", ""),
        "x_step": 1.0,
        "y_step": 1.0,
        "x_min": float(np.min(x_idx)),
        "x_max": float(np.max(x_idx)),
        "y_min": float(np.min(y_idx)),
        "y_max": float(np.max(y_idx)),
        "x_resolution": shape[1],
        "y_resolution": shape[0],
    }
    return y_idx, x_idx, coordinate_context


def _coefficient_units(
    names: list[str],
    coordinate_context: dict[str, Any],
) -> dict[str, str]:
    height_unit = str(coordinate_context.get("height_unit") or "height")
    x_unit = str(coordinate_context.get("x_unit") or "x")
    y_unit = str(coordinate_context.get("y_unit") or "y")
    units: dict[str, str] = {}
    for name in names:
        if name in {"x_slope", "x"}:
            units[name] = f"{height_unit}/{x_unit}"
        elif name in {"y_slope", "y"}:
            units[name] = f"{height_unit}/{y_unit}"
        elif name == "x2":
            units[name] = f"{height_unit}/{x_unit}^2"
        elif name == "y2":
            units[name] = f"{height_unit}/{y_unit}^2"
        elif name == "xy":
            units[name] = f"{height_unit}/({x_unit}*{y_unit})"
        else:
            units[name] = height_unit
    return units


def _surface_design(
    shape: tuple[int, int],
    model: str,
    physical_context: dict[str, Any] | None = None,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    y_idx, x_idx, coordinate_context = surface_coordinates(shape, physical_context)
    if model == "plane":
        return np.stack([x_idx, y_idx, np.ones(shape, dtype=float)], axis=-1), [
            "x_slope",
            "y_slope",
            "offset",
        ], coordinate_context
    if model == "poly2":
        return np.stack(
            [
                x_idx**2,
                y_idx**2,
                x_idx * y_idx,
                x_idx,
                y_idx,
                np.ones(shape, dtype=float),
            ],
            axis=-1,
        ), ["x2", "y2", "xy", "x", "y", "offset"], coordinate_context
    raise ValueError("surface model must be plane or poly2")


def fit_surface(
    matrix: np.ndarray,
    mask: np.ndarray | None,
    model: str,
    physical_context: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    matrix = np.asarray(matrix, dtype=float)
    design, names, coordinate_context = _surface_design(
        matrix.shape,
        model,
        physical_context,
    )
    valid = np.isfinite(matrix)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)

    min_points = max(len(names) * 3, len(names) + 1)
    if int(valid.sum()) < min_points:
        return np.zeros_like(matrix, dtype=float), {
            "model": model,
            "fit_status": "skipped_insufficient_points",
            "fit_points": int(valid.sum()),
            "min_points": int(min_points),
            "coordinate_context": coordinate_context,
        }

    fit_design = design[valid].reshape(int(valid.sum()), len(names))
    coeff, *_ = np.linalg.lstsq(fit_design, matrix[valid], rcond=None)
    surface = np.tensordot(design, coeff, axes=([-1], [0]))
    return surface, {
        "model": model,
        "fit_status": "ok",
        "fit_points": int(valid.sum()),
        "coefficients": {name: float(value) for name, value in zip(names, coeff)},
        "coefficient_units": _coefficient_units(names, coordinate_context),
        "coordinate_context": coordinate_context,
        "surface_range": float(np.max(surface) - np.min(surface)),
    }


def subtract_surface(
    matrix: np.ndarray,
    mask: np.ndarray | None,
    model: str,
    physical_context: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    surface, diagnostics = fit_surface(matrix, mask, model, physical_context)
    return matrix - surface, diagnostics


__all__ = ["fit_surface", "mask_values", "subtract_surface", "surface_coordinates"]

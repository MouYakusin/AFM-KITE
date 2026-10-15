"""Shared line-profile artifact contract helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from afm_kite_core.contracts import coordinate_line_id
from afm_kite_core.artifacts.writers import safe_stem


def safe_profile_name(value: str) -> str:
    return safe_stem(value, fallback="profile")


def line_id(payload: dict[str, Any]) -> str:
    existing = payload.get("line_id")
    if existing:
        return str(existing)
    coordinates = payload.get("coordinates", {})
    return coordinate_line_id(
        payload.get("shape", []),
        coordinates.get("x_px", []),
        coordinates.get("y_px", []),
    )


def window_indices(
    payload: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    windows = payload["windows"]
    left = np.asarray(windows["left_substrate_indices"], dtype=int)
    feature = np.asarray(windows["feature_window_indices"], dtype=int)
    right = np.asarray(windows["right_substrate_indices"], dtype=int)
    return left, feature, right


def sampling_contract(payload: dict[str, Any]) -> dict[str, Any]:
    """Return normalized band-sampling fields for profile checks."""

    sampling = payload.get("sampling", {}) or {}
    width = int(sampling.get("line_width_px", 1))
    return {
        "line_width_px": width,
        "aggregation": str(sampling.get("aggregation", "median")),
        "interpolation": str(sampling.get("interpolation", "linear")),
        "offsets_px": list(
            sampling.get("offsets_px", [0.0] if width == 1 else [])
        ),
    }


def validate_compatible_profiles(
    profiles: Sequence[tuple[Path, dict[str, Any]]],
) -> None:
    first = profiles[0][1]
    first_line_id = str(first.get("line_id", ""))
    first_x = np.asarray(first["coordinates"]["x_px"], dtype=float)
    first_y = np.asarray(first["coordinates"]["y_px"], dtype=float)
    first_windows = first["windows"]
    for path, payload in profiles[1:]:
        if str(payload.get("line_id", "")) != first_line_id:
            raise ValueError(f"Profile uses a different line_id: {path}")
        x_px = np.asarray(payload["coordinates"]["x_px"], dtype=float)
        y_px = np.asarray(payload["coordinates"]["y_px"], dtype=float)
        if x_px.shape != first_x.shape or y_px.shape != first_y.shape:
            raise ValueError(f"Profile coordinate count does not match: {path}")
        if not np.allclose(x_px, first_x) or not np.allclose(y_px, first_y):
            raise ValueError(f"Profile coordinates do not match the shared line: {path}")
        for key in (
            "left_substrate_indices",
            "feature_window_indices",
            "right_substrate_indices",
        ):
            if list(payload["windows"][key]) != list(first_windows[key]):
                raise ValueError(f"Profile window {key} does not match: {path}")
        if sampling_contract(payload) != sampling_contract(first):
            raise ValueError(f"Profile sampling contract does not match: {path}")


__all__ = [
    "line_id",
    "safe_profile_name",
    "sampling_contract",
    "validate_compatible_profiles",
    "window_indices",
]

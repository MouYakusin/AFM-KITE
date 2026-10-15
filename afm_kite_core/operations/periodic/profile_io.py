"""Profile loading helpers for periodic measurements."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from afm_kite_core.data_io.paths import resolve_readable_file


def _choose_column(
    requested: str,
    preferred: list[str],
    fallback_index: int,
    names: list[str],
) -> str:
    if requested:
        if requested not in names:
            raise ValueError(
                f"column {requested!r} not found; available columns: {', '.join(names)}"
            )
        return requested
    for candidate in preferred:
        if candidate in names:
            return candidate
    return names[fallback_index]


def load_profile_data(
    profile_data_path: str,
    coordinate_column: str = "",
    height_column: str = "",
) -> tuple[np.ndarray, np.ndarray, Path, dict[str, Any]]:
    """Load an NPY or text profile and describe the selected source columns."""

    resolved = resolve_readable_file(profile_data_path)
    metadata: dict[str, Any] = {
        "format": resolved.suffix.lower().lstrip(".") or "text",
        "coordinate_column": coordinate_column or "auto",
        "height_column": height_column or "auto",
    }

    if resolved.suffix.lower() == ".npy":
        data = np.load(resolved)
        if data.ndim == 1:
            profile = np.asarray(data, dtype=float)
            coordinates = np.arange(profile.size, dtype=float)
            metadata.update({"source_columns": ["index", "value"]})
            return coordinates, profile, resolved, metadata
        if data.ndim == 2 and data.shape[1] >= 2:
            coordinates = np.asarray(data[:, 0], dtype=float)
            profile = np.asarray(data[:, -1], dtype=float)
            metadata.update(
                {"source_columns": ["column_0", f"column_{data.shape[1] - 1}"]}
            )
            return coordinates, profile, resolved, metadata
        raise ValueError(
            "NPY profile input must be 1D or a 2D table with at least two columns"
        )

    try:
        named = np.genfromtxt(
            resolved,
            delimiter="\t",
            names=True,
            dtype=float,
            encoding="utf-8",
        )
    except ValueError:
        named = np.asarray([])

    if getattr(named, "dtype", None) is not None and named.dtype.names:
        names = list(named.dtype.names)
        coordinate_name = _choose_column(
            coordinate_column,
            ["coordinate_px", "index_px", "coordinate"],
            0,
            names,
        )
        height_name = _choose_column(
            height_column,
            ["smoothed_profile_height", "profile_height", "height"],
            min(1, len(names) - 1),
            names,
        )
        coordinates = np.asarray(named[coordinate_name], dtype=float)
        profile = np.asarray(named[height_name], dtype=float)
        metadata.update(
            {
                "source_columns": names,
                "selected_coordinate_column": coordinate_name,
                "selected_height_column": height_name,
            }
        )
        return coordinates, profile, resolved, metadata

    data = np.loadtxt(resolved, delimiter=None)
    if data.ndim == 1:
        profile = np.asarray(data, dtype=float)
        coordinates = np.arange(profile.size, dtype=float)
        metadata.update({"source_columns": ["index", "value"]})
        return coordinates, profile, resolved, metadata
    if data.ndim == 2 and data.shape[1] >= 2:
        coordinates = np.asarray(data[:, 0], dtype=float)
        profile = np.asarray(data[:, -1], dtype=float)
        metadata.update(
            {"source_columns": ["column_0", f"column_{data.shape[1] - 1}"]}
        )
        return coordinates, profile, resolved, metadata
    raise ValueError(
        "Profile text input must be 1D values or a table with at least two columns"
    )


__all__ = ["load_profile_data"]

"""Residual-feature artifact preparation stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from afm_kite_core.contracts import RESIDUAL_PREPARATION_SCHEMA
from afm_kite_core.data_io.matrix import height_unit as get_height_unit
from afm_kite_core.data_io.matrix import (
    load_height_data,
    load_height_data_and_metadata,
    metadata_summary,
    physical_area_context_from_metadata,
)
from afm_kite_core.artifacts.writers import write_json_file
from afm_kite_core.data_io.paths import ensure_output_dir, resolve_readable_file


def write_defect_prepared_json(
    input_path: str,
    leveled_data_path: str,
    residual_data_path: str,
    output_dir: str,
    metadata_path: str,
    leveling: str,
    background_sigma_px: float,
    leveled_tsv_path: str,
    residual_tsv_path: str,
) -> dict[str, Any]:
    """Write the prepared defect-depth JSON contract from existing artifacts."""

    matrix, metadata, resolved_input = load_height_data_and_metadata(
        input_path,
        metadata_path,
    )
    leveled = load_height_data(leveled_data_path)
    residual = load_height_data(residual_data_path)
    if leveled.shape != residual.shape:
        raise ValueError(
            "Leveled and residual shapes must match, got "
            f"{leveled.shape} and {residual.shape}"
        )
    if matrix.shape != leveled.shape:
        raise ValueError(
            f"Input and leveled shapes must match, got {matrix.shape} and {leveled.shape}"
        )

    physical = physical_area_context_from_metadata(metadata) if metadata else {}
    out_dir = ensure_output_dir(output_dir)
    stem = resolved_input.stem
    finite = np.isfinite(leveled)
    if not np.any(finite):
        raise ValueError("Cannot write a prepared JSON for an all-NaN height matrix")

    height_unit = physical.get("height_unit") or (
        get_height_unit(metadata) if metadata else "μm"
    )
    area_unit = "μm^2"
    if (
        physical.get("x_unit")
        and physical.get("y_unit")
        and physical.get("x_unit") == physical.get("y_unit")
    ):
        area_unit = f"{physical['x_unit']}^2"

    leveled_path = resolve_readable_file(leveled_data_path)
    residual_path = resolve_readable_file(residual_data_path)
    leveled_tsv = (
        Path(leveled_tsv_path).expanduser().resolve()
        if leveled_tsv_path
        else leveled_path.with_suffix(".tsv")
    )
    residual_tsv = (
        Path(residual_tsv_path).expanduser().resolve()
        if residual_tsv_path
        else residual_path.with_suffix(".tsv")
    )
    prepared_json = out_dir / f"{stem}_defect_depth_prepared.json"

    payload: dict[str, Any] = {
        "status": "ok",
        "step": "write_defect_prepared_json",
        "schema": RESIDUAL_PREPARATION_SCHEMA,
        "input_path": str(resolved_input),
        "source_stem": stem,
        "matrix_shape": list(matrix.shape),
        "metadata_summary": metadata_summary(metadata) if metadata else {},
        "leveling": {"mode": leveling or "unspecified"},
        "height_unit": height_unit,
        "area_unit": area_unit,
        "physical_context": physical,
        "background_sigma_px": float(background_sigma_px),
        "global_height_range": float(
            np.max(leveled[finite]) - np.min(leveled[finite])
        ),
        "artifacts": {
            "leveled_npy_path": leveled_path,
            "leveled_tsv_path": leveled_tsv if leveled_tsv.exists() else "",
            "residual_npy_path": residual_path,
            "residual_tsv_path": residual_tsv if residual_tsv.exists() else "",
            "prepared_json_path": prepared_json,
        },
    }
    write_json_file(prepared_json, payload)
    return payload


__all__ = ["write_defect_prepared_json"]

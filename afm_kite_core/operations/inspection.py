"""Read-only matrix inspection stage implementation."""

from __future__ import annotations

from typing import Any

from afm_kite_core.algorithms.statistics import finite_matrix_summary
from afm_kite_core.data_io.matrix import (
    height_unit,
    load_height_data_and_metadata,
    metadata_summary,
)


def inspect_height_data(input_path: str, metadata_path: str) -> dict[str, Any]:
    """Inspect an AFM matrix and metadata without generating image artifacts."""

    matrix, metadata, resolved_input = load_height_data_and_metadata(
        input_path,
        metadata_path,
    )
    payload = {
        "status": "ok",
        "step": "inspect_height_data",
        "input_path": str(resolved_input),
        "matrix": finite_matrix_summary(matrix),
        "metadata_summary": metadata_summary(metadata) if metadata else {},
        "height_unit": height_unit(metadata) if metadata else "μm",
    }
    return payload


__all__ = ["inspect_height_data"]

"""Public data-ingestion and matrix-inspection tools."""

from __future__ import annotations

from afm_kite_core.tools.boundary import JsonDict, error_result, invoke_operation


def data_ingest(
    input_path: str,
    output_dir: str = "outputs/data/ingest",
) -> JsonDict:
    """Parse one supported raw AFM/KPFM export and save every detected channel.

    A user-supplied ``.npy`` matrix is not a raw-export workflow entry point.
    """
    if not str(input_path).strip():
        return error_result("data-ingest", "input_path is required")

    from afm_kite_core.operations.ingest.workflow import read_afm_data

    return invoke_operation(
        "data-ingest",
        read_afm_data,
        input_path=input_path,
        output_dir=output_dir,
    )


def matrix_inspect(input_path: str, metadata_path: str = "") -> JsonDict:
    """Inspect matrix shape, finite coverage, value range, unit, and metadata."""
    from afm_kite_core.operations.inspection import inspect_height_data

    return invoke_operation(
        "matrix-inspect",
        inspect_height_data,
        input_path=input_path,
        metadata_path=metadata_path,
    )


__all__ = ["data_ingest", "matrix_inspect"]

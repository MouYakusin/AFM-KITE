"""AFM matrix, metadata, and channel input helpers."""

from .channels import parse_afm_channels
from .matrix import (
    DEFAULT_PHYSICAL_UNIT,
    DEFAULT_PIXEL_SIZE,
    height_unit,
    load_height_data,
    load_height_data_and_metadata,
    metadata_sidecar_path,
    metadata_summary,
    metadata_with_defaults,
    parse_afm_txt,
    physical_area_context,
    physical_area_context_from_metadata,
    resolve_matrix_metadata_path,
)
from .paths import ensure_output_dir, resolve_readable_file

__all__ = [
    "ensure_output_dir",
    "DEFAULT_PHYSICAL_UNIT",
    "DEFAULT_PIXEL_SIZE",
    "height_unit",
    "load_height_data",
    "load_height_data_and_metadata",
    "metadata_sidecar_path",
    "metadata_summary",
    "metadata_with_defaults",
    "parse_afm_channels",
    "parse_afm_txt",
    "physical_area_context",
    "physical_area_context_from_metadata",
    "resolve_matrix_metadata_path",
    "resolve_readable_file",
]

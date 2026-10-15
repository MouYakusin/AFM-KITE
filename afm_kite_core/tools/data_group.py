"""Public multi-file data relationship tool."""

from __future__ import annotations

from typing import Optional

from afm_kite_core.contracts import DataRelationship, RegistrationMode, RelationshipSource
from afm_kite_core.tools.boundary import JsonDict, error_result, invoke_operation


def data_group_build(
    data_manifest_paths: list[str],
    output_dir: str = "outputs/data/data_group",
    relationship: DataRelationship = "collection",
    labels: Optional[list[str]] = None,
    reference_data_index: int = 0,
    registration: RegistrationMode = "none",
    registration_channel: str = "height",
    relationship_source: RelationshipSource = "agent_default",
) -> JsonDict:
    """Organize two or more single-file ingest manifests into one data group."""
    if len(data_manifest_paths) < 2:
        return error_result(
            "data-group-build", "data_manifest_paths must contain at least two paths"
        )
    from afm_kite_core.operations.grouping.workflow import build_data_group

    return invoke_operation(
        "data-group-build",
        build_data_group,
        data_manifest_paths=data_manifest_paths,
        output_dir=output_dir,
        relationship=relationship,
        labels=labels,
        reference_data_index=reference_data_index,
        registration=registration,
        registration_channel=registration_channel,
        relationship_source=relationship_source,
    )


__all__ = ["data_group_build"]

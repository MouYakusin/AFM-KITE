"""Human-readable residual-feature report generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from afm_kite_core.artifacts.writers import write_text_file


def write_defect_depth_report(report_path: Path, payload: dict[str, Any]) -> None:
    height_unit = payload.get("height_unit", "μm")
    area_unit = payload.get("area_unit", "μm^2")
    leveling_mode = payload.get("leveling", {}).get("mode", "unspecified")
    lines = [
        "# AFM Defect Depth Report",
        "",
        f"- Input: `{payload['input_path']}`",
        f"- Matrix shape: {payload['matrix_shape']}",
        f"- Leveling: {leveling_mode}",
        f"- Defect polarity: {payload['polarity']}",
        f"- Candidate count: {payload.get('candidate_count', 0)}",
        f"- Selection mode: {payload.get('selection_mode', 'all')}",
        (
            "- Selected candidate IDs: "
            f"{', '.join(payload.get('selected_candidate_ids', [])) or '<none>'}"
        ),
        f"- Height unit: {height_unit}",
        f"- Area unit: {area_unit}",
        (
            "- Depth convention: positive depth is local reference median minus "
            "defect minimum for dark defects, or defect maximum minus local "
            "reference median for bright defects."
        ),
        "",
        "## Defects",
        "",
        (
            f"| defect | type | bbox_xyxy | centroid_yx | centroid_xy | "
            f"area_px | area_{area_unit} | max_depth_{height_unit} | "
            f"mean_depth_{height_unit} | median_depth_{height_unit} | "
            f"reference_median_{height_unit} | extremum_{height_unit} |"
        ),
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    defects = payload.get("defects", [])
    if not defects:
        lines.append("| none |  |  |  |  |  |  |  |  |  |  |  |")
    else:
        for defect in defects:
            lines.append(
                (
                    "| {defect_id} | {type} | {bbox_xyxy} | {centroid_yx} | "
                    "{centroid_xy_physical} | {area_px} | {area_physical} | "
                    "{max_depth} | {mean_depth} | {median_depth} | "
                    "{local_reference_median} | {extremum_height} |"
                ).format(
                    defect_id=defect["defect_id"],
                    type=defect["type"],
                    bbox_xyxy=defect["bbox_xyxy"],
                    centroid_yx=defect["centroid_yx"],
                    centroid_xy_physical=defect.get("centroid_xy_physical"),
                    area_px=defect["area_px"],
                    area_physical=defect.get("area_physical"),
                    max_depth=defect["max_depth"],
                    mean_depth=defect["mean_depth"],
                    median_depth=defect["median_depth"],
                    local_reference_median=defect["local_reference_median"],
                    extremum_height=defect["extremum_height"],
                )
            )

    lines.extend(["", "## Artifacts", ""])
    for key, value in payload.get("artifacts", {}).items():
        lines.append(f"- {key}: `{value}`")
    write_text_file(report_path, "\n".join(lines) + "\n")


__all__ = ["write_defect_depth_report"]

"""Surface and region roughness stage implementations."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from afm_kite_core.algorithms.roughness import (
    masked_roughness_stats,
    normalise_roughness_scopes,
)
from afm_kite_core.algorithms.segmentation import (
    evaluate_value_segmentation_quality,
)
from afm_kite_core.algorithms.statistics import surface_stats
from afm_kite_core.contracts import LOCAL_OUTLIERS_SCHEMA, VALUE_REGIONS_SCHEMA
from afm_kite_core.data_io.matrix import height_unit as get_height_unit
from afm_kite_core.data_io.matrix import (
    load_height_data_and_metadata,
    physical_area_context_from_metadata,
)
from afm_kite_core.serialization import to_jsonable
from afm_kite_core.artifacts.writers import (
    read_json_file,
    write_json_file,
    write_text_file,
)
from afm_kite_core.data_io.paths import ensure_output_dir, resolve_readable_file


def _validated_segmentation_quality(
    regions_payload: dict[str, Any],
) -> dict[str, Any]:
    quality = regions_payload.get("quality")
    if isinstance(quality, dict) and isinstance(quality.get("passed"), bool):
        validated_quality = {
            "passed": quality["passed"],
            "flags": list(quality.get("flags") or []),
            "source": "recorded",
        }
    else:
        diagnostics = regions_payload.get("score_diagnostics")
        selected_score = regions_payload.get("selected_score")
        if not isinstance(diagnostics, dict) or selected_score is None:
            raise ValueError(
                "Region segmentation quality is unavailable; rerun value_regions_segment"
            )
        validated_quality = {
            **evaluate_value_segmentation_quality(
                float(selected_score),
                diagnostics,
            ),
            "source": "derived_from_legacy_diagnostics",
        }

    if not validated_quality["passed"]:
        flags = (
            ", ".join(validated_quality["flags"])
            or "unspecified_quality_failure"
        )
        raise ValueError(f"Region segmentation quality failed: {flags}")
    return validated_quality


def _write_report(
    report_path: Path,
    payload: dict[str, Any],
    unit: str,
    area_unit: str,
) -> None:
    height_unit = unit or "μm"
    area_label = area_unit or "μm^2"
    scopes = list(payload["scopes"])
    lines = [
        "# AFM Roughness Report",
        "",
        f"- Height data: `{payload['height_data_path']}`",
        f"- Regions: `{payload.get('regions_json_path') or '<not supplied>'}`",
        f"- Requested scopes: {', '.join(scopes)}",
        f"- Height unit: {height_unit}",
        f"- Area unit: {area_label}",
    ]
    stat_keys = [
        "mean_height",
        "median_height",
        "min_height",
        "max_height",
        "Sa",
        "Sq",
        "Sz",
        "Ssk",
        "Sku",
    ]

    if "surface" in scopes:
        lines.extend(["", "## Full Surface", ""])
        for key in stat_keys:
            lines.append(f"- {key}: {payload['full_surface'].get(key)}")

    if "background" in scopes:
        background = payload["background"]
        lines.extend(["", "## Background", ""])
        for key in stat_keys:
            lines.append(f"- {key}: {background.get(key)}")
        for key in [
            "contaminant_count",
            "contaminant_area_px",
            "contaminant_fraction",
            "clean_Sa",
            "clean_Sq",
            "clean_Sz",
        ]:
            lines.append(f"- {key}: {background.get(key)}")

    if "region_union" in scopes:
        region_union = payload["region_union"]
        lines.extend(["", "## Combined Segmented-Region Union", ""])
        lines.append(f"- region_count: {region_union.get('region_count')}")
        lines.append(f"- label_values: {region_union.get('label_values')}")
        lines.append(f"- area_px: {region_union.get('area_px')}")
        lines.append(f"- area_{area_label}: {region_union.get('area_physical')}")
        for key in stat_keys:
            lines.append(f"- {key}: {region_union.get(key)}")
        for key in [
            "contaminant_count",
            "contaminant_area_px",
            "contaminant_fraction",
            "clean_Sa",
            "clean_Sq",
            "clean_Sz",
        ]:
            lines.append(f"- {key}: {region_union.get(key)}")

    if "regions" in scopes:
        regions = payload["regions"]
        lines.extend(["", "## Per-Region Roughness Metrics", ""])
        if not regions:
            lines.append("No segmented target regions were found.")
        else:
            lines.extend(
                [
                    (
                        f"| region | label | bbox_xyxy | area_px | area_{area_label} | "
                        f"Sa_{height_unit} | Sq_{height_unit} | Sz_{height_unit} | "
                        f"Ssk | Sku | outlier_count | outlier_fraction | "
                        f"clean_Sa_{height_unit} | clean_Sq_{height_unit} | "
                        f"clean_Sz_{height_unit} |"
                    ),
                    "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for region in regions:
                lines.append(
                    (
                        "| {region_id} | {label_value} | {bbox_xyxy} | {area_px} | "
                        "{area_physical} | {Sa} | {Sq} | {Sz} | {Ssk} | {Sku} | "
                        "{contaminant_count} | {contaminant_fraction} | "
                        "{clean_Sa} | {clean_Sq} | {clean_Sz} |"
                    ).format(
                        region_id=region.get("region_id"),
                        label_value=region.get("label_value"),
                        bbox_xyxy=region.get("bbox_xyxy"),
                        area_px=region.get("area_px"),
                        area_physical=region.get("area_physical"),
                        Sa=region.get("Sa"),
                        Sq=region.get("Sq"),
                        Sz=region.get("Sz"),
                        Ssk=region.get("Ssk"),
                        Sku=region.get("Sku"),
                        contaminant_count=region.get("contaminant_count", 0),
                        contaminant_fraction=region.get("contaminant_fraction", 0.0),
                        clean_Sa=region.get("clean_Sa"),
                        clean_Sq=region.get("clean_Sq"),
                        clean_Sz=region.get("clean_Sz"),
                    )
                )

    if payload.get("contaminants_json_path"):
        contaminants = payload.get("contaminants", [])
        lines.extend(["", "## Local Outliers", ""])
        if not contaminants:
            lines.append("No local outliers were detected in the requested scopes.")
        else:
            lines.extend(
                [
                    "| outlier | region | type | bbox_xyxy | area_px | contrast | robust_z |",
                    "|---|---|---|---|---:|---:|---:|",
                ]
            )
            for contaminant in contaminants:
                lines.append(
                    "| {contaminant_id} | {region_id} | {type} | {bbox_xyxy} | "
                    "{area_px} | {height_contrast} | {max_robust_z} |".format(
                        contaminant_id=contaminant.get("contaminant_id"),
                        region_id=contaminant.get("region_id"),
                        type=contaminant.get("type"),
                        bbox_xyxy=contaminant.get("bbox_xyxy"),
                        area_px=contaminant.get("area_px"),
                        height_contrast=contaminant.get("height_contrast"),
                        max_robust_z=contaminant.get("max_robust_z"),
                    )
                )

    write_text_file(report_path, "\n".join(lines) + "\n")


def calculate_roughness(
    height_data_path: str,
    regions_json_path: str,
    output_dir: str,
    unit: str,
    metadata_path: str,
    contaminants_json_path: str,
    scopes: Optional[Union[list[str], tuple[str, ...]]],
) -> dict[str, Any]:
    """Calculate requested roughness scopes without leveling or filtering."""

    requested_scopes = normalise_roughness_scopes(scopes, regions_json_path)
    if contaminants_json_path and not regions_json_path:
        raise ValueError("--contaminants-json-path requires a regions_json_path")
    if contaminants_json_path and not set(requested_scopes).intersection(
        {"regions", "region_union", "background"}
    ):
        raise ValueError(
            "--contaminants-json-path requires regions, region_union, or "
            "background in scopes"
        )

    matrix, metadata, resolved_input = load_height_data_and_metadata(
        height_data_path,
        metadata_path,
    )
    out_dir = ensure_output_dir(output_dir)
    stem = resolved_input.stem
    metrics_json = out_dir / f"{stem}_roughness_metrics.json"
    report_md = out_dir / f"{stem}_roughness_report.md"

    physical = physical_area_context_from_metadata(metadata) if metadata else {}
    pixel_area = physical.get("pixel_area")
    area_unit = ""
    if physical.get("x_unit") and physical.get(
            "y_unit") and physical.get("x_unit") == physical.get("y_unit"):
        area_unit = f"{physical['x_unit']}^2"
    height_unit = unit or (
        get_height_unit(metadata) if metadata else physical.get("height_unit", "")
    )

    payload: dict[str, Any] = {
        "status": "ok",
        "step": "calculate_roughness",
        "scopes": requested_scopes,
        "height_data_path": str(resolved_input),
        "regions_json_path": "",
        "height_unit": height_unit or "μm",
        "area_unit": area_unit or "μm^2",
        "contaminants_json_path": "",
        "metrics_json_path": metrics_json,
        "report_md_path": report_md,
    }

    if "surface" in requested_scopes:
        full_surface_stats = surface_stats(matrix)
        payload["full_surface"] = full_surface_stats
        payload["surface_roughness"] = full_surface_stats

    if set(requested_scopes).intersection(
        {"regions", "region_union", "background"}
    ):
        regions_path, regions_payload = read_json_file(
            regions_json_path,
            expected_schema=VALUE_REGIONS_SCHEMA,
            required_keys=("label_npy_path", "regions"),
        )
        payload["segmentation_quality"] = _validated_segmentation_quality(
            regions_payload
        )
        label_image = np.load(
            resolve_readable_file(regions_payload["label_npy_path"])
        )
        if label_image.shape != matrix.shape:
            raise ValueError(
                f"Region label shape {label_image.shape} does not match "
                f"matrix shape {matrix.shape}"
            )
        payload["regions_json_path"] = str(regions_path)

        contaminants_payload: dict[str, Any] = {"contaminants": []}
        contaminant_label_image = np.zeros(matrix.shape, dtype=np.int32)
        contaminants_path = None
        if contaminants_json_path:
            contaminants_path, contaminants_payload = read_json_file(
                contaminants_json_path,
                expected_schema=LOCAL_OUTLIERS_SCHEMA,
                required_keys=("contaminant_label_npy_path", "contaminants"),
            )
            contaminant_label_image = np.load(
                resolve_readable_file(
                    contaminants_payload["contaminant_label_npy_path"]
                )
            )
            if contaminant_label_image.shape != matrix.shape:
                raise ValueError(
                    "Contaminant label shape "
                    f"{contaminant_label_image.shape} does not match "
                    f"matrix shape {matrix.shape}"
                )

        contaminants_by_region: dict[str, list[dict[str, Any]]] = {}
        for contaminant in contaminants_payload.get("contaminants", []):
            region_id = str(contaminant.get("region_id", ""))
            contaminants_by_region.setdefault(region_id, []).append(contaminant)

        selected_region_ids: set[str] = set()
        region_records = list(regions_payload.get("regions", []))
        if "regions" in requested_scopes:
            region_metrics: list[dict[str, Any]] = []
            for region in region_records:
                region_id = str(region["region_id"])
                selected_region_ids.add(region_id)
                label_value = int(region["label_value"])
                metrics = masked_roughness_stats(
                    matrix,
                    label_image == label_value,
                    contaminant_label_image,
                    contaminants_by_region.get(region_id, []),
                    pixel_area,
                )
                region_metrics.append(
                    {
                        **region,
                        **metrics,
                        "height_unit": height_unit or "μm",
                        "area_unit": area_unit or "μm^2",
                    }
                )
            payload["region_count"] = len(region_metrics)
            payload["regions"] = region_metrics

        if "region_union" in requested_scopes:
            union_region_ids = {
                str(region["region_id"])
                for region in region_records
            }
            selected_region_ids.update(union_region_ids)
            label_values = [
                int(value)
                for value in np.unique(label_image[label_image > 0])
            ]
            union_contaminants = [
                contaminant
                for region_id in union_region_ids
                for contaminant in contaminants_by_region.get(region_id, [])
            ]
            payload["region_union"] = {
                **masked_roughness_stats(
                    matrix,
                    label_image > 0,
                    contaminant_label_image,
                    union_contaminants,
                    pixel_area,
                ),
                "region_count": len(label_values),
                "label_values": label_values,
                "height_unit": height_unit or "μm",
                "area_unit": area_unit or "μm^2",
            }

        if "background" in requested_scopes:
            payload["background"] = {
                **masked_roughness_stats(
                    matrix,
                    label_image == 0,
                    contaminant_label_image,
                    contaminants_by_region.get("background", []),
                    pixel_area,
                ),
                "height_unit": height_unit or "μm",
                "area_unit": area_unit or "μm^2",
            }

        selected_contaminants = [
            contaminant
            for contaminant in contaminants_payload.get("contaminants", [])
            if (
                set(requested_scopes).intersection({"regions", "region_union"})
                and str(contaminant.get("region_id", "")) in selected_region_ids
            )
            or (
                "background" in requested_scopes
                and str(contaminant.get("region_id", "")) == "background"
            )
        ]
        payload["contaminant_count"] = len(selected_contaminants)
        payload["contaminants"] = selected_contaminants
        payload["contaminants_json_path"] = (
            str(contaminants_path) if contaminants_path else ""
        )

    write_json_file(metrics_json, payload)
    _write_report(report_md, to_jsonable(payload), height_unit, area_unit)
    return payload
